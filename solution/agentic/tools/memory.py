"""長期メモリ（udahub.db）の読み書き。設計書（agentic/design/architecture.md）の第7章。

短期メモリ（同じチケット内の会話）は LangGraph の MemorySaver が持つ。
ここでは、チケットをまたいで残す「長期メモリ」を扱う。

  CultPass の user_id
     │  udahub.users.external_user_id で対応づける
     ▼
  udahub.users ──< tickets ──┬── ticket_metadata（status, main_issue_type, tags）
                             └──< ticket_messages（role, content）

LLM がツールとして呼ぶのではなく、グラフのノード（load_memory / save_memory）が呼ぶ。
そのため @tool ラッパーは作らない。
返り値の形は cultpass_tools.py と同じ {"ok": ..., "data"/"error": ...}。

前提: solution/ フォルダが sys.path に入っていること
"""
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from data.models import udahub

# solution/agentic/tools/memory.py → parents[2] = solution/
SOLUTION_DIR = Path(__file__).resolve().parents[2]
UDAHUB_DB_PATH = SOLUTION_DIR / "data" / "core" / "udahub.db"

ACCOUNT_ID = "cultpass"                       # 最初の導入顧客
VALID_STATUSES = {"open", "resolved", "escalated"}
PAST_TICKET_LIMIT = 3                         # 過去のチケットは新しい順に最大3件
SNIPPET_LENGTH = 150                          # 要約に入れる発言の最大文字数

_engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}")
Session = sessionmaker(bind=_engine)


def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _db_missing() -> bool:
    # SQLite はファイルがないと空の DB を勝手に作ってしまうので、先に確認する
    return not UDAHUB_DB_PATH.exists()


def _short(text: str | None) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= SNIPPET_LENGTH else text[:SNIPPET_LENGTH] + "..."


# ---------------------------------------------------------------
# チケットの作成（03 ノートブックで、新しい問い合わせが来たときに使う）
# ---------------------------------------------------------------
def create_ticket(external_user_id: str, channel: str = "chat") -> dict:
    """チケットと ticket_metadata（status=open）を作り、ticket_id を返す。

    udahub.db にまだいないユーザーなら、ユーザーも作る。
    （user_name には ID を入れる。名前などの個人情報は増やさない）
    最初のメッセージはここでは保存しない。save_messages にまとめて保存させる。
    """
    if not external_user_id:
        return _error("external_user_id is required")
    if _db_missing():
        return _error(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    try:
        with Session() as session:
            user = session.query(udahub.User).filter_by(
                account_id=ACCOUNT_ID, external_user_id=external_user_id,
            ).first()
            if user is None:
                user = udahub.User(
                    user_id=str(uuid.uuid4()),
                    account_id=ACCOUNT_ID,
                    external_user_id=external_user_id,
                    user_name=external_user_id,
                )
                session.add(user)

            ticket = udahub.Ticket(
                ticket_id=str(uuid.uuid4()),
                account_id=ACCOUNT_ID,
                user_id=user.user_id,
                channel=channel,
                created_at=datetime.now(),
            )
            session.add_all([
                ticket,
                udahub.TicketMetadata(ticket_id=ticket.ticket_id, status="open"),
            ])
            session.commit()
            return _ok({"ticket_id": ticket.ticket_id})
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


# ---------------------------------------------------------------
# 読み込み（load_memory ノード）
# ---------------------------------------------------------------
def load_past_context(external_user_id: str, current_ticket_id: str | None = None,
                      limit: int = PAST_TICKET_LIMIT) -> dict:
    """同じユーザーの過去のチケットを新しい順に取り出し、短い文章にまとめる。

    data の中身:
        tickets : [{ticket_id, created_at, status, main_issue_type, tags,
                    first_user_message, last_ai_message}, ...]
        summary : LLM に渡す文章（= state の past_context）
    初めてのユーザーはエラーではない（tickets が空になる）。
    """
    if not external_user_id:
        return _error("external_user_id is required")
    if _db_missing():
        return _error(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    try:
        with Session() as session:
            query = (
                session.query(udahub.Ticket)
                .join(udahub.User)
                .filter(udahub.User.account_id == ACCOUNT_ID,
                        udahub.User.external_user_id == external_user_id)
            )
            if current_ticket_id:
                query = query.filter(udahub.Ticket.ticket_id != current_ticket_id)
            past = query.order_by(udahub.Ticket.created_at.desc()).limit(limit).all()

            tickets = []
            for t in past:
                messages = sorted(t.messages, key=lambda m: m.created_at)
                user_msgs = [m.content for m in messages if m.role == udahub.RoleEnum.user]
                ai_msgs = [m.content for m in messages if m.role == udahub.RoleEnum.ai]
                meta = t.ticket_metadata
                tickets.append({
                    "ticket_id": t.ticket_id,
                    "created_at": t.created_at.strftime("%Y-%m-%d") if t.created_at else "",
                    "status": meta.status if meta else "unknown",
                    "main_issue_type": meta.main_issue_type if meta else None,
                    "tags": meta.tags if meta else None,
                    "first_user_message": _short(user_msgs[0]) if user_msgs else "",
                    "last_ai_message": _short(ai_msgs[-1]) if ai_msgs else "",
                })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")

    return _ok({"tickets": tickets, "summary": _format_summary(tickets)})


def _format_summary(tickets: list[dict]) -> str:
    """例: - 2026-11-10 [resolved] technical (qr_code): "The QR code..." → "Please..." """
    if not tickets:
        return "No past tickets from this customer."

    lines = ["Past tickets from this customer (newest first):"]
    for t in tickets:
        line = f"- {t['created_at']} [{t['status']}]"
        if t["main_issue_type"]:
            line += f" {t['main_issue_type']}"
        if t["tags"]:
            line += f" ({t['tags']})"
        line += f': "{t["first_user_message"]}"'
        if t["last_ai_message"]:
            line += f' → "{t["last_ai_message"]}"'
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------
# 保存（save_memory ノード）
# ---------------------------------------------------------------
def save_messages(ticket_id: str, messages: list[BaseMessage]) -> dict:
    """LangChain のメッセージを ticket_messages に保存する。

    - 保存するのは「ユーザーの発言」と「本文のある AI の発言」だけ。
      ツールの呼び出しや結果は、短期メモリ（MemorySaver）で確認できるので入れない。
    - message.id がすでに保存されていれば飛ばす（同じチケットで会話が続いても二重に保存しない）。
    """
    if not ticket_id:
        return _error("ticket_id is required")
    if _db_missing():
        return _error(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    try:
        with Session() as session:
            if session.get(udahub.Ticket, ticket_id) is None:
                return _error(f"Ticket not found: {ticket_id}")

            saved_ids = {
                mid for (mid,) in session.query(udahub.TicketMessage.message_id)
                .filter_by(ticket_id=ticket_id)
            }

            # 同じ秒に保存しても順番がわかるように、時刻を1マイクロ秒ずつずらす
            now = datetime.now()
            count = 0
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = udahub.RoleEnum.user
                elif isinstance(msg, AIMessage) and msg.text.strip():
                    role = udahub.RoleEnum.ai
                else:
                    continue  # ツールの呼び出しだけの AI メッセージ、ToolMessage、SystemMessage

                message_id = msg.id or str(uuid.uuid4())
                if message_id in saved_ids:
                    continue

                session.add(udahub.TicketMessage(
                    message_id=message_id,
                    ticket_id=ticket_id,
                    role=role,
                    content=msg.text,
                    created_at=now + timedelta(microseconds=count),
                ))
                saved_ids.add(message_id)
                count += 1

            session.commit()
            return _ok({"ticket_id": ticket_id, "saved": count})
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def update_ticket_status(ticket_id: str, status: str,
                         main_issue_type: str | None = None,
                         tags: str | None = None) -> dict:
    """チケットの結果（resolved / escalated）と分類を ticket_metadata に保存する。

    main_issue_type と tags は、渡されたときだけ上書きする。
    """
    if not ticket_id:
        return _error("ticket_id is required")
    if status not in VALID_STATUSES:
        return _error(f"Invalid status: {status} (use one of {sorted(VALID_STATUSES)})")
    if _db_missing():
        return _error(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    try:
        with Session() as session:
            meta = session.get(udahub.TicketMetadata, ticket_id)
            if meta is None:
                return _error(f"Ticket not found: {ticket_id}")

            meta.status = status
            if main_issue_type is not None:
                meta.main_issue_type = main_issue_type
            if tags is not None:
                meta.tags = tags
            session.commit()
            return _ok({
                "ticket_id": ticket_id,
                "status": meta.status,
                "main_issue_type": meta.main_issue_type,
                "tags": meta.tags,
            })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")
