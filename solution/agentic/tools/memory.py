"""Read and write long-term memory (udahub.db). Design doc (agentic/design/architecture.md), Chapter 7.

Short-term memory (the conversation within one ticket) is held by LangGraph's MemorySaver.
This module handles "long-term memory" that persists across tickets.

  CultPass user_id
     │  mapped via udahub.users.external_user_id
     ▼
  udahub.users ──< tickets ──┬── ticket_metadata (status, main_issue_type, tags)
                             └──< ticket_messages (role, content)

These are called by graph nodes (load_memory / save_memory), not by the LLM as tools.
So there are no @tool wrappers.
Return values have the same shape as cultpass_tools.py: {"ok": ..., "data"/"error": ...}.

Prerequisite: the solution/ folder is on sys.path
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

ACCOUNT_ID = "cultpass"                       # first customer
VALID_STATUSES = {"open", "resolved", "escalated"}
PAST_TICKET_LIMIT = 3                         # up to 3 past tickets, newest first
SNIPPET_LENGTH = 150                          # max characters per message in the summary

_engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}")
Session = sessionmaker(bind=_engine)


def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _db_missing() -> bool:
    # SQLite silently creates an empty DB if the file is missing, so check first
    return not UDAHUB_DB_PATH.exists()


def _short(text: str | None) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= SNIPPET_LENGTH else text[:SNIPPET_LENGTH] + "..."


# ---------------------------------------------------------------
# Ticket creation (used in notebook 03 when a new inquiry arrives)
# ---------------------------------------------------------------
def create_ticket(external_user_id: str, channel: str = "chat") -> dict:
    """Create a ticket and its ticket_metadata (status=open), and return the ticket_id.

    If the user is not in udahub.db yet, create the user as well.
    (user_name is set to the ID, so no extra personal data such as names is stored.)
    The first message is not saved here; save_messages saves all messages together.
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
# Loading (load_memory node)
# ---------------------------------------------------------------
def get_ticket_info(ticket_id: str) -> dict:
    """Return the ticket owner (CultPass user_id) and its metadata.

    Used by load_memory to get the user_id from the thread_id (= ticket_id).
    (because chat_interface only passes messages)

    Contents of data:
        ticket_id, external_user_id, channel, created_at, status, main_issue_type, tags
    """
    if not ticket_id:
        return _error("ticket_id is required")
    if _db_missing():
        return _error(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    try:
        with Session() as session:
            ticket = session.get(udahub.Ticket, ticket_id)
            if ticket is None:
                return _error(f"Ticket not found: {ticket_id}")
            meta = ticket.ticket_metadata
            return _ok({
                "ticket_id": ticket.ticket_id,
                "external_user_id": ticket.user.external_user_id,
                "channel": ticket.channel,
                "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
                "status": meta.status if meta else None,
                "main_issue_type": meta.main_issue_type if meta else None,
                "tags": meta.tags if meta else None,
            })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def load_past_context(external_user_id: str, current_ticket_id: str | None = None,
                      limit: int = PAST_TICKET_LIMIT) -> dict:
    """Fetch the same user's past tickets, newest first, and summarize them as short text.

    Contents of data:
        tickets : [{ticket_id, created_at, status, main_issue_type, tags,
                    first_user_message, last_ai_message}, ...]
        summary : text passed to the LLM (= past_context in the state)
    A first-time user is not an error (tickets is just empty).
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
    """Example: - 2026-11-10 [resolved] technical (qr_code): "The QR code..." → "Please..." """
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
# Saving (save_memory node)
# ---------------------------------------------------------------
def save_messages(ticket_id: str, messages: list[BaseMessage]) -> dict:
    """Save LangChain messages to ticket_messages.

    - Only "user messages" and "AI messages with text content" are saved.
      Tool calls and results are skipped because they can be checked in short-term memory (MemorySaver).
    - Skip messages whose message.id is already saved (no duplicates when a conversation continues in the same ticket).
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

            # Shift each timestamp by 1 microsecond so the order is kept even within the same second
            now = datetime.now()
            count = 0
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = udahub.RoleEnum.user
                elif isinstance(msg, AIMessage) and msg.text.strip():
                    role = udahub.RoleEnum.ai
                else:
                    continue  # AI messages with only tool calls, ToolMessage, SystemMessage

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


def save_handoff_summary(ticket_id: str, summary: str) -> dict:
    """Save the handoff summary for the human agent to ticket_messages with role=system.

    This keeps it separate from the customer conversation (user / ai).
    load_past_context only reads user and ai messages, so the summary never leaks into past_context.
    """
    if not ticket_id:
        return _error("ticket_id is required")
    if not summary or not summary.strip():
        return _error("summary is required")
    if _db_missing():
        return _error(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    try:
        with Session() as session:
            if session.get(udahub.Ticket, ticket_id) is None:
                return _error(f"Ticket not found: {ticket_id}")
            message = udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                role=udahub.RoleEnum.system,
                content=f"[Handoff] {summary.strip()}",
                created_at=datetime.now(),
            )
            session.add(message)
            session.commit()
            return _ok({"ticket_id": ticket_id, "message_id": message.message_id})
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def update_ticket_status(ticket_id: str, status: str,
                         main_issue_type: str | None = None,
                         tags: str | None = None) -> dict:
    """Save the ticket result (resolved / escalated) and classification to ticket_metadata.

    main_issue_type and tags are overwritten only when given.
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
