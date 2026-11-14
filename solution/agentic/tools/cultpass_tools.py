"""CultPass DB を操作するツール。

設計書（agentic/design/architecture.md の 6.1 節）に沿って、2層に分ける。
- 本体    : 普通の Python 関数。結果は {"ok": ..., "data"/"error": ...} の dict で返す。
- ラッパー : LangChain の @tool で本体を包む（後で追加する。MCP に移るときはここを差し替える）。

前提: solution/ フォルダが sys.path に入っていること
（solution/ でノートブックを開けば、ほかのノートブックと同じく `data.models` を import できる）。
"""
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from data.models import cultpass

# このファイルの場所を基準に DB のパスを決める（実行場所に左右されない）
# solution/agentic/tools/cultpass_tools.py → parents[2] = solution/
SOLUTION_DIR = Path(__file__).resolve().parents[2]
CULTPASS_DB_PATH = SOLUTION_DIR / "data" / "external" / "cultpass.db"

_engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}")
Session = sessionmaker(bind=_engine)


# ---------------------------------------------------------------
# 返り値の形をそろえるための小さなヘルパー
# ---------------------------------------------------------------
def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _db_missing() -> bool:
    # SQLite はファイルがないと空の DB を勝手に作ってしまうので、先に確認する
    return not CULTPASS_DB_PATH.exists()


# ---------------------------------------------------------------
# 本体（DB 操作）
# ---------------------------------------------------------------
def get_user_profile(user_id: str) -> dict:
    """ユーザーが存在するか、ブロックされていないかを返す。"""
    if not user_id:
        return _error("user_id is required")
    if _db_missing():
        return _error(f"CultPass DB not found: {CULTPASS_DB_PATH}")

    try:
        with Session() as session:
            user = session.get(cultpass.User, user_id)
            if user is None:
                return _error(f"User not found: {user_id}")
            return _ok({
                "user_id": user.user_id,
                "is_blocked": user.is_blocked,
            })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def get_subscription(user_id: str) -> dict:
    """ユーザーのサブスクリプション（状態・プラン・月の利用枠）を返す。"""
    if not user_id:
        return _error("user_id is required")
    if _db_missing():
        return _error(f"CultPass DB not found: {CULTPASS_DB_PATH}")

    try:
        with Session() as session:
            user = session.get(cultpass.User, user_id)
            if user is None:
                return _error(f"User not found: {user_id}")
            sub = user.subscription
            if sub is None:
                return _error(f"No subscription found for user: {user_id}")
            return _ok({
                "subscription_id": sub.subscription_id,
                "status": sub.status,                # active / cancelled など
                "tier": sub.tier,                    # basic / premium
                "monthly_quota": sub.monthly_quota,  # 月に使える体験の数
                "started_at": sub.started_at.isoformat() if sub.started_at else None,
                "ended_at": sub.ended_at.isoformat() if sub.ended_at else None,
            })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def get_reservations(user_id: str) -> dict:
    """ユーザーの予約一覧を返す。予約が0件でも成功（空リスト）として返す。"""
    if not user_id:
        return _error("user_id is required")
    if _db_missing():
        return _error(f"CultPass DB not found: {CULTPASS_DB_PATH}")

    try:
        with Session() as session:
            user = session.get(cultpass.User, user_id)
            if user is None:
                return _error(f"User not found: {user_id}")
            reservations = [
                {
                    "reservation_id": r.reservation_id,
                    "experience_title": r.experience.title,
                    "location": r.experience.location,
                    "when": r.experience.when.isoformat(),
                    "is_premium": r.experience.is_premium,
                    "status": r.status,  # reserved / cancelled
                }
                for r in user.reservations
            ]
            return _ok(reservations)
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def cancel_reservation(user_id: str, reservation_id: str) -> dict:
    """予約をキャンセルする（status を 'cancelled' にし、体験の空き枠を1つ戻す）。

    ナレッジ記事「How to Cancel or Change a Reservation」のルールに合わせて、
    開始まで24時間を切っていてもキャンセルはできる。その場合は within_24h=True を返し、
    クレジットが消費されることをエージェントがユーザーに伝えられるようにする。
    """
    if not user_id or not reservation_id:
        return _error("user_id and reservation_id are required")
    if _db_missing():
        return _error(f"CultPass DB not found: {CULTPASS_DB_PATH}")

    try:
        with Session() as session:
            user = session.get(cultpass.User, user_id)
            if user is None:
                return _error(f"User not found: {user_id}")
            if user.is_blocked:
                return _error("User account is blocked")

            reservation = session.get(cultpass.Reservation, reservation_id)
            if reservation is None:
                return _error(f"Reservation not found: {reservation_id}")
            # 他人の予約はキャンセルさせない
            if reservation.user_id != user_id:
                return _error("Reservation does not belong to this user")
            if reservation.status == "cancelled":
                return _error("Reservation is already cancelled")

            event_time = reservation.experience.when
            now = datetime.now()
            if event_time <= now:
                return _error("The event has already started or finished")

            reservation.status = "cancelled"
            reservation.experience.slots_available += 1
            session.commit()

            return _ok({
                "reservation_id": reservation.reservation_id,
                "experience_title": reservation.experience.title,
                "status": reservation.status,
                "within_24h": event_time - now < timedelta(hours=24),
            })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


# ---------------------------------------------------------------
# ラッパー（LangChain の @tool）
# - LLM に見える名前は @tool("...") で指定する（本体と同じ名前にそろえる）
# - docstring がそのまま LLM 向けのツール説明になるので、英語で書く
# - MCP に移るときは、この部分だけを @mcp.tool に差し替える
# ---------------------------------------------------------------
@tool("get_user_profile", parse_docstring=True)
def get_user_profile_tool(user_id: str) -> dict:
    """Check whether a CultPass user exists and whether the account is blocked.

    Args:
        user_id: The CultPass user ID of the customer.
    """
    return get_user_profile(user_id)


@tool("get_subscription", parse_docstring=True)
def get_subscription_tool(user_id: str) -> dict:
    """Get the user's CultPass subscription: status (active/cancelled),
    tier (basic/premium), monthly experience quota, and start/end dates.

    Args:
        user_id: The CultPass user ID of the customer.
    """
    return get_subscription(user_id)


@tool("get_reservations", parse_docstring=True)
def get_reservations_tool(user_id: str) -> dict:
    """List the user's reservations with reservation_id, experience title,
    location, date/time, premium flag, and status (reserved/cancelled).

    Args:
        user_id: The CultPass user ID of the customer.
    """
    return get_reservations(user_id)


@tool("cancel_reservation", parse_docstring=True)
def cancel_reservation_tool(user_id: str, reservation_id: str) -> dict:
    """Cancel one of the user's reservations. Use get_reservations first to
    find the reservation_id. The result includes within_24h: if True, the
    cancellation is less than 24 hours before the event and one experience
    credit is used up; if False, the credit is returned.

    Args:
        user_id: The CultPass user ID of the customer.
        reservation_id: The ID of the reservation to cancel.
    """
    return cancel_reservation(user_id, reservation_id)


# エージェントに渡すツールの一覧
CULTPASS_TOOLS = [
    get_user_profile_tool,
    get_subscription_tool,
    get_reservations_tool,
    cancel_reservation_tool,
]
