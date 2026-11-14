"""Tools that operate on the CultPass DB.

Following the design doc (agentic/design/architecture.md, Section 6.1), there are two layers:
- Core    : plain Python functions. They return a dict {"ok": ..., "data"/"error": ...}.
- Wrapper : LangChain @tool around the core functions (swap this part when moving to MCP).

Prerequisite: the solution/ folder is on sys.path
(if the notebook is opened in solution/, `data.models` can be imported just like in the other notebooks).
"""
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from data.models import cultpass

# Resolve the DB path relative to this file (independent of the working directory)
# solution/agentic/tools/cultpass_tools.py → parents[2] = solution/
SOLUTION_DIR = Path(__file__).resolve().parents[2]
CULTPASS_DB_PATH = SOLUTION_DIR / "data" / "external" / "cultpass.db"

_engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}")
Session = sessionmaker(bind=_engine)


# ---------------------------------------------------------------
# Small helpers to keep the return shape consistent
# ---------------------------------------------------------------
def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _db_missing() -> bool:
    # SQLite silently creates an empty DB if the file is missing, so check first
    return not CULTPASS_DB_PATH.exists()


# ---------------------------------------------------------------
# Core functions (DB operations)
# ---------------------------------------------------------------
def get_user_profile(user_id: str) -> dict:
    """Return whether the user exists and is not blocked."""
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
    """Return the user's subscription (status, plan, monthly quota)."""
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
                "status": sub.status,                # active / cancelled, etc.
                "tier": sub.tier,                    # basic / premium
                "monthly_quota": sub.monthly_quota,  # number of experiences allowed per month
                "started_at": sub.started_at.isoformat() if sub.started_at else None,
                "ended_at": sub.ended_at.isoformat() if sub.ended_at else None,
            })
    except SQLAlchemyError as e:
        return _error(f"Database error: {e}")


def get_reservations(user_id: str) -> dict:
    """Return the user's reservations. Zero reservations is still a success (empty list)."""
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
    """Cancel a reservation (set status to 'cancelled' and give one slot back to the experience).

    Following the rules in the knowledge article "How to Cancel or Change a Reservation",
    cancellation is allowed even within 24 hours of the start. In that case within_24h=True is returned
    so the agent can tell the user that the credit will be used.
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
            # Do not allow cancelling someone else's reservation
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
# Wrappers (LangChain @tool)
# - The name the LLM sees is set with @tool("...") (same as the core function name)
# - The docstring becomes the tool description for the LLM, so it is written in English
# - When moving to MCP, replace only this part with @mcp.tool
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


# List of tools given to the agent
CULTPASS_TOOLS = [
    get_user_profile_tool,
    get_subscription_tool,
    get_reservations_tool,
    cancel_reservation_tool,
]
