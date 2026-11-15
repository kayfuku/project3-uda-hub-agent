"""Tests for long-term memory (agentic/tools/memory.py).

The real udahub.db is never modified.
Before each test, the DB is copied to a temp folder and memory.py is patched to use the copy.
No API calls are made.

How to run (from solution/):
    ../venv/bin/python -m pytest -v
"""
import shutil
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.tools import memory
from data.models import udahub

USER = "test-user-1"  # CultPass user not yet in udahub.db (for testing)


# ---------------------------------------------------------------
# Fixtures (test setup)
# ---------------------------------------------------------------
@pytest.fixture
def db(tmp_path, monkeypatch):
    """Copy the DB and make memory.py use the copy."""
    test_db = tmp_path / "udahub.db"
    shutil.copy(memory.UDAHUB_DB_PATH, test_db)

    test_engine = create_engine(f"sqlite:///{test_db}")
    udahub.Base.metadata.create_all(test_engine)  # add tables missing from an older DB (e.g. customer_preferences)
    TestSession = sessionmaker(bind=test_engine)
    monkeypatch.setattr(memory, "UDAHUB_DB_PATH", test_db)
    monkeypatch.setattr(memory, "Session", TestSession)
    return TestSession


def finished_ticket(user_id, question, answer, status="resolved", **meta):
    """Create a ticket, save a conversation, and record the result (one inquiry handled end to end)."""
    ticket_id = memory.create_ticket(user_id)["data"]["ticket_id"]
    memory.save_messages(ticket_id, [HumanMessage(question, id=f"{ticket_id}-q"),
                                     AIMessage(answer, id=f"{ticket_id}-a")])
    memory.update_ticket_status(ticket_id, status, **meta)
    return ticket_id


def set_created_at(db, ticket_id, when):
    """Overwrite the ticket's created_at (for ordering tests)."""
    with db() as s:
        s.get(udahub.Ticket, ticket_id).created_at = when
        s.commit()


# ---------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------
def test_create_ticket_creates_user_and_open_ticket(db):
    result = memory.create_ticket(USER)

    assert result["ok"] is True
    with db() as s:
        meta = s.get(udahub.TicketMetadata, result["data"]["ticket_id"])
        assert meta.status == "open"
        users = s.query(udahub.User).filter_by(external_user_id=USER).all()
        assert len(users) == 1


def test_create_ticket_reuses_existing_user(db):
    memory.create_ticket(USER)
    memory.create_ticket(USER)

    with db() as s:
        assert s.query(udahub.User).filter_by(external_user_id=USER).count() == 1


def test_create_ticket_requires_user_id(db):
    assert memory.create_ticket("") == {"ok": False, "error": "external_user_id is required"}


# ---------------------------------------------------------------
# save_messages
# ---------------------------------------------------------------
def test_save_messages_keeps_only_user_and_ai_text(db):
    ticket_id = memory.create_ticket(USER)["data"]["ticket_id"]
    messages = [
        HumanMessage("Cancel my booking", id="m1"),
        AIMessage("", id="m2", tool_calls=[{"name": "get_reservations",
                                            "args": {"user_id": USER}, "id": "call1"}]),
        ToolMessage('{"ok": true}', tool_call_id="call1", id="m3"),
        AIMessage("Your booking is cancelled.", id="m4"),
    ]

    result = memory.save_messages(ticket_id, messages)

    assert result["data"]["saved"] == 2
    with db() as s:
        rows = s.query(udahub.TicketMessage).filter_by(ticket_id=ticket_id).all()
        assert sorted(r.message_id for r in rows) == ["m1", "m4"]


def test_save_messages_does_not_save_twice(db):
    """When a conversation continues in the same ticket, passing the list with earlier messages again does not duplicate them."""
    ticket_id = memory.create_ticket(USER)["data"]["ticket_id"]
    first_turn = [HumanMessage("Hi", id="m1"), AIMessage("Hello!", id="m2")]
    second_turn = first_turn + [HumanMessage("Thanks", id="m3")]

    memory.save_messages(ticket_id, first_turn)
    result = memory.save_messages(ticket_id, second_turn)

    assert result["data"]["saved"] == 1
    with db() as s:
        assert s.query(udahub.TicketMessage).filter_by(ticket_id=ticket_id).count() == 3


def test_save_messages_unknown_ticket(db):
    result = memory.save_messages("no-such-ticket", [HumanMessage("Hi")])
    assert result == {"ok": False, "error": "Ticket not found: no-such-ticket"}


# ---------------------------------------------------------------
# update_ticket_status
# ---------------------------------------------------------------
def test_update_ticket_status(db):
    ticket_id = memory.create_ticket(USER)["data"]["ticket_id"]

    result = memory.update_ticket_status(ticket_id, "escalated",
                                         main_issue_type="billing", tags="refund")

    assert result["ok"] is True
    assert result["data"]["status"] == "escalated"
    assert result["data"]["main_issue_type"] == "billing"
    assert result["data"]["tags"] == "refund"


def test_update_ticket_status_rejects_invalid_status(db):
    ticket_id = memory.create_ticket(USER)["data"]["ticket_id"]
    result = memory.update_ticket_status(ticket_id, "done")
    assert result["ok"] is False
    assert "Invalid status" in result["error"]


def test_update_ticket_status_unknown_ticket(db):
    result = memory.update_ticket_status("no-such-ticket", "resolved")
    assert result == {"ok": False, "error": "Ticket not found: no-such-ticket"}


# ---------------------------------------------------------------
# load_past_context
# ---------------------------------------------------------------
def test_new_customer_has_no_past_context(db):
    result = memory.load_past_context("never-seen-user")

    assert result["ok"] is True
    assert result["data"]["tickets"] == []
    assert result["data"]["summary"] == "No past tickets from this customer."


def test_returning_customer_gets_past_ticket(db):
    """The content of the first ticket is available as past context in the second ticket."""
    finished_ticket(USER, "The QR code won't scan", "Please increase screen brightness.",
                    main_issue_type="technical", tags="qr_code")
    current = memory.create_ticket(USER)["data"]["ticket_id"]

    result = memory.load_past_context(USER, current_ticket_id=current)

    tickets = result["data"]["tickets"]
    assert len(tickets) == 1  # the current ticket is not included
    assert tickets[0]["status"] == "resolved"
    assert tickets[0]["first_user_message"] == "The QR code won't scan"
    assert tickets[0]["last_ai_message"] == "Please increase screen brightness."

    summary = result["data"]["summary"]
    assert "[resolved] technical (qr_code)" in summary
    assert "The QR code won't scan" in summary


def test_past_context_is_newest_first_and_limited(db):
    for day in range(1, 5):  # 4 tickets, Nov 1-4
        tid = finished_ticket(USER, f"question on day {day}", f"answer {day}")
        set_created_at(db, tid, datetime(2026, 11, day))

    tickets = memory.load_past_context(USER)["data"]["tickets"]

    assert [t["created_at"] for t in tickets] == ["2026-11-04", "2026-11-03", "2026-11-02"]


def test_past_context_does_not_mix_users(db):
    finished_ticket("other-user", "Other user's question", "Other answer")

    assert memory.load_past_context(USER)["data"]["tickets"] == []


# ---------------------------------------------------------------
# get_ticket_info
# ---------------------------------------------------------------
def test_get_ticket_info_returns_owner_and_metadata(db):
    """The ticket owner and metadata can be found from the thread_id (= ticket_id)."""
    ticket_id = memory.create_ticket(USER, channel="email")["data"]["ticket_id"]

    result = memory.get_ticket_info(ticket_id)

    assert result["ok"] is True
    assert result["data"]["external_user_id"] == USER
    assert result["data"]["channel"] == "email"
    assert result["data"]["status"] == "open"


def test_get_ticket_info_unknown_ticket(db):
    result = memory.get_ticket_info("no-such-ticket")
    assert result == {"ok": False, "error": "Ticket not found: no-such-ticket"}


def test_get_ticket_info_requires_ticket_id(db):
    assert memory.get_ticket_info("") == {"ok": False, "error": "ticket_id is required"}


# ---------------------------------------------------------------
# save_handoff_summary
# ---------------------------------------------------------------
def test_save_handoff_summary_saves_system_message(db):
    ticket_id = memory.create_ticket(USER)["data"]["ticket_id"]

    result = memory.save_handoff_summary(ticket_id, "Customer reports a duplicate charge.")

    assert result["ok"] is True
    with db() as s:
        rows = s.query(udahub.TicketMessage).filter_by(ticket_id=ticket_id).all()
        assert len(rows) == 1
        assert rows[0].role == udahub.RoleEnum.system
        assert rows[0].content == "[Handoff] Customer reports a duplicate charge."


def test_handoff_summary_is_not_in_past_context(db):
    """The summary is for the human agent, so it does not appear in the next past_context (summary of customer conversations)."""
    ticket_id = finished_ticket(USER, "I was charged twice", "We will forward this.",
                                status="escalated")
    memory.save_handoff_summary(ticket_id, "Duplicate charge, needs billing team.")

    summary = memory.load_past_context(USER)["data"]["summary"]

    assert "I was charged twice" in summary
    assert "Handoff" not in summary


def test_save_handoff_summary_rejects_empty_summary(db):
    ticket_id = memory.create_ticket(USER)["data"]["ticket_id"]
    assert memory.save_handoff_summary(ticket_id, "  ") == {"ok": False, "error": "summary is required"}


def test_save_handoff_summary_unknown_ticket(db):
    result = memory.save_handoff_summary("no-such-ticket", "summary")
    assert result == {"ok": False, "error": "Ticket not found: no-such-ticket"}


# ---------------------------------------------------------------
# save_preferences / load_preferences
# ---------------------------------------------------------------
def test_new_customer_has_no_preferences(db):
    assert memory.load_preferences("never-seen-user") == {"ok": True, "data": {}}


def test_preferences_are_kept_across_tickets(db):
    """Preferences saved in one ticket can be loaded in a later ticket of the same user."""
    memory.create_ticket(USER)
    memory.save_preferences(USER, {"language": "Japanese", "interests": "jazz concerts"})
    memory.create_ticket(USER)  # a new session

    result = memory.load_preferences(USER)

    assert result["data"] == {"language": "Japanese", "interests": "jazz concerts"}


def test_save_preferences_overwrites_only_given_keys(db):
    memory.create_ticket(USER)
    memory.save_preferences(USER, {"language": "Japanese", "others": "Keep replies short"})

    result = memory.save_preferences(USER, {"language": "English"})

    assert result["data"]["saved"] == ["language"]
    assert memory.load_preferences(USER)["data"] == {
        "language": "English", "others": "Keep replies short"}


def test_save_preferences_skips_unknown_keys_and_empty_values(db):
    memory.create_ticket(USER)

    result = memory.save_preferences(USER, {"language": "  ", "interests": None,
                                            "phone": "090-0000-0000"})

    assert result == {"ok": True, "data": {"saved": []}}
    assert memory.load_preferences(USER)["data"] == {}


def test_preferences_do_not_mix_users(db):
    memory.create_ticket("other-user")
    memory.save_preferences("other-user", {"language": "Japanese"})

    assert memory.load_preferences(USER)["data"] == {}


def test_save_preferences_unknown_user(db):
    result = memory.save_preferences("no-such-user", {"language": "Japanese"})
    assert result == {"ok": False, "error": "User not found: no-such-user"}
