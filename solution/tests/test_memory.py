"""長期メモリ（agentic/tools/memory.py）のテスト。

本物の udahub.db は書き換えない。
各テストの前に DB を一時フォルダにコピーし、memory.py がそのコピーを使うように差し替える。
API は使わない。

実行方法（solution/ で）:
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

USER = "test-user-1"  # udahub.db にまだいない CultPass ユーザー（テスト用）


# ---------------------------------------------------------------
# フィクスチャ（テストの準備）
# ---------------------------------------------------------------
@pytest.fixture
def db(tmp_path, monkeypatch):
    """DB のコピーを作り、memory.py がそのコピーを使うようにする。"""
    test_db = tmp_path / "udahub.db"
    shutil.copy(memory.UDAHUB_DB_PATH, test_db)

    TestSession = sessionmaker(bind=create_engine(f"sqlite:///{test_db}"))
    monkeypatch.setattr(memory, "UDAHUB_DB_PATH", test_db)
    monkeypatch.setattr(memory, "Session", TestSession)
    return TestSession


def finished_ticket(user_id, question, answer, status="resolved", **meta):
    """チケットを作り、会話を保存して、結果を記録する（1件の問い合わせを最後まで処理した状態）。"""
    ticket_id = memory.create_ticket(user_id)["data"]["ticket_id"]
    memory.save_messages(ticket_id, [HumanMessage(question, id=f"{ticket_id}-q"),
                                     AIMessage(answer, id=f"{ticket_id}-a")])
    memory.update_ticket_status(ticket_id, status, **meta)
    return ticket_id


def set_created_at(db, ticket_id, when):
    """チケットの作成日時を書き換える（並び順のテスト用）。"""
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
    """同じチケットで会話が続き、前の発言を含むリストをもう一度渡しても二重にならない。"""
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
    """1回目のチケットの内容が、2回目のチケットで過去の文脈として取り出せる。"""
    finished_ticket(USER, "The QR code won't scan", "Please increase screen brightness.",
                    main_issue_type="technical", tags="qr_code")
    current = memory.create_ticket(USER)["data"]["ticket_id"]

    result = memory.load_past_context(USER, current_ticket_id=current)

    tickets = result["data"]["tickets"]
    assert len(tickets) == 1  # 今のチケットは含まない
    assert tickets[0]["status"] == "resolved"
    assert tickets[0]["first_user_message"] == "The QR code won't scan"
    assert tickets[0]["last_ai_message"] == "Please increase screen brightness."

    summary = result["data"]["summary"]
    assert "[resolved] technical (qr_code)" in summary
    assert "The QR code won't scan" in summary


def test_past_context_is_newest_first_and_limited(db):
    for day in range(1, 5):  # 11/1〜11/4 の4件
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
    """thread_id（= ticket_id）から、チケットの持ち主と metadata がわかる。"""
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
    """要約は担当者向けなので、次回の past_context（お客さんとの会話の要約）には出てこない。"""
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
