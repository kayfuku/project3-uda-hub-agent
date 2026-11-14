"""CultPass DB ツール（agentic/tools/cultpass_tools.py）のテスト。

本物の cultpass.db は書き換えない。
各テストの前に DB を一時フォルダにコピーし、ツールがそのコピーを使うように差し替える。

実行方法（solution/ で）:
    ../venv/bin/python -m pytest -v
"""
import shutil
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.tools import cultpass_tools as ct
from data.models import cultpass


# ---------------------------------------------------------------
# フィクスチャ（テストの準備）
# ---------------------------------------------------------------
@pytest.fixture
def db(tmp_path, monkeypatch):
    """DB のコピーを作り、ツールがそのコピーを使うようにする。"""
    test_db = tmp_path / "cultpass.db"
    shutil.copy(ct.CULTPASS_DB_PATH, test_db)

    TestSession = sessionmaker(bind=create_engine(f"sqlite:///{test_db}"))
    monkeypatch.setattr(ct, "CULTPASS_DB_PATH", test_db)
    monkeypatch.setattr(ct, "Session", TestSession)
    return TestSession


@pytest.fixture
def users(db):
    """ブロックされていないユーザー2人と、ブロック済みユーザー1人の ID。"""
    with db() as s:
        normal = [u for (u,) in s.query(cultpass.User.user_id).filter_by(is_blocked=False)]
        blocked = s.query(cultpass.User.user_id).filter_by(is_blocked=True).first()[0]
    return {"normal": normal[0], "other": normal[1], "blocked": blocked}


@pytest.fixture
def make_reservation(db):
    """テスト用の予約を作る関数を返す。体験の開始時刻は now + hours にする。"""
    def _make(user_id, hours_from_now=48, status="reserved"):
        with db() as s:
            exp = s.query(cultpass.Experience).first()
            exp.when = datetime.now() + timedelta(hours=hours_from_now)
            rid = f"test-{user_id}-{hours_from_now}-{status}"
            s.add(cultpass.Reservation(
                reservation_id=rid, user_id=user_id,
                experience_id=exp.experience_id, status=status,
            ))
            s.commit()
            return rid
    return _make


# ---------------------------------------------------------------
# get_user_profile
# ---------------------------------------------------------------
def test_get_user_profile_found(users):
    result = ct.get_user_profile(users["normal"])
    assert result == {"ok": True, "data": {"user_id": users["normal"], "is_blocked": False}}


def test_get_user_profile_blocked(users):
    result = ct.get_user_profile(users["blocked"])
    assert result["ok"] is True
    assert result["data"]["is_blocked"] is True


def test_get_user_profile_not_found(db):
    result = ct.get_user_profile("no-such-user")
    assert result == {"ok": False, "error": "User not found: no-such-user"}


def test_get_user_profile_empty_id(db):
    assert ct.get_user_profile("") == {"ok": False, "error": "user_id is required"}


def test_db_missing(tmp_path, monkeypatch):
    # DB ファイルがないときに、空の DB を作らずエラーを返すこと
    missing = tmp_path / "missing.db"
    monkeypatch.setattr(ct, "CULTPASS_DB_PATH", missing)
    result = ct.get_user_profile("anyone")
    assert result["ok"] is False
    assert "DB not found" in result["error"]
    assert not missing.exists()


# ---------------------------------------------------------------
# get_subscription
# ---------------------------------------------------------------
def test_get_subscription_found(users):
    result = ct.get_subscription(users["normal"])
    assert result["ok"] is True
    assert set(result["data"]) >= {"status", "tier", "monthly_quota"}


def test_get_subscription_user_not_found(db):
    result = ct.get_subscription("no-such-user")
    assert result["ok"] is False
    assert "User not found" in result["error"]


# ---------------------------------------------------------------
# get_reservations
# ---------------------------------------------------------------
def test_get_reservations_empty_list(users):
    # 予約がないユーザーは、エラーではなく空リスト
    assert ct.get_reservations(users["normal"]) == {"ok": True, "data": []}


def test_get_reservations_lists_reservation(users, make_reservation):
    rid = make_reservation(users["normal"])
    result = ct.get_reservations(users["normal"])
    assert result["ok"] is True
    assert [r["reservation_id"] for r in result["data"]] == [rid]
    assert result["data"][0]["status"] == "reserved"


def test_get_reservations_user_not_found(db):
    assert ct.get_reservations("no-such-user")["ok"] is False


# ---------------------------------------------------------------
# cancel_reservation
# ---------------------------------------------------------------
def test_cancel_success_more_than_24h(db, users, make_reservation):
    rid = make_reservation(users["normal"], hours_from_now=48)
    with db() as s:
        slots_before = s.get(cultpass.Reservation, rid).experience.slots_available

    result = ct.cancel_reservation(users["normal"], rid)

    assert result["ok"] is True
    assert result["data"]["status"] == "cancelled"
    assert result["data"]["within_24h"] is False
    # DB にも反映されていること（ステータスと空き枠）
    with db() as s:
        r = s.get(cultpass.Reservation, rid)
        assert r.status == "cancelled"
        assert r.experience.slots_available == slots_before + 1


def test_cancel_success_within_24h(users, make_reservation):
    rid = make_reservation(users["normal"], hours_from_now=2)
    result = ct.cancel_reservation(users["normal"], rid)
    assert result["ok"] is True
    assert result["data"]["within_24h"] is True


def test_cancel_already_cancelled(users, make_reservation):
    rid = make_reservation(users["normal"])
    ct.cancel_reservation(users["normal"], rid)
    result = ct.cancel_reservation(users["normal"], rid)
    assert result == {"ok": False, "error": "Reservation is already cancelled"}


def test_cancel_other_users_reservation(db, users, make_reservation):
    rid = make_reservation(users["normal"])
    result = ct.cancel_reservation(users["other"], rid)
    assert result == {"ok": False, "error": "Reservation does not belong to this user"}
    # 予約は変わっていないこと
    with db() as s:
        assert s.get(cultpass.Reservation, rid).status == "reserved"


def test_cancel_blocked_user(users, make_reservation):
    rid = make_reservation(users["blocked"])
    result = ct.cancel_reservation(users["blocked"], rid)
    assert result == {"ok": False, "error": "User account is blocked"}


def test_cancel_event_already_started(users, make_reservation):
    rid = make_reservation(users["normal"], hours_from_now=-1)
    result = ct.cancel_reservation(users["normal"], rid)
    assert result == {"ok": False, "error": "The event has already started or finished"}


def test_cancel_reservation_not_found(users):
    result = ct.cancel_reservation(users["normal"], "no-such-reservation")
    assert result == {"ok": False, "error": "Reservation not found: no-such-reservation"}


def test_cancel_missing_arguments(db):
    result = ct.cancel_reservation("", "")
    assert result == {"ok": False, "error": "user_id and reservation_id are required"}


# ---------------------------------------------------------------
# @tool ラッパー
# ---------------------------------------------------------------
def test_tools_are_registered():
    names = [t.name for t in ct.CULTPASS_TOOLS]
    assert names == ["get_user_profile", "get_subscription",
                     "get_reservations", "cancel_reservation"]


def test_tool_invoke_returns_same_as_core(users):
    # ラッパーは本体を呼ぶだけなので、結果は同じになる
    tools = {t.name: t for t in ct.CULTPASS_TOOLS}
    assert tools["get_subscription"].invoke({"user_id": users["normal"]}) \
        == ct.get_subscription(users["normal"])
