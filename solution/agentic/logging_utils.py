"""構造化ログ（JSON Lines）。設計書の第9章。

1 行に 1 件の JSON を solution/logs/workflow.jsonl に追記する。
ticket_id や event で検索・絞り込みができる。

例:
  {"ts": "2026-11-14T10:00:00", "ticket_id": "t-001", "node": "classifier",
   "event": "classified", "category": "knowledge", "urgency": "low"}
"""
import json
import logging
from datetime import datetime
from pathlib import Path

# solution/agentic/logging_utils.py → parents[1] = solution/
SOLUTION_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = SOLUTION_DIR / "logs" / "workflow.jsonl"

_logger = logging.getLogger("udahub")


def _get_logger() -> logging.Logger:
    """最初に使うときに、ファイルへ書き出す設定をする（1 回だけ）。"""
    if not _logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))  # JSON だけを書く
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False  # ノートブックの画面には出さない
    return _logger


def log_event(ticket_id: str, node: str, event: str, **fields) -> None:
    """イベントを 1 行の JSON として記録する。"""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ticket_id": ticket_id,
        "node": node,
        "event": event,
        **fields,
    }
    _get_logger().info(json.dumps(record, ensure_ascii=False, default=str))


def read_logs(ticket_id: str | None = None) -> list[dict]:
    """ログを読み込む。ticket_id を渡すと、そのチケットの分だけ返す（ノートブックでの確認用）。"""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if ticket_id:
        records = [r for r in records if r.get("ticket_id") == ticket_id]
    return records
