"""Structured logging (JSON Lines). Design doc, Chapter 9.

Appends one JSON object per line to solution/logs/workflow.jsonl.
Entries can be searched and filtered by ticket_id or event.

Example:
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
    """Set up the file handler on first use (only once)."""
    if not _logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))  # write only the JSON
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False  # do not show in the notebook output
    return _logger


def log_event(ticket_id: str, node: str, event: str, **fields) -> None:
    """Record an event as one line of JSON."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ticket_id": ticket_id,
        "node": node,
        "event": event,
        **fields,
    }
    _get_logger().info(json.dumps(record, ensure_ascii=False, default=str))


def read_logs(ticket_id: str | None = None) -> list[dict]:
    """Read the logs. If ticket_id is given, return only that ticket's entries (for checking in the notebook)."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if ticket_id:
        records = [r for r in records if r.get("ticket_id") == ticket_id]
    return records
