"""Tests for knowledge search (agentic/tools/knowledge_search.py).

Split into two groups:
- A. Tests without the API : input checks, error handling, wrapper, article loading
- B. Tests with the API    : create real embeddings and check confidence and threshold
                          (skipped automatically if VOCAREUM_API_KEY is not set)

The questions in B were chosen from the design doc Section 5.2 measurements as ones far from the threshold (0.30).

How to run (from solution/):
    ../venv/bin/python -m pytest -v
"""
import os

import pytest
from dotenv import load_dotenv

from agentic.tools import knowledge_search as ks

load_dotenv()

needs_api = pytest.mark.skipif(
    not os.getenv("VOCAREUM_API_KEY"),
    reason="VOCAREUM_API_KEY is not set",
)


# ---------------------------------------------------------------
# A. Tests without the API
# ---------------------------------------------------------------
@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_returns_error(query):
    result = ks.search_knowledge(query)
    assert result == {"ok": False, "error": "query is required"}


def test_search_failure_returns_error(monkeypatch):
    """If an exception occurs during search, return ok=False instead of raising."""
    def broken_store():
        raise RuntimeError("connection failed")

    monkeypatch.setattr(ks, "get_vector_store", broken_store)
    result = ks.search_knowledge("I was charged twice")

    assert result["ok"] is False
    assert "connection failed" in result["error"]


def test_load_articles_reads_all_articles():
    docs = ks.load_articles()

    assert len(docs) == 14
    # The embedded text includes the title
    assert all(d.metadata["title"] in d.page_content for d in docs)


def test_tool_wrapper_schema():
    """The name and arguments seen by the LLM are correct (k is hidden)."""
    assert ks.search_knowledge_tool.name == "search_knowledge"
    assert list(ks.search_knowledge_tool.args) == ["query"]


def test_tool_wrapper_calls_main_function():
    result = ks.search_knowledge_tool.invoke({"query": ""})
    assert result == {"ok": False, "error": "query is required"}


# ---------------------------------------------------------------
# B. Tests with the API
# ---------------------------------------------------------------
@needs_api
def test_relevant_question_is_confident():
    result = ks.search_knowledge("I was charged twice this month")

    assert result["ok"] is True
    data = result["data"]
    assert data["is_confident"] is True
    assert data["articles"][0]["title"] == "Unexpected or Duplicate Charges"


@needs_api
def test_unrelated_question_is_not_confident():
    result = ks.search_knowledge("What's the weather like in Bahia next week?")

    assert result["ok"] is True
    assert result["data"]["is_confident"] is False


@needs_api
def test_articles_are_sorted_by_score():
    result = ks.search_knowledge("How do I cancel my booking for tomorrow?")
    data = result["data"]
    scores = [a["score"] for a in data["articles"]]

    assert len(scores) == ks.TOP_K
    assert scores == sorted(scores, reverse=True)
    assert data["confidence"] == scores[0]
