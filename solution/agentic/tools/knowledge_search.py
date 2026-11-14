"""Knowledge article search (RAG). Design doc (agentic/design/architecture.md), Chapter 5.

Uses LangChain's Chroma (langchain_chroma).

Flow:
  [Setup]  udahub.db knowledge table -> Documents -> stored in Chroma (only when empty)
  [Search] similarity_search_with_score(question) -> closest articles and cosine distances
           -> confidence = 1 - cosine distance of the top article

Return values have the same shape as cultpass_tools.py: {"ok": ..., "data"/"error": ...}.

Prerequisites:
- the solution/ folder is on sys.path
- the VOCAREUM_API_KEY environment variable is set (the caller runs load_dotenv())
"""
import os
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import udahub

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
# solution/agentic/tools/knowledge_search.py → parents[2] = solution/
SOLUTION_DIR = Path(__file__).resolve().parents[2]
UDAHUB_DB_PATH = SOLUTION_DIR / "data" / "core" / "udahub.db"
CHROMA_DIR = SOLUTION_DIR / "data" / "core" / "chroma"

COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"  # Use OpenAI via Udacity (Vocareum)
# Measured (2026-11-14): relevant questions 0.312-0.659 / unrelated questions 0.136-0.253 -> 0.30 in between
CONFIDENCE_THRESHOLD = 0.30
TOP_K = 3

# The vector store is created only once, on first use
_vector_store = None


def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


# ---------------------------------------------------------------
# Setup: load articles and build the index
# ---------------------------------------------------------------
def load_articles() -> list[Document]:
    """Load articles from the knowledge table in udahub.db and turn them into a list of Documents.

    page_content (the text that gets embedded) combines the title, tags, and body.
    """
    if not UDAHUB_DB_PATH.exists():
        raise FileNotFoundError(f"UDA-Hub DB not found: {UDAHUB_DB_PATH}")

    Session = sessionmaker(bind=create_engine(f"sqlite:///{UDAHUB_DB_PATH}"))
    with Session() as session:
        return [
            Document(
                id=a.article_id,
                page_content=f"{a.title}\nTags: {a.tags or ''}\n{a.content}",
                metadata={"article_id": a.article_id, "title": a.title, "tags": a.tags or ""},
            )
            for a in session.query(udahub.Knowledge).all()
        ]


def get_vector_store(rebuild: bool = False) -> Chroma:
    """Return the Chroma vector store. If it is empty (or rebuild=True), add the articles."""
    global _vector_store

    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OPENAI_BASE_URL,
                api_key=os.getenv("VOCAREUM_API_KEY"),
            ),
            persist_directory=str(CHROMA_DIR),               # persist to disk
            collection_metadata={"hnsw:space": "cosine"},    # cosine distance (= 1 - cosine similarity)
            client_settings=Settings(anonymized_telemetry=False),  # do not send usage data externally
        )

    if rebuild:
        # When articles are updated, clear the collection and rebuild it
        _vector_store.reset_collection()

    if not _vector_store.get(limit=1)["ids"]:
        docs = load_articles()
        _vector_store.add_documents(documents=docs, ids=[d.id for d in docs])

    return _vector_store


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------
def search_knowledge(query: str, k: int = TOP_K) -> dict:
    """Return the k articles closest to the question.

    Contents of data:
        confidence   : score of the top article (0-1)
        is_confident : whether confidence >= threshold (False means escalate)
        articles     : [{article_id, title, tags, content, score}, ...] (highest score first)
    """
    if not query or not query.strip():
        return _error("query is required")

    try:
        # The returned score is a "distance" (smaller = closer)
        results = get_vector_store().similarity_search_with_score(query=query, k=k)
    except Exception as e:  # missing DB, API error, etc.
        return _error(f"Knowledge search failed: {e}")

    articles = [
        {
            "article_id": doc.metadata["article_id"],
            "title": doc.metadata["title"],
            "tags": doc.metadata["tags"],
            "content": doc.page_content,
            "score": round(1 - distance, 4),  # cosine distance -> similarity
        }
        for doc, distance in results
    ]

    confidence = articles[0]["score"] if articles else 0.0
    return _ok({
        "confidence": confidence,
        "is_confident": confidence >= CONFIDENCE_THRESHOLD,
        "articles": articles,
    })


# ---------------------------------------------------------------
# Wrapper (LangChain @tool). Same pattern as cultpass_tools.py.
# - The name the LLM sees is "search_knowledge", same as the core function
# - k is not chosen by the LLM; it is fixed to TOP_K (3)
# - The docstring becomes the tool description for the LLM, so it is written in English
# ---------------------------------------------------------------
@tool("search_knowledge", parse_docstring=True)
def search_knowledge_tool(query: str) -> dict:
    """Search the CultPass help-center articles for the customer's question.
    Returns the top matching articles with a similarity score, plus
    confidence (the top score, 0-1) and is_confident. If is_confident is
    False, the articles are probably not relevant: do not answer from them,
    escalate to a human instead.

    Args:
        query: The customer's question, in their own words.
    """
    return search_knowledge(query)
