"""ナレッジ記事の検索（RAG）。設計書（agentic/design/architecture.md）の第5章。

LangChain の Chroma（langchain_chroma）を使う。

流れ:
  【準備】udahub.db の knowledge テーブル → Document にする → Chroma に保存（空のときだけ）
  【検索】similarity_search_with_score(質問) → 近い記事と cosine 距離
          → 信頼度 = 1 − 1位の cosine 距離

返り値の形は cultpass_tools.py と同じ {"ok": ..., "data"/"error": ...}。

前提:
- solution/ フォルダが sys.path に入っていること
- 環境変数 VOCAREUM_API_KEY が設定されていること（呼び出す側で load_dotenv() しておく）
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
# 設定
# ---------------------------------------------------------------
# solution/agentic/tools/knowledge_search.py → parents[2] = solution/
SOLUTION_DIR = Path(__file__).resolve().parents[2]
UDAHUB_DB_PATH = SOLUTION_DIR / "data" / "core" / "udahub.db"
CHROMA_DIR = SOLUTION_DIR / "data" / "core" / "chroma"

COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"  # Udacity（Vocareum）経由で OpenAI を使う
# 実測（2026-11-14）: 関係ある質問 0.312〜0.659 / 関係ない質問 0.136〜0.253 → その間の 0.30
CONFIDENCE_THRESHOLD = 0.30
TOP_K = 3

# ベクトルストアは、最初に使うときに1回だけ作る
_vector_store = None


def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


# ---------------------------------------------------------------
# 準備：記事の読み込みとインデックス作成
# ---------------------------------------------------------------
def load_articles() -> list[Document]:
    """udahub.db の knowledge テーブルから記事を読み込み、Document のリストにする。

    page_content（埋め込まれる文字列）には、タイトル・タグ・本文をまとめて入れる。
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
    """Chroma のベクトルストアを返す。空のとき（または rebuild=True のとき）は記事を登録する。"""
    global _vector_store

    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OPENAI_BASE_URL,
                api_key=os.getenv("VOCAREUM_API_KEY"),
            ),
            persist_directory=str(CHROMA_DIR),               # ディスクに保存する
            collection_metadata={"hnsw:space": "cosine"},    # 距離は cosine（= 1 − コサイン類似度）
            client_settings=Settings(anonymized_telemetry=False),  # 利用状況を外部に送らない
        )

    if rebuild:
        # 記事を更新したときは、コレクションを空にして作り直す
        _vector_store.reset_collection()

    if not _vector_store.get(limit=1)["ids"]:
        docs = load_articles()
        _vector_store.add_documents(documents=docs, ids=[d.id for d in docs])

    return _vector_store


# ---------------------------------------------------------------
# 検索
# ---------------------------------------------------------------
def search_knowledge(query: str, k: int = TOP_K) -> dict:
    """質問に近い記事を k 件返す。

    data の中身:
        confidence   : 1位の記事のスコア（0〜1）
        is_confident : confidence がしきい値以上か（False ならエスカレーション）
        articles     : [{article_id, title, tags, content, score}, ...]（スコアの高い順）
    """
    if not query or not query.strip():
        return _error("query is required")

    try:
        # 返ってくる score は「距離」（小さいほど近い）
        results = get_vector_store().similarity_search_with_score(query=query, k=k)
    except Exception as e:  # DB がない、API エラーなど
        return _error(f"Knowledge search failed: {e}")

    articles = [
        {
            "article_id": doc.metadata["article_id"],
            "title": doc.metadata["title"],
            "tags": doc.metadata["tags"],
            "content": doc.page_content,
            "score": round(1 - distance, 4),  # cosine 距離 → 類似度
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
# ラッパー（LangChain の @tool）。cultpass_tools.py と同じ形。
# - LLM に見える名前は本体と同じ "search_knowledge"
# - k は LLM に選ばせず、TOP_K（3件）で固定する
# - docstring がそのまま LLM 向けのツール説明になるので、英語で書く
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
