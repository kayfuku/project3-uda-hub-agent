"""4 つのエージェントが共通で使う LLM。

前提: 環境変数 VOCAREUM_API_KEY が設定されていること（呼び出す側で load_dotenv() しておく）
"""
import os

from langchain_openai import ChatOpenAI

LLM_MODEL = "gpt-5.4-mini"
# Udacity（Vocareum）経由で OpenAI を使う
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"


def get_llm() -> ChatOpenAI:
    """エージェント用の ChatOpenAI を作る。

    temperature は指定しない（gpt-5 系は既定値以外を受け付けないことがあるため）。
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=os.getenv("VOCAREUM_API_KEY"),
    )
