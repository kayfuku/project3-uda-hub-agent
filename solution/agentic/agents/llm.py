"""LLM shared by the four agents.

Prerequisite: the VOCAREUM_API_KEY environment variable is set (the caller runs load_dotenv())
"""
import os

from langchain_openai import ChatOpenAI

LLM_MODEL = "gpt-5.4-mini"
# Use OpenAI via Udacity (Vocareum)
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"


def get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance for the agents.

    temperature is not set (gpt-5 models may reject values other than the default).
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=os.getenv("VOCAREUM_API_KEY"),
    )
