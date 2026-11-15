"""LLM and prompt parts shared by the agents.

Prerequisite: the VOCAREUM_API_KEY environment variable is set (the caller runs load_dotenv())
"""
import os

from langchain_openai import ChatOpenAI

from agentic.tools.memory import format_preferences

# Shared by Knowledge / Account / Escalation, at the top of their system prompts.
# The rules do not depend on the kind of preference, so adding a new preference key needs no change here.
PREFERENCE_RULES = """\
Customer preferences (stated by the customer in earlier inquiries): {preferences}
- These preferences take priority over the default style rules below.
  Example: reply in the preferred language even if the message or the articles are in another language.
- Follow every preference about how to reply. Use interests only when they are relevant.
"""


def preference_rules(preferences: dict) -> str:
    """Return PREFERENCE_RULES filled with the customer's preferences."""
    return PREFERENCE_RULES.format(preferences=format_preferences(preferences))

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
