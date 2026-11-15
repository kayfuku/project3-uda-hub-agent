"""Knowledge Agent. Design doc, Section 3.2 and Chapter 5.

Searches the knowledge articles and answers based only on their content.

  (1) search_knowledge(latest user message) -> confidence and top 3 articles
  (2) search failed, or is_confident is False -> no answer; return with escalation_reason set
  (3) True -> pass the top 3 articles to the LLM to answer (structured output)
      If an article says to hand off to a human, the LLM returns needs_escalation=True

Whether to escalate is decided by the graph (workflow.py) based on the returned escalation_reason.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.classifier import _latest_user_message
from agentic.agents.llm import get_llm, preference_rules
from agentic.state import SupportState
from agentic.tools.knowledge_search import CONFIDENCE_THRESHOLD, search_knowledge


class KnowledgeAnswer(BaseModel):
    """LLM output of the Knowledge Agent.

    applied_preferences is answered first, so the preferences to follow are written down before the answer
    (without it, the agent sometimes replied in English although the customer prefers Japanese).
    """
    applied_preferences: str = Field(
        description="The customer preferences this reply must follow, copied from 'Customer preferences'. "
                    "'None' if there are none.")
    answer: str = Field(
        description="The reply to the customer, based only on the articles and following applied_preferences.")
    used_article_titles: list[str] = Field(
        description="Titles of the articles the answer is based on.")
    needs_escalation: bool = Field(
        description="True if the articles say this case must be handled by a human, "
                    "or if the articles do not contain the answer.")
    escalation_reason: str = Field(
        description="Why a human is needed. Empty string if needs_escalation is False.")


SYSTEM_PROMPT = """\
You are the knowledge agent of UDA-Hub, answering CultPass customers.
Answer ONLY with information from the help-center articles below. Do not invent policies, prices or steps.

{preference_rules}
Default style rules:
- Reply in a short and friendly way.
- If no preference says otherwise, reply in the same language as the customer.

Rules:
- Some articles say a case must be escalated to a human (for example duplicate charges or refund exceptions).
  Set needs_escalation to true ONLY if what the customer has already told you meets that escalation condition.
  In that case, give the guidance from the article and tell them a support agent will follow up.
- If the article first gives steps to try, and escalates only when those steps fail, give the steps and
  set needs_escalation to false. The customer can write again if the steps do not work.
  Example: "The QR code won't scan" -> give the troubleshooting steps, needs_escalation = false.
  Example: "I was charged twice for the same month" -> the condition is met, needs_escalation = true.
- If the articles do not contain the answer, set needs_escalation to true and do not guess.
"""


def _format_articles(articles: list[dict]) -> str:
    """Format the retrieved articles as a string for the LLM."""
    return "\n\n".join(
        f"[Article {i}] (score {a['score']})\n{a['content']}"
        for i, a in enumerate(articles, start=1)
    )


def knowledge_node(state: SupportState) -> dict:
    """Search the articles, answer, and return only the fields to write to the State."""
    question = _latest_user_message(state)

    # (1) Search
    result = search_knowledge(question)
    if not result["ok"]:
        return {
            "confidence": 0.0,
            "retrieved_articles": [],
            "escalation_reason": result["error"],
        }

    data = result["data"]
    # Keep only titles and scores in the State, not the body (for checking in logs and tests)
    retrieved = [
        {"article_id": a["article_id"], "title": a["title"], "score": a["score"]}
        for a in data["articles"]
    ]

    # (2) Low confidence -> escalate without answering
    if not data["is_confident"]:
        return {
            "confidence": data["confidence"],
            "retrieved_articles": retrieved,
            "escalation_reason": (
                f"Low knowledge confidence ({data['confidence']:.2f} < {CONFIDENCE_THRESHOLD})"
            ),
        }

    # (3) The LLM answers based on the articles
    user_prompt = (
        f"Past context: {state.get('past_context') or 'None'}\n\n"
        f"Help-center articles:\n{_format_articles(data['articles'])}\n\n"
        f"Customer message:\n{question}"
    )
    llm = get_llm().with_structured_output(KnowledgeAnswer)
    answer = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT.format(
            preference_rules=preference_rules(state.get("customer_preferences", {})))),
        HumanMessage(content=user_prompt),
    ])

    update = {
        "messages": [AIMessage(content=answer.answer, name="knowledge_agent")],
        "confidence": data["confidence"],
        "retrieved_articles": retrieved,
    }
    if answer.needs_escalation:
        # Return the guidance and also hand off to a human
        update["escalation_reason"] = answer.escalation_reason or "Article requires a human"
    else:
        update["escalation_reason"] = None
        update["resolution"] = "resolved"
    return update
