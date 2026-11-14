"""Classifier Agent (supervisor). Design doc, Section 3.1.

Decides the route (category) and urgency from the ticket content and metadata.
Uses LLM structured output (a Pydantic model), so the output shape is always the same.

  Input:  latest user message, recent conversation (up to 6 messages), ticket_metadata, past_context
  Output: {"category": ..., "urgency": ..., "classification_reason": ...}
"""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.llm import get_llm
from agentic.state import SupportState

# How many recent messages to pass (to classify short replies like "Yes, please cancel it")
RECENT_MESSAGE_LIMIT = 6


class Classification(BaseModel):
    """Classifier output. The LLM answers in this shape."""
    category: Literal["knowledge", "account", "escalation"] = Field(
        description="Which agent should handle the ticket."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent the ticket is."
    )
    reason: str = Field(
        description="One short sentence explaining the decision.")


# The classification rules match the table in design doc Section 3.1
SYSTEM_PROMPT = """\
You are the classifier of UDA-Hub, a customer support system for CultPass \
(a subscription service for cultural experiences such as museums, concerts and workshops).
Read the customer's latest message and decide which agent should handle it.

category:
- "knowledge": general questions or how-to questions that do not need the customer's own data.
  Examples: "The QR code won't scan", "What does my subscription include?", "How do I change my payment method?"
  Questions about charges, payments or refunds are also "knowledge", because the account tools have no billing data.
  Examples: "I was charged twice this month", "Can I get a refund?"
- "account": the customer wants to look at or change their OWN data (reservations, subscription, account status).
  Examples: "Show my reservations", "Cancel my booking for next week", "Is my subscription active?"
  Questions about the customer's OWN plan details (their tier, how many experiences THEY can use) are also "account",
  because the quota differs per customer and only the account data has the correct number.
  Examples: "How many experiences can I use per month on my plan?", "Which plan am I on?"
- "escalation": a human must decide. Clear complaints, legal threats, or the customer asks for a human.
  Examples: "I want to talk to a manager", "I will contact my lawyer"

urgency:
- "high": the customer is angry, mentions legal action, or an event is about to start (today or tomorrow).
- "medium": the customer cannot use the service or needs an action on their account.
- "low": general questions with no time pressure.

Use the ticket metadata and the past context only as hints. The latest message is the most important.
If the latest message is a short reply (for example "yes", "no", "the first one") to a question
the assistant asked in the recent conversation, choose the category of that question.
"""


def _latest_user_message(state: SupportState) -> str:
    """Get the latest user message from messages."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _recent_conversation(state: SupportState) -> str:
    """Turn the conversation before the latest user message into a string (up to RECENT_MESSAGE_LIMIT messages).

    Only customer and AI text is used (tool calls and results are left out).
    """
    messages = state["messages"]
    # Find the position of the latest user message and use only what comes before it
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0
    )
    lines = []
    for m in messages[:last_human]:
        if isinstance(m, HumanMessage):
            lines.append(f"Customer: {m.text}")
        elif isinstance(m, AIMessage) and m.text.strip():
            lines.append(f"Assistant: {m.text}")
    return "\n".join(lines[-RECENT_MESSAGE_LIMIT:]) or "None"


def classifier_node(state: SupportState) -> dict:
    """Classify the ticket and return only the fields to write to the State."""
    user_prompt = (
        f"Ticket metadata: {state.get('ticket_metadata', {})}\n"
        f"Past context: {state.get('past_context') or 'None'}\n\n"
        f"Recent conversation:\n{_recent_conversation(state)}\n\n"
        f"Latest customer message:\n{_latest_user_message(state)}"
    )

    llm = get_llm().with_structured_output(Classification)
    result = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    return {
        "category": result.category,
        "urgency": result.urgency,
        "classification_reason": result.reason,
    }
