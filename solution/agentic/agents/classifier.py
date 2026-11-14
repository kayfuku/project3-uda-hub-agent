"""Classifier Agent（スーパーバイザー）。設計書の 3.1 節。

チケットの内容とメタデータから、行き先（category）と緊急度（urgency）を決める。
LLM の構造化出力（Pydantic モデル）を使うので、返り値の形が毎回そろう。

  入力: 最新のユーザーメッセージ、ticket_metadata、past_context
  出力: {"category": ..., "urgency": ..., "classification_reason": ...}
"""
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.llm import get_llm
from agentic.state import SupportState


class Classification(BaseModel):
    """Classifier の出力。LLM はこの形で答える。"""
    category: Literal["knowledge", "account", "escalation"] = Field(
        description="Which agent should handle the ticket."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent the ticket is."
    )
    reason: str = Field(
        description="One short sentence explaining the decision.")


# 分類の基準は設計書 3.1 節の表と同じ
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
- "escalation": a human must decide. Clear complaints, legal threats, or the customer asks for a human.
  Examples: "I want to talk to a manager", "I will contact my lawyer"

urgency:
- "high": the customer is angry, mentions legal action, or an event is about to start (today or tomorrow).
- "medium": the customer cannot use the service or needs an action on their account.
- "low": general questions with no time pressure.

Use the ticket metadata and the past context only as hints. The latest message is the most important.
"""


def _latest_user_message(state: SupportState) -> str:
    """messages の中から、いちばん新しいユーザーの発言を取り出す。"""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def classifier_node(state: SupportState) -> dict:
    """チケットを分類し、State に書き込む項目だけを返す。"""
    user_prompt = (
        f"Ticket metadata: {state.get('ticket_metadata', {})}\n"
        f"Past context: {state.get('past_context') or 'None'}\n\n"
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
