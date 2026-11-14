"""Escalation Agent。設計書の 3.4 節。

人間の担当者に引き継ぐ。担当者がすぐ対応できるよう、要約を残す。

  入力: 会話全体、escalation_reason（なければ classification_reason）、urgency、
        このターンで先に出した案内（あれば、コードで customer_message の前につなぐ）
  出力: customer_message（お客さん向け。お客さんの言語）
        handoff_summary（担当者向け。英語）
  返す項目: {"messages": [...], "handoff_summary": ..., "resolution": "escalated"}

DB への書き込み（status を escalated にするなど）は、このノードでは行わない。
DB に書く場所は save_memory（workflow.py）の 1 か所にまとめる。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agentic.agents.classifier import _latest_user_message
from agentic.agents.llm import get_llm
from agentic.state import SupportState


class EscalationOutput(BaseModel):
    """Escalation Agent の LLM の出力。

    customer_language を最初に答えさせて、返信の言語を先に決めさせる
    （これがないと、英語の問い合わせにスペイン語で返すことがあった）。
    """
    customer_language: str = Field(
        description="Language of the customer's latest message, e.g. English, Japanese.")
    customer_message: str = Field(
        description="Short message to the customer, in the customer's language.")
    handoff_summary: str = Field(
        description="Summary for the human support agent, in English.")


SYSTEM_PROMPT = """\
You are the escalation agent of UDA-Hub (customer support for CultPass).
This ticket is being handed over to a human support agent. Write two things.

customer_message (to the customer):
- Write in customer_language (the language of the customer's latest message). Keep it short, polite and calm.
- Tell them a member of our support team will follow up.
- Do not mention internal details such as confidence scores, tool names, error messages
  or that the account is "blocked". Say "our team needs to review your request" instead.
- If "Guidance already given in this turn" is not None, do not repeat it
  (it is shown to the customer together with your message).

handoff_summary (to the human agent, in English, 2-4 short lines):
- The customer's issue, the escalation reason, the urgency, and what has already been tried or answered.
"""


def _format_conversation(state: SupportState) -> str:
    """会話を、LLM に渡す文字列にする。ツールの呼び出しだけの AI メッセージは飛ばす。"""
    lines = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            lines.append(f"Customer: {m.text}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool result ({m.name}): {m.text}")
        elif isinstance(m, AIMessage) and m.text.strip():
            lines.append(f"Assistant: {m.text}")
    return "\n".join(lines)


def _guidance_in_this_turn(state: SupportState) -> str:
    """このターンで、先に AI が出した案内（最新のお客さんの発言より後の AI の本文）を返す。

    例: 二重請求 → Knowledge が「Billing History で確認できます」と案内 → ここに来る。
    chat_interface は最後のメッセージしか表示しないので、escalation_node で
    この案内を引き継ぎのメッセージの前につなぐ。案内がなければ空文字。
    """
    messages = state["messages"]
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    texts = [m.text for m in messages[last_human + 1:]
             if isinstance(m, AIMessage) and m.text.strip()]
    return "\n".join(texts)


def escalation_node(state: SupportState) -> dict:
    """引き継ぎのメッセージと要約を作り、State に書き込む項目だけを返す。"""
    # Classifier が直接ここに回した場合は escalation_reason がないので、分類の理由を使う
    reason = (state.get("escalation_reason")
              or state.get("classification_reason")
              or "Not specified")

    guidance = _guidance_in_this_turn(state)
    user_prompt = (
        f"Escalation reason: {reason}\n"
        f"Urgency: {state.get('urgency') or 'unknown'}\n"
        f"Customer user_id: {state.get('user_id') or 'unknown'}\n"
        f"Past context: {state.get('past_context') or 'None'}\n\n"
        f"Conversation:\n{_format_conversation(state)}\n\n"
        f"Guidance already given in this turn:\n{guidance or 'None'}\n\n"
        f"Customer's latest message:\n{_latest_user_message(state)}"
    )
    llm = get_llm().with_structured_output(EscalationOutput)
    result = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    # お客さんに見えるのは最後のメッセージだけなので、このターンの案内を前につなぐ（コードで確実に）
    content = f"{guidance}\n\n{result.customer_message}" if guidance else result.customer_message

    return {
        "messages": [AIMessage(content=content, name="escalation_agent")],
        "escalation_reason": reason,
        "handoff_summary": result.handoff_summary,
        "resolution": "escalated",
    }
