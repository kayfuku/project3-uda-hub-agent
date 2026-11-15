"""Escalation Agent. Design doc, Section 3.4.

Hands off to a human agent, leaving a summary so they can act right away.

  Input:   full conversation, escalation_reason (or classification_reason if missing), urgency,
           customer_preferences (reply language, contact method, etc.),
           any guidance already given this turn (if present, the code prepends it to customer_message)
  Output:  customer_message (for the customer, in the customer's language)
           handoff_summary (for the human agent, in English)
  Returns: {"messages": [...], "handoff_summary": ..., "resolution": "escalated"}

This node does not write to the DB (e.g. setting status to escalated).
All DB writes are kept in one place: save_memory (workflow.py).
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agentic.agents.classifier import _latest_user_message
from agentic.agents.llm import get_llm, preference_rules
from agentic.state import SupportState


class EscalationOutput(BaseModel):
    """LLM output of the Escalation Agent.

    customer_language is answered first so the reply language is decided up front
    (without it, the agent sometimes replied in Spanish to an English inquiry).
    """
    customer_language: str = Field(
        description="Language for the reply: the language in the customer preferences if set, "
                    "otherwise the language of the customer's latest message. e.g. English, Japanese.")
    customer_message: str = Field(
        description="Short message to the customer, in customer_language.")
    handoff_summary: str = Field(
        description="Summary for the human support agent, in English.")


SYSTEM_PROMPT = """\
You are the escalation agent of UDA-Hub (customer support for CultPass).
This ticket is being handed over to a human support agent. Write two things.

{preference_rules}
customer_message (to the customer):
- Write in customer_language. Keep it short, polite and calm.
- Tell them a member of our support team will follow up.
- Do not mention internal details such as confidence scores, tool names, error messages
  or that the account is "blocked". Say "our team needs to review your request" instead.
- If "Guidance already given in this turn" is not None, do not repeat it
  (it is shown to the customer together with your message).

handoff_summary (to the human agent, in English, 2-4 short lines):
- The customer's issue, the escalation reason, the urgency, and what has already been tried or answered.
- Add the saved customer preferences that matter for the follow-up (for example preferred language or contact method).
  Use only what is written in "Customer preferences". If it is None, do not mention preferences.
"""


def _format_conversation(state: SupportState) -> str:
    """Format the conversation as a string for the LLM. AI messages with only tool calls are skipped."""
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
    """Return guidance the AI already gave this turn (AI text after the latest customer message).

    Example: double charge -> Knowledge says "You can check it in Billing History" -> comes here.
    chat_interface shows only the last message, so escalation_node
    prepends this guidance to the handoff message. Empty string if there is no guidance.
    """
    messages = state["messages"]
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    texts = [m.text for m in messages[last_human + 1:]
             if isinstance(m, AIMessage) and m.text.strip()]
    return "\n".join(texts)


def escalation_node(state: SupportState) -> dict:
    """Create the handoff message and summary, and return only the fields to write to the State."""
    # If the Classifier routed here directly there is no escalation_reason, so use the classification reason
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
        SystemMessage(content=SYSTEM_PROMPT.format(
            preference_rules=preference_rules(state.get("customer_preferences", {})))),
        HumanMessage(content=user_prompt),
    ])

    # The customer sees only the last message, so prepend this turn's guidance (done in code to be reliable)
    content = f"{guidance}\n\n{result.customer_message}" if guidance else result.customer_message

    return {
        "messages": [AIMessage(content=content, name="escalation_agent")],
        "escalation_reason": reason,
        "handoff_summary": result.handoff_summary,
        "resolution": "escalated",
    }
