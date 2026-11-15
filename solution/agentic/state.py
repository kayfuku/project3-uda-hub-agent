"""State shared across the whole graph. Design doc (agentic/design/architecture.md), Chapter 4.

Imported by both the agents (agentic/agents/) and workflow.py.
It lives in its own file because putting it in workflow.py would cause a circular import.
"""
from langgraph.graph import MessagesState


class SupportState(MessagesState):
    # messages: Annotated[list[AnyMessage], add_messages]  <- inherited from MessagesState (conversation = short-term memory)
    ticket_id: str                    # = thread_id
    user_id: str                      # User ID on the CultPass side
    ticket_metadata: dict             # channel, main_issue_type, tags, etc.
    past_context: str                 # Summary of past interactions (long-term memory)
    customer_preferences: dict        # Saved preferences + ones stated this turn (long-term memory)
    new_preferences: dict             # Preferences stated this turn (save_memory saves them)
    category: str                     # Classifier result (knowledge / account / escalation)
    urgency: str                      # low / medium / high
    classification_reason: str        # Classifier's reasoning
    retrieved_articles: list[dict]    # Retrieved articles and their scores
    confidence: float                 # Knowledge agent's confidence
    escalation_reason: str | None     # Reason for escalation
    handoff_summary: str              # Handoff summary for the human agent (written by Escalation)
    resolution: str                   # resolved / escalated
