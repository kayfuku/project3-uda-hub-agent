"""UDA-Hub graph (StateGraph). Design doc, Chapter 2.

  START → load_memory → classifier ─┬─ knowledge ──────────────┬─→ save_memory → END
                                    ├─ account ⇄ account_tools ─┤
                                    └─ escalation ←─────────────┘ (on escalation)

Each agent in agentic/agents/ is "a function that takes the State and returns the fields to update".
This file only adds them as nodes and connects them with conditional edges.

load_memory and save_memory are helper nodes (not agents). They read and write long-term memory in udahub.db.

Logs (logs/workflow.jsonl) are written only in this file. Logging is kept out of the agent code;
instead, each node is wrapped in a small function (e.g. classifier_step) that "calls the agent and writes one log line".
"""
import json

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic.agents.account import account_node, account_tools_node
from agentic.agents.classifier import classifier_node
from agentic.agents.escalation import escalation_node
from agentic.agents.knowledge import knowledge_node
from agentic.logging_utils import log_event
from agentic.state import SupportState
from agentic.tools.memory import (
    create_ticket,
    get_ticket_info,
    load_past_context,
    save_handoff_summary,
    save_messages,
    update_ticket_status,
)


# ---------------------------------------------------------------
# Helper nodes (not agents)
# ---------------------------------------------------------------
def load_memory_node(state: SupportState, config: RunnableConfig) -> dict:
    """Start of a turn. Load the ticket info and long-term memory, and reset the previous turn's results.

    - chat_interface only passes messages, so we look up the ticket in udahub.db
      by thread_id (= ticket_id) to get user_id and ticket_metadata.
    - Past tickets of the same user become past_context (long-term memory).
    - MemorySaver keeps the whole State, so without a reset, leftovers such as
      the previous turn's escalation_reason would send us to the wrong node.
    Do not stop if a DB function returns an error (if user_id is empty, Account will escalate).
    """
    ticket_id = config["configurable"]["thread_id"]

    user_id, ticket_metadata, past_context, past_tickets = "", {}, "", 0
    info = get_ticket_info(ticket_id)
    if info["ok"]:
        data = info["data"]
        user_id = data["external_user_id"]
        ticket_metadata = {key: data[key] for key in
                           ("channel", "created_at", "main_issue_type", "tags")}
        past = load_past_context(user_id, current_ticket_id=ticket_id)
        if past["ok"]:
            past_context = past["data"]["summary"]
            past_tickets = len(past["data"]["tickets"])

    log_event(ticket_id, "load_memory", "memory_loaded", user_id=user_id,
              past_tickets=past_tickets, error=None if info["ok"] else info["error"])

    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "ticket_metadata": ticket_metadata,
        "past_context": past_context,
        # Reset the previous turn's results below
        "category": "",
        "urgency": "",
        "classification_reason": "",
        "retrieved_articles": [],
        "confidence": 0.0,
        "escalation_reason": None,
        "handoff_summary": "",
        "resolution": "",
    }


def save_memory_node(state: SupportState) -> dict:
    """End of a turn. Save the conversation and results to udahub.db (long-term memory).

    This is the only node that writes to the DB (agents never write to udahub.db).
    status is overwritten with this turn's result. However, once a ticket is escalated it
    stays escalated (so a pending human handoff is not hidden by a later 'resolved' turn).
    """
    ticket_id = state["ticket_id"]

    results = []  # Results of the DB functions (to log any errors)

    # Already-saved messages are skipped by message.id, so it is fine to pass all of them every turn
    results.append(save_messages(ticket_id, state["messages"]))

    if state.get("resolution") == "escalated" and state.get("handoff_summary"):
        results.append(save_handoff_summary(
            ticket_id, state["handoff_summary"]))

    status = state.get("resolution") or "open"
    info = get_ticket_info(ticket_id)
    if info["ok"] and info["data"]["status"] == "escalated":
        status = "escalated"

    results.append(update_ticket_status(
        ticket_id,
        status,
        main_issue_type=state.get("category") or None,
        tags=f"urgency:{state['urgency']}" if state.get("urgency") else None,
    ))

    errors = [r["error"] for r in results if not r["ok"]]
    log_event(ticket_id, "save_memory", "saved", status=status,
              resolution=state.get("resolution"), errors=errors)
    return {}


# ---------------------------------------------------------------
# Agent nodes (call the agent and write one log line)
# ---------------------------------------------------------------
def classifier_step(state: SupportState) -> dict:
    update = classifier_node(state)
    log_event(state["ticket_id"], "classifier", "classified",
              category=update["category"], urgency=update["urgency"],
              reason=update["classification_reason"])
    return update


def knowledge_step(state: SupportState) -> dict:
    update = knowledge_node(state)
    articles = update.get("retrieved_articles", [])
    log_event(state["ticket_id"], "knowledge", "retrieved",
              top_article=articles[0]["title"] if articles else None,
              confidence=update.get("confidence"),
              escalation_reason=update.get("escalation_reason"))
    return update


def account_step(state: SupportState) -> dict:
    update = account_node(state)
    messages = update.get("messages", [])
    log_event(state["ticket_id"], "account", "account_step",
              tool_calls=[c["name"]
                          for c in messages[-1].tool_calls] if messages else [],
              escalation_reason=update.get("escalation_reason"))
    return update


def account_tools_step(state: SupportState) -> dict:
    update = account_tools_node(state)
    for message in update["messages"]:  # one line per tool
        result = json.loads(message.content)
        log_event(state["ticket_id"], "account_tools", "tool_called",
                  tool=message.name, ok=result["ok"], error=result.get("error"))
    return update


def escalation_step(state: SupportState) -> dict:
    update = escalation_node(state)
    log_event(state["ticket_id"], "escalation", "escalated",
              reason=update["escalation_reason"], urgency=state.get("urgency"))
    return update


# ---------------------------------------------------------------
# Routing (conditional edges). The returned string = name of the next node
# ---------------------------------------------------------------
def route_after_classifier(state: SupportState) -> str:
    return state["category"]  # "knowledge" / "account" / "escalation"


def route_after_knowledge(state: SupportState) -> str:
    return "escalation" if state.get("escalation_reason") else "save_memory"


def route_after_account(state: SupportState) -> str:
    if state.get("escalation_reason"):
        return "escalation"
    if state["messages"][-1].tool_calls:
        return "account_tools"
    return "save_memory"  # final answer


def route_after_account_tools(state: SupportState) -> str:
    return "escalation" if state.get("escalation_reason") else "account"


# ---------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------
def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("classifier", classifier_step)
    graph.add_node("knowledge", knowledge_step)
    graph.add_node("account", account_step)
    graph.add_node("account_tools", account_tools_step)
    graph.add_node("escalation", escalation_step)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "classifier")
    graph.add_conditional_edges("classifier", route_after_classifier,
                                ["knowledge", "account", "escalation"])
    graph.add_conditional_edges("knowledge", route_after_knowledge,
                                ["escalation", "save_memory"])
    graph.add_conditional_edges("account", route_after_account,
                                ["account_tools", "escalation", "save_memory"])
    graph.add_conditional_edges("account_tools", route_after_account_tools,
                                ["account", "escalation"])
    graph.add_edge("escalation", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer)


# 03_agentic_app.ipynb imports this. Short-term memory = MemorySaver (per thread_id)
orchestrator = build_graph(checkpointer=MemorySaver())


def new_ticket(user_id: str, channel: str = "chat") -> str:
    """Create a ticket for a new inquiry in udahub.db and return its ticket_id (= thread_id).

    Usage in notebook 03:
        ticket_id = new_ticket("f1f10d")
        chat_interface(orchestrator, ticket_id)
    """
    result = create_ticket(user_id, channel=channel)
    if not result["ok"]:
        raise ValueError(result["error"])
    return result["data"]["ticket_id"]
