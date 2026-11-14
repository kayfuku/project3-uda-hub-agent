"""Account Agent. Design doc, Section 3.3.

Uses tools to query and update the CultPass DB and handle user-specific requests.
In the graph this is an account_node <-> account_tools_node loop (the loop is built in workflow.py).

  account_node
    (1) Call get_user_profile(user_id) directly in code (not left to the LLM)
        Not found / blocked -> set escalation_reason and stop (the LLM is not called)
    (2) LLM (bind_tools) -> if there are tool_calls, go to account_tools_node
        If there are no tool_calls, it is the final answer (resolution = "resolved")
  account_tools_node
    (3) Run the tools and return ToolMessages. user_id is always replaced with the State value
        If any tool returns ok=False, set escalation_reason
"""
import json
from datetime import date

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agentic.agents.llm import get_llm
from agentic.state import SupportState
from agentic.tools.cultpass_tools import (
    cancel_reservation_tool,
    get_reservations_tool,
    get_subscription_tool,
    get_user_profile,
)

# Tools given to the LLM. get_user_profile is not included because the code calls it in step (1)
ACCOUNT_TOOLS = [get_subscription_tool,
                 get_reservations_tool, cancel_reservation_tool]
TOOLS_BY_NAME = {t.name: t for t in ACCOUNT_TOOLS}

SYSTEM_PROMPT = """\
You are the account agent of UDA-Hub, helping a CultPass customer with their own account.
Today is {today}. The customer's user_id is "{user_id}". Never ask the customer for their user ID.

- Use the tools to look up or change the customer's data. Do not guess data you have not looked up.
- To cancel a reservation, call get_reservations first to find the reservation_id.
  If it is not clear which reservation the customer means, ask them instead of cancelling.
- After a cancellation, if within_24h is true, tell the customer that one experience credit was used up.
  If it is false, tell them the credit was returned.
- Reply in the same language as the customer, in a short and friendly way.
"""


def account_node(state: SupportState) -> dict:
    """Check the user, then call the LLM, and return only the fields to write to the State."""
    user_id = state.get("user_id", "")

    # (1) Check the user (always done in code)
    profile = get_user_profile(user_id)
    if not profile["ok"]:
        return {"escalation_reason": profile["error"]}
    if profile["data"]["is_blocked"]:
        return {"escalation_reason": "User account is blocked"}

    # (2) Call the LLM with the tools
    llm = get_llm().bind_tools(ACCOUNT_TOOLS)
    system = SystemMessage(content=SYSTEM_PROMPT.format(
        today=date.today(), user_id=user_id))
    response = llm.invoke([system] + state["messages"])
    response.name = "account_agent"

    if response.tool_calls:
        # Tool call -> go to account_tools_node (not finished yet)
        return {"messages": [response], "escalation_reason": None}

    # No tool_calls = final answer
    return {"messages": [response], "escalation_reason": None, "resolution": "resolved"}


def account_tools_node(state: SupportState) -> dict:
    """Run the tool_calls in the last AIMessage and return ToolMessages."""
    last_message: AIMessage = state["messages"][-1]
    tool_messages = []
    escalation_reason = None

    for call in last_message.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            result = {"ok": False, "error": f"Unknown tool: {call['name']}"}
        else:
            # Only the State's user can be operated on, even if the LLM specifies another user_id
            # Overwriting user_id with the State value means any other ID from the LLM is ignored.
            args = {**call["args"], "user_id": state.get("user_id", "")}
            result = tool.invoke(args)

        if not result["ok"] and escalation_reason is None:
            escalation_reason = f"{call['name']} failed: {result['error']}"

        tool_messages.append(ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=call["id"],
            name=call["name"],
        ))

    return {"messages": tool_messages, "escalation_reason": escalation_reason}
