"""Account Agent。設計書の 3.3 節。

ツールで CultPass DB を照会・操作し、ユーザー個別の依頼を処理する。
グラフ上では account_node ⇄ account_tools_node のループになる（ループは workflow.py で組む）。

  account_node
    ① get_user_profile(user_id) をコードで直接呼ぶ（LLM に任せない）
       いない／ブロック済み → escalation_reason を入れて終わり（LLM は呼ばない）
    ② LLM（bind_tools）→ tool_calls があれば account_tools_node へ
       tool_calls がなければ最終回答（resolution = "resolved"）
  account_tools_node
    ③ ツールを実行して ToolMessage を返す。user_id は必ず State の値に差し替える
       ok=False のツールがあれば escalation_reason を入れる
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

# LLM に渡すツール。get_user_profile は①でコードが呼ぶので渡さない
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
    """ユーザーを確認してから LLM を呼び、State に書き込む項目だけを返す。"""
    user_id = state.get("user_id", "")

    # ① ユーザーの確認（コードで必ず行う）
    profile = get_user_profile(user_id)
    if not profile["ok"]:
        return {"escalation_reason": profile["error"]}
    if profile["data"]["is_blocked"]:
        return {"escalation_reason": "User account is blocked"}

    # ② LLM にツールを渡して呼ぶ
    llm = get_llm().bind_tools(ACCOUNT_TOOLS)
    system = SystemMessage(content=SYSTEM_PROMPT.format(
        today=date.today(), user_id=user_id))
    response = llm.invoke([system] + state["messages"])
    response.name = "account_agent"

    if response.tool_calls:
        # ツールを呼ぶ → account_tools_node へ（まだ終わらない）
        return {"messages": [response], "escalation_reason": None}

    # tool_calls がない = 最終回答
    return {"messages": [response], "escalation_reason": None, "resolution": "resolved"}


def account_tools_node(state: SupportState) -> dict:
    """直前の AIMessage の tool_calls を実行し、ToolMessage を返す。"""
    last_message: AIMessage = state["messages"][-1]
    tool_messages = []
    escalation_reason = None

    for call in last_message.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            result = {"ok": False, "error": f"Unknown tool: {call['name']}"}
        else:
            # LLM が他人の user_id を指定しても、State のユーザーしか操作できないようにする
            # user_id を State の値で上書きすることで、LLM が他人の ID を指定しても無視される。
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
