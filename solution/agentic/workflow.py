"""UDA-Hub のグラフ（StateGraph）。設計書の第2章。

  START → load_memory → classifier ─┬─ knowledge ──────────────┬─→ save_memory → END
                                    ├─ account ⇄ account_tools ─┤
                                    └─ escalation ←─────────────┘（エスカレーションのとき）

各エージェントは agentic/agents/ にある「State を受け取り、更新する項目を返す関数」。
ここでは、それをノードとして並べ、条件付きエッジでつなぐだけにする。

load_memory と save_memory は補助ノード（エージェントではない）。udahub.db の長期メモリを読み書きする。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic.agents.account import account_node, account_tools_node
from agentic.agents.classifier import classifier_node
from agentic.agents.escalation import escalation_node
from agentic.agents.knowledge import knowledge_node
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
# 補助ノード（エージェントではない）
# ---------------------------------------------------------------
def load_memory_node(state: SupportState, config: RunnableConfig) -> dict:
    """ターンの入口。チケットの情報と長期メモリを読み、前のターンの結果をリセットする。

    - chat_interface は messages しか渡さないので、thread_id（= ticket_id）から
      udahub.db のチケットを引いて、user_id と ticket_metadata を知る。
    - 同じユーザーの過去のチケットを past_context にする（長期メモリ）。
    - MemorySaver は State を丸ごと残すので、リセットしないと
      前のターンの escalation_reason などで、間違った行き先に進んでしまう。
    DB の関数がエラーを返しても止めない（user_id が空なら、Account がエスカレーションする）。
    """
    ticket_id = config["configurable"]["thread_id"]

    user_id, ticket_metadata, past_context = "", {}, ""
    info = get_ticket_info(ticket_id)
    if info["ok"]:
        data = info["data"]
        user_id = data["external_user_id"]
        ticket_metadata = {key: data[key] for key in
                           ("channel", "created_at", "main_issue_type", "tags")}
        past = load_past_context(user_id, current_ticket_id=ticket_id)
        if past["ok"]:
            past_context = past["data"]["summary"]

    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "ticket_metadata": ticket_metadata,
        "past_context": past_context,
        # 以下は前のターンの結果のリセット
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
    """ターンの出口。会話と結果を udahub.db に保存する（長期メモリ）。

    DB に書くのはこのノードだけ（エージェントは udahub.db に書かない）。
    status はそのターンの結果で上書きする。ただし、一度 escalated になったチケットは
    escalated のままにする（担当者の対応待ちが、次のターンの resolved で見えなくならないように）。
    """
    ticket_id = state["ticket_id"]

    # 同じ発言は message.id で飛ばすので、毎ターン全部渡してよい
    save_messages(ticket_id, state["messages"])

    if state.get("resolution") == "escalated" and state.get("handoff_summary"):
        save_handoff_summary(ticket_id, state["handoff_summary"])

    status = state.get("resolution") or "open"
    info = get_ticket_info(ticket_id)
    if info["ok"] and info["data"]["status"] == "escalated":
        status = "escalated"

    update_ticket_status(
        ticket_id,
        status,
        main_issue_type=state.get("category") or None,
        tags=f"urgency:{state['urgency']}" if state.get("urgency") else None,
    )
    return {}


# ---------------------------------------------------------------
# ルーティング（条件付きエッジ）。返す文字列 = 次のノードの名前
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
    return "save_memory"  # 最終回答


def route_after_account_tools(state: SupportState) -> str:
    return "escalation" if state.get("escalation_reason") else "account"


# ---------------------------------------------------------------
# グラフの組み立て
# ---------------------------------------------------------------
def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("account", account_node)
    graph.add_node("account_tools", account_tools_node)
    graph.add_node("escalation", escalation_node)
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


# 03_agentic_app.ipynb はこれを import する。短期メモリ = MemorySaver（thread_id ごと）
orchestrator = build_graph(checkpointer=MemorySaver())


def new_ticket(user_id: str, channel: str = "chat") -> str:
    """新しい問い合わせのチケットを udahub.db に作り、ticket_id（= thread_id）を返す。

    03 ノートブックでの使い方:
        ticket_id = new_ticket("f1f10d")
        chat_interface(orchestrator, ticket_id)
    """
    result = create_ticket(user_id, channel=channel)
    if not result["ok"]:
        raise ValueError(result["error"])
    return result["data"]["ticket_id"]
