"""Knowledge Agent。設計書の 3.2 節と第5章。

ナレッジ記事を検索し、記事の内容だけに基づいて回答する。

  ① search_knowledge(最新のユーザーメッセージ) → confidence と上位 3 件
  ② 検索の失敗、または is_confident が False → 回答は作らず、escalation_reason を入れて返す
  ③ True → 上位 3 件を LLM に渡して回答させる（構造化出力）
     記事が人への引き継ぎを指示しているときは、LLM が needs_escalation=True を返す

エスカレーションに進むかどうかは、返した escalation_reason を見てグラフ側（workflow.py）で決める。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.classifier import _latest_user_message
from agentic.agents.llm import get_llm
from agentic.state import SupportState
from agentic.tools.knowledge_search import CONFIDENCE_THRESHOLD, search_knowledge


class KnowledgeAnswer(BaseModel):
    """Knowledge Agent の LLM の出力。"""
    answer: str = Field(
        description="The reply to the customer, based only on the articles.")
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

- Reply in the same language as the customer, in a short and friendly way.
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
    """検索した記事を、LLM に渡す文字列にする。"""
    return "\n\n".join(
        f"[Article {i}] (score {a['score']})\n{a['content']}"
        for i, a in enumerate(articles, start=1)
    )


def knowledge_node(state: SupportState) -> dict:
    """記事を検索して回答し、State に書き込む項目だけを返す。"""
    question = _latest_user_message(state)

    # ① 検索
    result = search_knowledge(question)
    if not result["ok"]:
        return {
            "confidence": 0.0,
            "retrieved_articles": [],
            "escalation_reason": result["error"],
        }

    data = result["data"]
    # State には本文を入れず、タイトルとスコアだけ残す（ログやテストで確認するため）
    retrieved = [
        {"article_id": a["article_id"], "title": a["title"], "score": a["score"]}
        for a in data["articles"]
    ]

    # ② 信頼度が低い → 回答を作らずにエスカレーション
    if not data["is_confident"]:
        return {
            "confidence": data["confidence"],
            "retrieved_articles": retrieved,
            "escalation_reason": (
                f"Low knowledge confidence ({data['confidence']:.2f} < {CONFIDENCE_THRESHOLD})"
            ),
        }

    # ③ 記事をもとに LLM が回答する
    user_prompt = (
        f"Past context: {state.get('past_context') or 'None'}\n\n"
        f"Help-center articles:\n{_format_articles(data['articles'])}\n\n"
        f"Customer message:\n{question}"
    )
    llm = get_llm().with_structured_output(KnowledgeAnswer)
    answer = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    update = {
        "messages": [AIMessage(content=answer.answer, name="knowledge_agent")],
        "confidence": data["confidence"],
        "retrieved_articles": retrieved,
    }
    if answer.needs_escalation:
        # 案内は返したうえで、人に引き継ぐ
        update["escalation_reason"] = answer.escalation_reason or "Article requires a human"
    else:
        update["escalation_reason"] = None
        update["resolution"] = "resolved"
    return update
