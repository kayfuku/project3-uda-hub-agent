"""グラフ全体で共有する状態（State）。設計書（agentic/design/architecture.md）の第4章。

エージェント（agentic/agents/）と workflow.py の両方が import する。
workflow.py に置くと循環 import になるので、別ファイルにしている。
"""
from langgraph.graph import MessagesState


class SupportState(MessagesState):
    # messages: list[AnyMessage]  ← MessagesState から継承（会話 = 短期メモリ）
    ticket_id: str                    # = thread_id
    user_id: str                      # CultPass 側のユーザー ID
    ticket_metadata: dict             # channel, main_issue_type, tags など
    past_context: str                 # 過去のやり取りの要約（長期メモリ）
    category: str                     # Classifier の分類結果（knowledge / account / escalation）
    urgency: str                      # low / medium / high
    classification_reason: str        # Classifier の判断理由
    retrieved_articles: list[dict]    # 検索した記事とスコア
    confidence: float                 # Knowledge の信頼度
    escalation_reason: str | None     # エスカレーションの理由
    handoff_summary: str              # 担当者向けの引き継ぎの要約（Escalation が書く）
    resolution: str                   # resolved / escalated
