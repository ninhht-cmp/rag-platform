"""
app/plugins/__init__.py
────────────────────────
Register all use-case plugins.
Import this module at application startup.

To add a new use case:
1. Create contracts/<id>.yaml (optional, for docs)
2. Add a UseCasePlugin block below
3. Done — platform auto-wires routing, RBAC, evaluation
"""
from __future__ import annotations

from app.core.plugin_registry import (
    EvalThresholds,
    PluginStatus,
    RBACRule,
    RetrievalConfig,
    UseCasePlugin,
    registry,
)


def register_all_plugins() -> None:
    """Call once at startup. Safe to call multiple times (idempotent)."""

    # ──────────────────────────────────────────────────────────────
    # 1. INTERNAL KNOWLEDGE BASE
    # ──────────────────────────────────────────────────────────────
    registry.register(UseCasePlugin(
        id="knowledge_base",
        name="Internal Knowledge Base",
        description="Company policies, SOPs, onboarding guides, technical docs",
        collection_name="uc_knowledge_base",
        status=PluginStatus.PRODUCTION,
        intent_patterns=[
            r"chính sách|policy|quy trình|sop|procedure",
            r"onboarding|hướng dẫn|how to|làm thế nào",
            r"quy định|regulation|internal|nội bộ",
            r"benefit|phúc lợi|leave|nghỉ phép",
        ],
        system_prompt_path="knowledge_base_system.j2",
        retrieval=RetrievalConfig(
            chunk_size=512,
            chunk_overlap=64,
            top_k=5,
            score_threshold=0.3,
            hybrid_search=True,
            reranker_enabled=True,
        ),
        rbac=RBACRule(
            allowed_roles=["admin", "user", "support_agent", "sales_rep", "analyst"],
        ),
        eval_thresholds=EvalThresholds(
            faithfulness=0.85,
            answer_relevancy=0.80,
            context_recall=0.75,
        ),
        monthly_token_budget=5_000_000,
        citation_required=True,
        escalation_pattern=(
            r"(?i).*(termination|resignation|legal action|lawsuit"
            r"|legal advice|tư vấn pháp lý|sue|attorney).*"
        ),
    ))

    # ──────────────────────────────────────────────────────────────
    # 2. CUSTOMER SUPPORT BOT
    # ──────────────────────────────────────────────────────────────
    registry.register(UseCasePlugin(
        id="customer_support",
        name="Customer Support Bot",
        description="Handle tier-1 customer tickets, FAQ, troubleshooting",
        collection_name="uc_customer_support",
        status=PluginStatus.PRODUCTION,
        intent_patterns=[
            r"lỗi|bug|error|không hoạt động|broken|không chạy",
            r"hoàn tiền|refund|cancel|hủy|return|đổi trả",
            r"hỗ trợ|support|help|giúp|tôi cần",
            r"tài khoản|account|login|đăng nhập|password|mật khẩu",
            r"billing|thanh toán|invoice|hóa đơn",
        ],
        system_prompt_path="customer_support_system.j2",
        retrieval=RetrievalConfig(
            chunk_size=400,
            chunk_overlap=50,
            top_k=4,
            score_threshold=0.3,
            hybrid_search=True,
            bm25_weight=0.40,    # higher BM25 — exact keyword match matters for support
            semantic_weight=0.60,
            reranker_enabled=True,
        ),
        rbac=RBACRule(
            allowed_roles=["*"],   # public-facing
        ),
        eval_thresholds=EvalThresholds(
            faithfulness=0.82,
            answer_relevancy=0.78,
            context_recall=0.72,
        ),
        monthly_token_budget=10_000_000,
        agent_tools=["create_ticket", "lookup_order", "check_account_status"],
        human_in_loop_triggers=["refund_amount_over_500k", "account_suspension", "data_deletion"],
        citation_required=False,        # customers don't need source citations
        escalation_pattern=r"urgent|khẩn cấp|manager|supervisor|complain|khiếu nại|lawsuit|sue",
    ))

    # ──────────────────────────────────────────────────────────────
    # 3. DOCUMENT Q&A
    # ──────────────────────────────────────────────────────────────
    registry.register(UseCasePlugin(
        id="document_qa",
        name="Document Q&A",
        description="Upload any document and ask questions about it",
        collection_name="uc_document_qa",
        status=PluginStatus.PRODUCTION,
        intent_patterns=[
            r"hợp đồng|contract|agreement|nda|mou",
            r"báo cáo|report|analysis|phân tích",
            r"tài liệu|document|file|pdf|docx",
            r"điều khoản|clause|term|condition",
        ],
        system_prompt_path="document_qa_system.j2",
        retrieval=RetrievalConfig(
            chunk_size=600,         # larger chunks for better context in docs
            chunk_overlap=100,
            top_k=6,
            score_threshold=0.3,
            hybrid_search=True,
            reranker_enabled=True,
        ),
        rbac=RBACRule(
            allowed_roles=["admin", "analyst", "user"],
            metadata_filters={"user_id": "${user.id}"},   # per-user isolation
        ),
        eval_thresholds=EvalThresholds(
            faithfulness=0.88,      # higher bar — document accuracy critical
            answer_relevancy=0.82,
            context_recall=0.78,
        ),
        monthly_token_budget=3_000_000,
        citation_required=True,
        escalation_pattern=(
            r"(?i).*(lawsuit|legal action|legal advice|tư vấn pháp lý"
            r"|sue|attorney|lawyer|complaint|sign|ký kết).*"
        ),
    ))

    # ──────────────────────────────────────────────────────────────
    # 4. SALES AUTOMATION
    # ──────────────────────────────────────────────────────────────
    registry.register(UseCasePlugin(
        id="sales_automation",
        name="Sales Automation",
        description="Prospect research, email drafting, battle cards, CRM automation",
        collection_name="uc_sales_automation",
        status=PluginStatus.PRODUCTION,
        intent_patterns=[
            r"khách hàng|prospect|lead|client|customer",
            r"email|pitch|proposal|đề xuất|offer",
            r"competitor|đối thủ|comparison|vs\.|versus",
            r"pricing|giá|quote|báo giá|discount|chiết khấu",
            r"deal|opportunity|pipeline|forecast|dự báo",
        ],
        system_prompt_path="sales_automation_system.j2",
        retrieval=RetrievalConfig(
            chunk_size=512,
            chunk_overlap=64,
            top_k=5,
            score_threshold=0.3,
            hybrid_search=True,
            reranker_enabled=True,
        ),
        rbac=RBACRule(
            allowed_roles=["admin", "sales_rep"],
        ),
        eval_thresholds=EvalThresholds(
            faithfulness=0.80,
            answer_relevancy=0.80,
            context_recall=0.72,
        ),
        monthly_token_budget=8_000_000,
        agent_tools=["web_search", "crm_lookup", "draft_email", "create_crm_activity"],
        human_in_loop_triggers=["send_email", "update_pricing", "contract_send"],
        citation_required=True,
        escalation_pattern=r"",
    ))
