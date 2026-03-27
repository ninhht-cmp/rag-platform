"""
app/core/plugin_registry.py
────────────────────────────
Plugin registry — the extensibility engine.
Add a new use-case by registering a UseCase plugin.
Zero infra changes required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PluginStatus(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class RBACRule:
    """Access control: which roles can query which collection."""

    allowed_roles: list[str]
    metadata_filters: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    score_threshold: float = 0.70
    hybrid_search: bool = True
    bm25_weight: float = 0.30
    semantic_weight: float = 0.70
    reranker_enabled: bool = True


@dataclass
class EvalThresholds:
    faithfulness: float = 0.82
    answer_relevancy: float = 0.78
    context_recall: float = 0.75


@dataclass
class UseCasePlugin:
    """
    Contract every use-case must satisfy.
    Fill this in → platform auto-wires everything.
    """

    id: str  # unique slug: "knowledge_base"
    name: str  # human label
    description: str
    collection_name: str  # Qdrant collection — MUST be unique
    status: PluginStatus
    intent_patterns: list[str]  # regex patterns for fast routing
    system_prompt_path: str  # path to Jinja2 template
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rbac: RBACRule = field(default_factory=lambda: RBACRule(allowed_roles=["*"]))
    eval_thresholds: EvalThresholds = field(default_factory=EvalThresholds)
    monthly_token_budget: int = 5_000_000
    agent_tools: list[str] = field(default_factory=list)
    human_in_loop_triggers: list[str] = field(default_factory=list)
    citation_required: bool = True
    escalation_pattern: str = ""

    def matches_intent(self, query: str) -> bool:
        """Fast pattern-based intent matching before LLM classification."""
        q = query.lower()
        return any(re.search(p, q) for p in self.intent_patterns)


class PluginRegistry:
    """
    Singleton registry.
    Hot-reload supported: call register() at any time.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, UseCasePlugin] = {}

    def register(self, plugin: UseCasePlugin) -> None:
        if plugin.id in self._plugins:
            logger.warning("plugin.overwrite", plugin_id=plugin.id)
        self._plugins[plugin.id] = plugin
        logger.info(
            "plugin.registered",
            plugin_id=plugin.id,
            status=plugin.status,
            collection=plugin.collection_name,
        )

    def get(self, plugin_id: str) -> UseCasePlugin | None:
        return self._plugins.get(plugin_id)

    def get_active(self) -> list[UseCasePlugin]:
        """Return only production-status plugins."""
        return [p for p in self._plugins.values() if p.status == PluginStatus.PRODUCTION]

    def route_by_intent(self, query: str, user_roles: list[str]) -> UseCasePlugin | None:
        """
        Fast intent routing:
        1. Pattern match (free, instant)
        2. Role filter (RBAC)
        Returns None if no match → caller should use LLM classifier fallback.
        """
        for plugin in self.get_active():
            if not plugin.matches_intent(query):
                continue
            if "*" in plugin.rbac.allowed_roles:
                return plugin
            if any(r in plugin.rbac.allowed_roles for r in user_roles):
                return plugin
        return None

    def list_ids(self) -> list[str]:
        return list(self._plugins.keys())

    def __len__(self) -> int:
        return len(self._plugins)


# ── Global singleton ──────────────────────────────────────────────
registry = PluginRegistry()
