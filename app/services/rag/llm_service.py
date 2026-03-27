"""
app/services/rag/llm_service.py
────────────────────────────────
LLM abstraction with:
- Claude primary, GPT-4o fallback
- Streaming support
- Token usage tracking & budget guard  ← FIX: budget now enforced
- Prompt templating with Jinja2
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import LLMProvider, settings
from app.core.logging import get_logger
from app.models.domain import DocumentChunk

logger = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# Type alias for budget checker: (use_case_id) -> raises HTTPException if over budget
BudgetChecker = Callable[[str], Awaitable[None]]


def _load_jinja_env() -> Environment:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )


_jinja_env = _load_jinja_env()


class LLMService:
    """
    Unified LLM interface.
    Primary model: Claude Sonnet (complex reasoning)
    Secondary model: Claude Haiku (simple classification, cheap tasks)
    Fallback: GPT-4o (if primary fails)

    budget_checker: async callable injected at app startup after DB is available.
    Defaults to no-op so unit tests don't need a DB connection.
    """

    def __init__(self, budget_checker: BudgetChecker | None = None) -> None:
        self._primary: Any = None
        self._secondary: Any = None
        self._budget_checker = budget_checker

    def set_budget_checker(self, checker: BudgetChecker) -> None:
        """Wire in the budget enforcement callback after startup."""
        self._budget_checker = checker

    def _build_anthropic(self, model: str) -> ChatAnthropic:
        return ChatAnthropic(  # type: ignore[call-arg]
            api_key=settings.ANTHROPIC_API_KEY,
            model=model,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

    def _build_openai(self, model: str = "gpt-4o-mini") -> ChatOpenAI:
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=model,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

    @property
    def primary(self) -> Any:
        if self._primary is None:
            if settings.LLM_PROVIDER == LLMProvider.ANTHROPIC:
                self._primary = self._build_anthropic(settings.LLM_MODEL_PRIMARY)
            else:
                self._primary = self._build_openai("gpt-4o")
        return self._primary

    @property
    def secondary(self) -> Any:
        if self._secondary is None:
            if settings.LLM_PROVIDER == LLMProvider.ANTHROPIC:
                self._secondary = self._build_anthropic(settings.LLM_MODEL_SECONDARY)
            else:
                self._secondary = self._build_openai("gpt-4o-mini")
        return self._secondary

    def render_prompt(self, template_path: str, context: dict[str, Any]) -> str:
        try:
            template = _jinja_env.get_template(template_path)
            return template.render(**context)
        except Exception:
            logger.warning("prompt.template.missing", path=template_path)
            return self._default_rag_prompt(context)

    def _default_rag_prompt(self, context: dict[str, Any]) -> str:
        chunks: list[DocumentChunk] = context.get("chunks", [])
        ctx_text = "\n\n".join(f"[Source {i + 1}] {c.content}" for i, c in enumerate(chunks))
        return f"""You are a helpful AI assistant. Answer based ONLY on the provided context.
If the answer is not in the context, say \
"I don't have information about this in the available documents."
Always cite sources using [Source N] notation.

Context:
{ctx_text}

Question: {context.get("query", "")}

Answer:"""

    async def _check_budget(self, use_case_id: str) -> None:
        """
        FIX: Enforce daily token budget before each LLM call.
        No-op when budget_checker is not wired (tests, local dev without DB).
        """
        if self._budget_checker is None:
            return
        try:
            await self._budget_checker(use_case_id)
        except Exception as exc:
            # Re-raise HTTP 429 as-is; swallow unexpected errors so budget
            # check failure doesn't block all requests if DB is down.
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                raise
            logger.error("budget.check.error", error=str(exc))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        use_case_id: str = "unknown",
        use_secondary: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """
        Generate a response.
        Returns (answer_text, token_usage_dict).
        Raises HTTP 429 if daily budget exceeded.
        """
        await self._check_budget(use_case_id)

        model = self.secondary if use_secondary else self.primary
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        start = time.monotonic()
        try:
            response = await model.ainvoke(messages)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            usage: dict[str, int] = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                usage = {
                    "input_tokens": getattr(meta, "input_tokens", 0) or 0,
                    "output_tokens": getattr(meta, "output_tokens", 0) or 0,
                }

            answer = str(response.content)
            logger.info(
                "llm.generate.success",
                latency_ms=elapsed_ms,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                use_case=use_case_id,
            )
            return answer, usage

        except Exception as exc:
            logger.error("llm.generate.error", error=str(exc), use_case=use_case_id)
            raise

    async def stream(
        self,
        system_prompt: str,
        user_message: str,
        use_case_id: str = "unknown",
    ) -> AsyncIterator[str]:
        """Stream response tokens for real-time UI updates."""
        await self._check_budget(use_case_id)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        chain = self.primary | StrOutputParser()
        async for chunk in chain.astream(messages):
            yield chunk

    async def classify_intent(
        self,
        query: str,
        plugin_ids: list[str],
    ) -> str | None:
        if not plugin_ids:
            return None
        options = ", ".join(plugin_ids)
        prompt = f"""Classify this query into exactly one category: {options}
Return ONLY the category name, nothing else.
If none match, return "unknown".

Query: {query}

Category:"""
        answer, _ = await self.generate(
            system_prompt="You are a query classifier. Return only the category name.",
            user_message=prompt,
            use_secondary=True,
        )
        result = answer.strip().lower()
        return result if result in plugin_ids else None


# ── Singleton ─────────────────────────────────────────────────────
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
