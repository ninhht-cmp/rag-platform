"""
app/services/agent/agent_service.py
─────────────────────────────────────
LangGraph-based agent for use cases that need tool execution.

Flow (ReAct pattern):
  User query
      ↓
  RAG retrieval (context)
      ↓
  LLM decides: answer OR use tool
      ↓
  If tool: execute → observe → back to LLM
      ↓
  Final answer with citations

Human-in-loop: certain tool calls require explicit user approval.
The agent pauses and returns an "approval_required" status.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.logging import get_logger
from app.core.plugin_registry import UseCasePlugin
from app.models.domain import QueryRequest, QueryResponse, QueryStatus, User
from app.services.agent.session_service import SessionService
from app.services.agent.tools import get_tools_for_plugin
from app.services.rag.llm_service import get_llm_service
from app.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)

MAX_AGENT_ITERATIONS = 5   # prevent infinite loops


class AgentState(TypedDict):
    messages: list[Any]
    iteration: int
    requires_approval: bool
    approval_tool: str


class AgentService:
    """
    Agentic query handler.
    Falls back to pure RAG if no tools are configured for the plugin.
    """

    def __init__(self, session_service: SessionService | None = None) -> None:
        self._llm = get_llm_service()
        self._rag = RAGPipeline()
        self._session = session_service

    async def run(
        self,
        request: QueryRequest,
        user: User,
        plugin: UseCasePlugin,
    ) -> QueryResponse:
        """
        If plugin has agent_tools → run agentic loop.
        Otherwise → pure RAG pipeline.
        """
        tools = get_tools_for_plugin(plugin.agent_tools)

        if not tools:
            # No tools → plain RAG
            return await self._rag.query(request, user)

        return await self._agentic_query(request, user, plugin, tools)

    async def _agentic_query(
        self,
        request: QueryRequest,
        user: User,
        plugin: UseCasePlugin,
        tools: list[Any],
    ) -> QueryResponse:
        """ReAct agent loop with tool execution."""
        import time
        start_ms = int(time.monotonic() * 1000)

        # Get conversation history for context
        history_text = ""
        if self._session and request.session_id:
            history_text = await self._session.format_for_prompt(request.session_id)

        # First: get RAG context
        rag_response = await self._rag.query(request, user)
        rag_context = rag_response.answer

        # Build agent system prompt
        system = self._build_agent_system_prompt(plugin, rag_context, history_text)

        # Bind tools to LLM
        llm_with_tools = self._llm.primary.bind_tools(tools)

        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=request.query),
        ]

        final_answer = ""
        iteration = 0
        tool_names_used: list[str] = []

        # ReAct loop
        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # Check if LLM wants to call tools
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # Final answer — no more tools needed
                final_answer = str(response.content)
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                # Human-in-loop check
                if tool_name in plugin.human_in_loop_triggers:
                    logger.info(
                        "agent.human_in_loop_required",
                        tool=tool_name,
                        use_case=plugin.id,
                    )
                    return QueryResponse(
                        query=request.query,
                        answer=(
                            f"This action ({tool_name.replace('_', ' ')}) requires "
                            f"human approval before proceeding. "
                            f"Please confirm you want to continue."
                        ),
                        use_case_id=plugin.id,
                        status=QueryStatus.PENDING,
                        confidence=0.9,
                        session_id=request.session_id,
                        latency_ms=int(time.monotonic() * 1000) - start_ms,
                    )

                # Execute the tool
                tool_obj = next(
                    (t for t in tools if t.name == tool_name), None  # type: ignore[union-attr]
                )
                if tool_obj is None:
                    tool_result = f"Error: tool '{tool_name}' not found"
                else:
                    try:
                        import asyncio as _asyncio
                        # 30s timeout per tool — prevent hanging requests
                        tool_result = await _asyncio.wait_for(
                            tool_obj.ainvoke(tool_args),
                            timeout=30.0,
                        )
                        tool_names_used.append(tool_name)
                        logger.info(
                            "agent.tool_executed",
                            tool=tool_name,
                            use_case=plugin.id,
                        )
                    except _asyncio.TimeoutError:
                        tool_result = f"Tool '{tool_name}' timed out after 30 seconds."
                        logger.error("agent.tool_timeout", tool=tool_name)
                    except Exception as exc:
                        tool_result = f"Tool error: {exc}"
                        logger.error("agent.tool_error", tool=tool_name, error=str(exc))

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                )

        if not final_answer:
            final_answer = "I completed the requested actions. Is there anything else I can help with?"

        # Persist session history
        if self._session and request.session_id:
            await self._session.append(request.session_id, "user", request.query)
            await self._session.append(request.session_id, "assistant", final_answer[:500])

        latency_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "agent.complete",
            use_case=plugin.id,
            iterations=iteration,
            tools_used=tool_names_used,
            latency_ms=latency_ms,
        )

        return QueryResponse(
            query=request.query,
            answer=final_answer,
            use_case_id=plugin.id,
            citations=rag_response.citations,
            confidence=rag_response.confidence,
            status=QueryStatus.COMPLETED,
            session_id=request.session_id,
            latency_ms=latency_ms,
        )

    def _build_agent_system_prompt(
        self,
        plugin: UseCasePlugin,
        rag_context: str,
        history: str,
    ) -> str:
        parts = [
            f"You are an AI agent for the {plugin.name} use case.",
            "You have access to tools AND a knowledge base context.",
            "",
            "INSTRUCTIONS:",
            "1. First try to answer from the knowledge base context below.",
            "2. Use tools only when the knowledge base doesn't have the answer or action is needed.",
            "3. Be concise and actionable.",
            "4. Always confirm before taking any irreversible action.",
            "",
            "KNOWLEDGE BASE CONTEXT:",
            rag_context,
        ]
        if history:
            parts.extend(["", history])
        return "\n".join(parts)
