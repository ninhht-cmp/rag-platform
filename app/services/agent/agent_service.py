"""
app/services/agent/agent_service.py
─────────────────────────────────────
LangGraph-based agentic flow.
Used by plugins that have agent_tools configured.
Supports multi-turn tool use with human-in-the-loop confirmation for
irreversible actions.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from typing_extensions import TypedDict

from app.core.config import settings
from app.core.logging import get_logger
from app.core.plugin_registry import UseCasePlugin
from app.models.domain import QueryRequest, QueryResponse, QueryStatus, User
from app.services.agent.session_service import SessionService
from app.services.agent.tools import get_tools_for_plugin

logger = get_logger(__name__)


# ── Agent State ───────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    use_case_id: str
    user_id: str
    session_id: str | None
    tool_calls_count: int
    final_answer: str | None


# ── Agent Service ─────────────────────────────────────────────────


class AgentService:
    """
    LangGraph agent for tool-enabled plugins.
    Handles multi-step tool use with safety guardrails.
    """

    MAX_TOOL_CALLS = 10  # prevent infinite loops

    def __init__(self) -> None:
        self._model: ChatAnthropic | None = None

    def _get_model(self, tools: list[Any]) -> Any:
        if self._model is None:
            self._model = ChatAnthropic(
                api_key=SecretStr(settings.ANTHROPIC_API_KEY),
                model=settings.LLM_MODEL_PRIMARY,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
            )
        return self._model.bind_tools(tools)

    def _build_graph(self, tools: list[Any]) -> Any:
        """Build LangGraph state machine for agent execution."""
        model_with_tools = self._get_model(tools)
        tool_map = {t.name: t for t in tools}

        def call_model(state: AgentState) -> dict[str, Any]:
            response = asyncio.get_event_loop().run_until_complete(
                model_with_tools.ainvoke(state["messages"])
            )
            return {"messages": [response]}

        async def call_model_async(state: AgentState) -> dict[str, Any]:
            response = await model_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        async def call_tools(state: AgentState) -> dict[str, Any]:
            last_message = state["messages"][-1]
            tool_calls_count = state.get("tool_calls_count", 0)

            if tool_calls_count >= self.MAX_TOOL_CALLS:
                logger.warning(
                    "agent.tool_call_limit_reached",
                    use_case=state["use_case_id"],
                    count=tool_calls_count,
                )
                return {
                    "messages": [
                        ToolMessage(
                            content=(
                                "Tool call limit reached. "
                                "Providing best answer with available info."
                            ),
                            tool_call_id="limit_reached",
                        )
                    ],
                    "tool_calls_count": tool_calls_count,
                }

            results = []
            for tool_call in last_message.tool_calls:
                tool = tool_map.get(tool_call["name"])
                if tool is None:
                    result = f"Tool '{tool_call['name']}' not found."
                else:
                    try:
                        result = await tool.ainvoke(tool_call["args"])
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                        logger.error(
                            "agent.tool_error",
                            tool=tool_call["name"],
                            error=str(exc),
                        )
                results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            return {
                "messages": results,
                "tool_calls_count": tool_calls_count + len(last_message.tool_calls),
            }

        def should_continue(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                if state.get("tool_calls_count", 0) >= self.MAX_TOOL_CALLS:
                    return END
                return "tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("agent", call_model_async)
        graph.add_node("tools", call_tools)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        return graph.compile()

    async def run(
        self,
        request: QueryRequest,
        user: User,
        plugin: UseCasePlugin,
    ) -> QueryResponse:
        """Execute agent for a plugin with tools configured."""
        tools = get_tools_for_plugin(plugin.agent_tools)
        if not tools:
            logger.warning("agent.no_tools", plugin=plugin.id)
            from app.services.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline()
            return await pipeline.query(request, user)

        # Load conversation history if session exists
        session_context = ""
        if request.session_id:
            try:
                from app.main import get_redis

                session_svc = SessionService(get_redis())
                session_context = await session_svc.format_for_prompt(request.session_id)
            except Exception as exc:
                logger.warning("agent.session_load_failed", error=str(exc))

        system_content = (
            f"You are a helpful AI assistant for {plugin.name}.\n"
            f"{plugin.description}\n\n"
            f"You have access to tools to help answer questions and take actions.\n"
            f"Always be helpful, accurate, and transparent about what you're doing.\n"
            f"For irreversible actions, always confirm with the user first.\n"
        )
        if session_context:
            system_content += f"\n{session_context}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=request.query),
        ]

        graph = self._build_graph(tools)

        try:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "messages": messages,
                        "use_case_id": plugin.id,
                        "user_id": str(user.id),
                        "session_id": request.session_id,
                        "tool_calls_count": 0,
                        "final_answer": None,
                    }
                ),
                timeout=120.0,
            )
        except TimeoutError:
            logger.error("agent.timeout", plugin=plugin.id, user=str(user.id))
            return QueryResponse(
                query=request.query,
                answer="The request timed out. Please try a simpler query.",
                use_case_id=plugin.id,
                status=QueryStatus.FAILED,
                confidence=0.0,
            )
        except Exception as exc:
            logger.error("agent.error", plugin=plugin.id, error=str(exc))
            return QueryResponse(
                query=request.query,
                answer=f"An error occurred while processing your request: {exc}",
                use_case_id=plugin.id,
                status=QueryStatus.FAILED,
                confidence=0.0,
            )

        # Extract final answer from last AI message
        final_message = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    continue
                final_message = msg
                break

        answer = (
            str(final_message.content) if final_message else "I was unable to generate a response."
        )

        # Persist to session if provided
        if request.session_id:
            try:
                from app.main import get_redis

                session_svc = SessionService(get_redis())
                await session_svc.append(request.session_id, "user", request.query)
                await session_svc.append(request.session_id, "assistant", answer)
            except Exception as exc:
                logger.warning("agent.session_persist_failed", error=str(exc))

        tool_calls_count = result.get("tool_calls_count", 0)
        logger.info(
            "agent.complete",
            plugin=plugin.id,
            tool_calls=tool_calls_count,
            user=str(user.id),
        )

        return QueryResponse(
            id=str(uuid.uuid4()),
            query=request.query,
            answer=answer,
            use_case_id=plugin.id,
            citations=[],
            confidence=0.85 if tool_calls_count > 0 else 0.7,
            status=QueryStatus.COMPLETED,
            session_id=request.session_id,
            token_usage={"tool_calls": tool_calls_count},
        )
