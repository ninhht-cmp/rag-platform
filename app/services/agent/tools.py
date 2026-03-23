"""
app/services/agent/tools.py
────────────────────────────
Agent tools — callable by LangGraph agents.

Each tool is a plain async function decorated with @tool.
Tools are registered per plugin in plugins/__init__.py.

Design principles:
- Each tool has narrow, single responsibility
- All tools fail gracefully (return error string, never raise)
- Human-in-loop tools require explicit approval flag
- No tool makes irreversible changes without confirmation
"""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Customer Support Tools ────────────────────────────────────────

@tool
async def create_support_ticket(
    subject: str,
    description: str,
    priority: str = "normal",
    user_email: str = "",
) -> str:
    """
    Create a support ticket in the ticketing system (Zendesk/Freshdesk).
    Use when the customer issue cannot be resolved automatically.
    priority: 'low' | 'normal' | 'high' | 'urgent'
    """
    logger.info(
        "tool.create_ticket",
        subject=subject[:60],
        priority=priority,
    )
    # Stub — replace with real Zendesk/Freshdesk API call
    # from zenpy import Zenpy
    # ticket = zendesk_client.tickets.create(...)
    ticket_id = f"TKT-{hash(subject) % 100000:05d}"
    return (
        f"Ticket created successfully.\n"
        f"Ticket ID: {ticket_id}\n"
        f"Priority: {priority}\n"
        f"You will receive an email at {user_email or 'your registered email'} "
        f"when a support agent responds."
    )


@tool
async def lookup_order_status(order_id: str) -> str:
    """
    Look up the status of a customer order.
    Returns current status, estimated delivery, and tracking info.
    """
    logger.info("tool.lookup_order", order_id=order_id)
    # Stub — replace with real order management API
    return (
        f"Order {order_id}: Status = Processing | "
        f"Estimated delivery: 3-5 business days | "
        f"Tracking: will be provided via email once shipped."
    )


@tool
async def check_account_status(user_email: str) -> str:
    """
    Check the current status and subscription info of a user account.
    """
    logger.info("tool.check_account", email_prefix=user_email[:3] + "***")
    # Stub — replace with real user management API
    return (
        f"Account status: Active | "
        f"Plan: Pro | "
        f"Next billing: 2026-04-15 | "
        f"Usage this month: 68%"
    )


# ── Sales Automation Tools ────────────────────────────────────────

@tool
async def web_search_company(company_name: str) -> str:
    """
    Search for recent news and information about a prospect company.
    Use before sales meetings to get up-to-date context.
    Returns: company description, recent news, funding info.
    """
    logger.info("tool.web_search", company=company_name)
    # Stub — replace with Tavily/Serper/Brave Search API
    # async with httpx.AsyncClient() as client:
    #     resp = await client.get(SEARCH_API, params={"q": company_name})
    return (
        f"Research for {company_name}:\n"
        f"- Industry: Technology/SaaS\n"
        f"- Size: 200-500 employees\n"
        f"- Recent news: Raised Series B funding ($25M) in Q1 2026\n"
        f"- Tech stack: AWS, Python, PostgreSQL\n"
        f"- Key challenges: scaling customer onboarding, reducing churn\n"
        f"Note: Replace this stub with real Tavily/Serper API integration."
    )


@tool
async def crm_lookup_prospect(company_name: str) -> str:
    """
    Look up a company in the CRM to get existing interaction history,
    deal stage, and assigned rep information.
    """
    logger.info("tool.crm_lookup", company=company_name)
    # Stub — replace with Salesforce/HubSpot API
    return (
        f"CRM data for {company_name}:\n"
        f"- Status: Prospect (first contact)\n"
        f"- Last interaction: No previous contact\n"
        f"- Assigned rep: Unassigned\n"
        f"- Notes: Inbound lead from website\n"
        f"Note: Replace with real Salesforce/HubSpot API integration."
    )


@tool
async def draft_outreach_email(
    prospect_name: str,
    company_name: str,
    pain_point: str,
    product_value_prop: str,
) -> str:
    """
    Draft a personalized outreach email for a sales prospect.
    IMPORTANT: Always requires human review before sending.
    Returns a draft — never auto-sends.
    """
    logger.info("tool.draft_email", company=company_name)
    return (
        f"DRAFT EMAIL — REQUIRES HUMAN REVIEW BEFORE SENDING\n"
        f"{'='*50}\n"
        f"To: {prospect_name} @ {company_name}\n"
        f"Subject: Quick question about {pain_point} at {company_name}\n\n"
        f"Hi {prospect_name},\n\n"
        f"I noticed {company_name} is scaling rapidly — congrats on the Series B!\n\n"
        f"Many companies at your stage struggle with {pain_point}. "
        f"We help with {product_value_prop}.\n\n"
        f"Worth a 15-minute call this week?\n\n"
        f"Best,\n[YOUR NAME]\n\n"
        f"{'='*50}\n"
        f"⚠️  Review and personalize before sending."
    )


@tool
async def create_crm_activity(
    company_name: str,
    activity_type: str,
    notes: str,
) -> str:
    """
    Log an activity in the CRM (call, email, meeting notes).
    activity_type: 'call' | 'email' | 'meeting' | 'note'
    """
    logger.info("tool.crm_activity", company=company_name, type=activity_type)
    # Stub — replace with Salesforce/HubSpot API
    return (
        f"CRM activity logged:\n"
        f"Company: {company_name}\n"
        f"Type: {activity_type}\n"
        f"Notes: {notes[:200]}\n"
        f"Activity ID: ACT-{hash(notes) % 10000:04d}"
    )


# ── Tool registry ─────────────────────────────────────────────────
# Map tool_id (used in plugin config) → callable tool

TOOL_REGISTRY: dict[str, object] = {
    "create_ticket": create_support_ticket,
    "lookup_order": lookup_order_status,
    "check_account_status": check_account_status,
    "web_search": web_search_company,
    "crm_lookup": crm_lookup_prospect,
    "draft_email": draft_outreach_email,
    "create_crm_activity": create_crm_activity,
}


def get_tools_for_plugin(tool_ids: list[str]) -> list[object]:
    """Return LangChain tool objects for a list of tool IDs."""
    tools = []
    for tid in tool_ids:
        t = TOOL_REGISTRY.get(tid)
        if t:
            tools.append(t)
        else:
            logger.warning("tool.not_found", tool_id=tid)
    return tools
