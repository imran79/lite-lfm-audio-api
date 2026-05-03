"""
Web search module — Tavily (primary) + DuckDuckGo (fallback).

Provides a unified ``search_web(query)`` function that tries Tavily first
and falls back to DuckDuckGo if Tavily is unconfigured or fails.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.config import SEARCH_PROVIDER, TAVILY_API_KEY, TAVILY_MAX_RESULTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tavily backend
# ---------------------------------------------------------------------------

def _search_tavily(query: str, max_results: int = 5) -> str:
    """Synchronous Tavily search (runs in executor thread)."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",  # gets more thorough results
            include_answer=True,      # Tavily's own summarised answer
        )
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        raise

    results = response.get("results", [])
    if not results:
        return "No results found."

    lines: list[str] = []

    # Prefer Tavily's summarised answer if available
    answer = response.get("answer")
    if answer:
        lines.append(f"Summary: {answer}")
        lines.append("")

    for i, r in enumerate(results, 1):
        title = r.get("title", "No Title")
        content = r.get("content", "")
        url = r.get("url", "")
        score = r.get("score", "")
        lines.append(f"{i}. {title}")
        if content:
            lines.append(f"   {content}")
        if url:
            lines.append(f"   Source: {url}")
        if score:
            lines.append(f"   Relevance: {score:.2f}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# DuckDuckGo backend
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, max_results: int = 3) -> str:
    """Synchronous DuckDuckGo search (runs in executor thread)."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        raise

    if not results:
        return "No results found."

    formatted_results: list[str] = []
    for res in results:
        title = res.get("title", "No Title")
        body = res.get("body", "")
        formatted_results.append(f"{title}\n{body}")
    return "Search Results:\n" + "\n\n".join(formatted_results)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

async def search_web(query: str) -> str:
    """
    Search the web for the given query.

    Strategy (in order):
    1. If ``SEARCH_PROVIDER == "tavily"`` and ``TAVILY_API_KEY`` is set → Tavily
    2. Else → DuckDuckGo (no API key needed, but may be rate-limited)
    3. If primary fails → fallback to the other provider

    Returns:
        A human-readable string with search results, or an error message.
    """
    logger.info("Web search for: %s (provider=%s)", query, SEARCH_PROVIDER)

    # -- Decide which order to try --
    if SEARCH_PROVIDER == "tavily" and TAVILY_API_KEY:
        primary: tuple[str, int] = ("tavily", TAVILY_MAX_RESULTS)
        fallback: tuple[str, int] = ("duckduckgo", 3)
    else:
        primary = ("duckduckgo", 3)
        fallback = ("tavily", TAVILY_MAX_RESULTS) if TAVILY_API_KEY else None

    attempts = [
        ("tavily", _search_tavily, TAVILY_MAX_RESULTS),
        ("duckduckgo", _search_duckduckgo, 3),
    ]

    # Re-order so the preferred provider is first
    if primary[0] == "duckduckgo":
        attempts.reverse()

    last_error: str | None = None

    for name, func, max_results in attempts:
        if name == "tavily" and not TAVILY_API_KEY:
            continue  # skip Tavily when no API key
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, func, query, max_results
            )
            logger.info("%s search succeeded for: %.60s", name, query)
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning("%s search failed, trying fallback: %s", name, last_error)
            continue

    return f"Search failed: {last_error or 'All search providers returned no results.'}"
