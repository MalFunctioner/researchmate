"""Tools the agent can use: web search and web page reading."""
import requests
from bs4 import BeautifulSoup

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (ResearchMate student project)"}


def web_search(query: str, max_results: int = config.RESULTS_PER_QUERY) -> list[dict]:
    """Search the web with Tavily. Returns [{title, url, content}]."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in resp.get("results", [])
        if r.get("url")
    ]


def fetch_page(url: str, max_chars: int = config.MAX_PAGE_CHARS) -> str:
    """Download a web page and return its readable text (empty string on failure)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        if "html" not in r.headers.get("Content-Type", "html"):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception:
        return ""
