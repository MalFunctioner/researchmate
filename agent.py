"""ResearchMate agent: a LangGraph workflow of five nodes.

    Planner -> Researcher -> Analyst -> Writer -> Reviewer
                   ^                                  |
                   +------- (gaps found, loop) -------+
"""
import json
import operator
import re
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, StateGraph

import config
import tools
from memory import NoteMemory


# ---------------------------------------------------------------- state
class ResearchState(TypedDict, total=False):
    topic: str
    sub_questions: list[str]
    pending_queries: list[str]       # queries the researcher should run next
    sources: list[dict]              # every source found: {id, title, url, content, question}
    new_source_ids: list[int]        # sources added in the latest research round
    draft: str
    feedback: str
    iterations: int
    approved: bool
    log: Annotated[list[str], operator.add]  # progress messages for the UI


# ---------------------------------------------------------------- helpers
def default_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0.3
    )


def _text(resp) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):  # some models return content blocks
        content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()


def _json(text: str, default):
    """Pull the first JSON object/array out of an LLM reply."""
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return default


# ---------------------------------------------------------------- agent
class ResearchAgent:
    def __init__(
        self,
        llm=None,
        search_fn: Callable = tools.web_search,
        fetch_fn: Callable = tools.fetch_page,
        max_sub_questions: int = config.MAX_SUB_QUESTIONS,
        max_loops: int = config.MAX_REVIEW_LOOPS,
    ):
        self.llm = llm or default_llm()
        self.search = search_fn
        self.fetch = fetch_fn
        self.max_sub_questions = max_sub_questions
        self.max_loops = max_loops
        self.memory = NoteMemory()
        self.graph = self._build_graph()

    def ask(self, prompt: str) -> str:
        return _text(self.llm.invoke(prompt))

    # ---- node 1: planner
    def planner(self, state: ResearchState) -> dict:
        prompt = f"""You are a research planner. Break the research topic below into
{self.max_sub_questions} focused sub-questions that together give a complete picture
(background, current state, key data, challenges, future outlook).
Each sub-question must work well as a web search query.

Topic: {state['topic']}

Reply ONLY with a JSON array of strings."""
        qs = _json(self.ask(prompt), [])
        qs = [q for q in qs if isinstance(q, str) and q.strip()][: self.max_sub_questions]
        if not qs:
            qs = [state["topic"]]
        return {
            "sub_questions": qs,
            "pending_queries": qs,
            "sources": [],
            "iterations": 0,
            "log": ["🧭 **Planner** created a research plan:\n" + "\n".join(f"- {q}" for q in qs)],
        }

    # ---- node 2: researcher (tool use)
    def researcher(self, state: ResearchState) -> dict:
        sources = list(state.get("sources", []))
        seen = {s["url"] for s in sources}
        new_ids, lines = [], []
        for q in state.get("pending_queries", []):
            try:
                results = self.search(q)
            except Exception as e:
                lines.append(f"- ⚠️ search failed for “{q}”: {e}")
                continue
            added = 0
            for r in results:
                if r["url"] in seen:
                    continue
                seen.add(r["url"])
                page = self.fetch(r["url"])
                content = page if len(page) > len(r.get("content", "")) else r.get("content", "")
                if not content:
                    continue
                sid = len(sources) + 1
                sources.append({"id": sid, "title": r.get("title") or r["url"],
                                "url": r["url"], "content": content, "question": q})
                new_ids.append(sid)
                added += 1
            lines.append(f"- 🔎 “{q}” → {added} new source(s)")
        return {
            "sources": sources,
            "new_source_ids": new_ids,
            "pending_queries": [],
            "log": ["🌐 **Researcher** searched the web and read pages:\n" + "\n".join(lines)],
        }

    # ---- node 3: analyst (extract notes -> memory)
    def analyst(self, state: ResearchState) -> dict:
        """Reads ALL new sources in a single LLM call (keeps API usage low)."""
        by_id = {s["id"]: s for s in state["sources"]}
        ids = [i for i in state.get("new_source_ids", []) if i in by_id]
        if not ids:
            return {"log": ["🧪 **Analyst** had no new sources to analyse."]}
        per_source = max(1500, 40000 // len(ids))  # keep the whole prompt a sensible size
        docs = "\n\n".join(
            f"=== SOURCE {i} ===\nQuestion it was found for: {by_id[i]['question']}\n"
            f"Title: {by_id[i]['title']}\nText: {by_id[i]['content'][:per_source]}"
            for i in ids
        )
        prompt = f"""You are a research analyst. For EACH source below, extract the key facts,
statistics, dates and viewpoints that help answer its question, as 3-6 concise bullet
points. Use only information in that source. If a source is irrelevant, use "IRRELEVANT".

Reply ONLY with a JSON object mapping source number to its notes, e.g.
{{"1": "- fact one\\n- fact two", "2": "IRRELEVANT"}}

{docs}"""
        result = _json(self.ask(prompt), {})
        n = 0
        for key, notes in (result.items() if isinstance(result, dict) else []):
            try:
                sid = int(str(key).strip().split()[-1])
            except ValueError:
                continue
            if sid not in by_id or not isinstance(notes, str) or "IRRELEVANT" in notes[:20].upper():
                continue
            s = by_id[sid]
            self.memory.add(notes, {"source_id": sid, "title": s["title"], "url": s["url"]})
            n += 1
        if n == 0:  # model reply unusable -> fall back to raw text so the writer still has material
            for sid in ids:
                s = by_id[sid]
                self.memory.add(s["content"][:1500], {"source_id": sid, "title": s["title"], "url": s["url"]})
            n = len(ids)
        return {"log": [f"🧪 **Analyst** extracted notes from {n} source(s) and stored them in memory."]}

    # ---- node 4: writer (RAG over notes)
    def writer(self, state: ResearchState) -> dict:
        blocks, used = [], set()
        for q in state["sub_questions"] + ([state["feedback"]] if state.get("feedback") else []):
            for note in self.memory.search(q, k=5):
                sid = note["metadata"]["source_id"]
                key = (sid, note["text"][:60])
                if key in used:
                    continue
                used.add(key)
                blocks.append(f"[Source {sid}] {note['metadata']['title']}\n{note['text']}")
        notes_text = "\n\n".join(blocks) or "No notes were collected."
        prompt = f"""You are a research writer. Write a well-structured research report in
Markdown on the topic below, using ONLY the research notes provided.

Topic: {state['topic']}

Required structure:
# <Report title>
## Introduction
## Key Findings   (use ### sub-headings for each theme)
## Analysis
## Conclusion

Rules:
- Cite sources inline with their number in square brackets, e.g. [2] or [1][3].
- Every factual claim or statistic must have a citation.
- Do NOT write a references section; it is added automatically.
- Be clear and specific, 600-900 words.
{"- Address this reviewer feedback: " + state['feedback'] if state.get('feedback') else ""}

Research notes:
{notes_text}"""
        draft = self.ask(prompt)
        return {"draft": draft, "log": [f"✍️ **Writer** drafted the report from {len(blocks)} notes."]}

    # ---- node 5: reviewer (reflection)
    def reviewer(self, state: ResearchState) -> dict:
        iterations = state.get("iterations", 0) + 1
        prompt = f"""You are a strict research reviewer. Check the report below for:
missing sections, claims without citations, important gaps in coverage of the topic.

Topic: {state['topic']}
Report:
{state['draft']}

Reply ONLY with JSON:
{{"approved": true/false, "feedback": "<short feedback>", "missing_queries": ["<web search query for each gap, max 2>"]}}"""
        verdict = _json(self.ask(prompt), {"approved": True, "feedback": "", "missing_queries": []})
        approved = bool(verdict.get("approved", True))
        queries = [q for q in verdict.get("missing_queries", []) if isinstance(q, str)][:2]
        if iterations >= self.max_loops or not queries:
            approved = True
        msg = (f"✅ **Reviewer** approved the report (round {iterations})."
               if approved else
               f"🔁 **Reviewer** found gaps (round {iterations}): {verdict.get('feedback', '')}\n"
               + "\n".join(f"- extra search: {q}" for q in queries))
        return {
            "approved": approved,
            "iterations": iterations,
            "feedback": verdict.get("feedback", "") if not approved else state.get("feedback", ""),
            "pending_queries": [] if approved else queries,
            "log": [msg],
        }

    @staticmethod
    def route_after_review(state: ResearchState) -> str:
        return "done" if state.get("approved") else "research_more"

    # ---- graph
    def _build_graph(self):
        g = StateGraph(ResearchState)
        g.add_node("planner", self.planner)
        g.add_node("researcher", self.researcher)
        g.add_node("analyst", self.analyst)
        g.add_node("writer", self.writer)
        g.add_node("reviewer", self.reviewer)
        g.set_entry_point("planner")
        g.add_edge("planner", "researcher")
        g.add_edge("researcher", "analyst")
        g.add_edge("analyst", "writer")
        g.add_edge("writer", "reviewer")
        g.add_conditional_edges(
            "reviewer", self.route_after_review, {"research_more": "researcher", "done": END}
        )
        return g.compile()

    # ---- public API
    def stream(self, topic: str):
        """Yields (node_name, update_dict) as the agent works."""
        for event in self.graph.stream({"topic": topic, "log": []}, stream_mode="updates"):
            for node, update in event.items():
                yield node, update


def build_final_report(draft: str, sources: list[dict]) -> str:
    """Append a references list containing only the sources actually cited."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", draft)})
    by_id = {s["id"]: s for s in sources}
    refs = [f"{i}. [{by_id[i]['title']}]({by_id[i]['url']})" for i in cited if i in by_id]
    if not refs:
        refs = [f"{s['id']}. [{s['title']}]({s['url']})" for s in sources]
    return draft.strip() + "\n\n## References\n" + "\n".join(refs) + "\n"
