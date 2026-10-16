# 🔬 ResearchMate – Autonomous AI Research Assistant Agent

ResearchMate is an agentic AI application that takes a research topic, plans its own
research, searches the live web, extracts notes into memory, writes a structured report
with citations, and **reviews its own draft**, going back for more research when it finds gaps.

## Architecture

```
          ┌──────────┐   ┌────────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐
 Topic ──▶│ Planner  │──▶│ Researcher │──▶│ Analyst │──▶│ Writer │──▶│ Reviewer │──▶ Report
          └──────────┘   └────────────┘   └─────────┘   └────────┘   └──────────┘
                               ▲   (web search +   (notes →           │
                               │    page reading)   ChromaDB)         │
                               └────────── gaps found (max 2 loops) ──┘
```

| Node | Role |
|---|---|
| Planner | Breaks the topic into focused sub-questions |
| Researcher | Calls the Tavily search tool and reads each web page (tool use) |
| Analyst | Extracts key facts per source and stores them in ChromaDB (memory) |
| Writer | Retrieves relevant notes per section (RAG) and writes a cited Markdown report |
| Reviewer | Critiques the draft (reflection); triggers extra searches if gaps exist |

## Tech stack
- **Language:** Python 3.10+
- **LLM:** Google Gemini (`gemini-2.5-flash`, configurable)
- **Agent framework:** LangGraph (LangChain)
- **Tools / APIs:** Tavily Search API, Requests + BeautifulSoup page reader
- **Storage:** ChromaDB (research notes), SQLite (report history)
- **Frontend:** Streamlit · **Export:** Markdown and PDF (fpdf2)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys (both have free tiers)
copy .env.example .env          # Windows  (cp on macOS/Linux)
#   GOOGLE_API_KEY  -> https://aistudio.google.com/app/apikey
#   TAVILY_API_KEY  -> https://app.tavily.com

# 4. Run
streamlit run app.py
```

The app opens at http://localhost:8501. You can also paste keys in the sidebar ⚙️ Settings
instead of using `.env`.

> On the first run, ChromaDB downloads a small embedding model (~80 MB). If that fails,
> the app automatically falls back to keyword-based note retrieval.

## Offline test
`python test_agent.py` runs the whole graph (including a review loop) with a fake LLM and
fake search tool, so you can verify the workflow without API keys.

## Project structure
```
app.py          Streamlit UI (live agent progress, report view, downloads, history)
agent.py        LangGraph agent: state, 5 nodes, conditional review loop
tools.py        web_search (Tavily) and fetch_page (web page reader)
memory.py       ChromaDB note memory + SQLite report history
export.py       Markdown → PDF
config.py       Settings from .env
test_agent.py   Offline end-to-end test
```

## Deploy (free)
1. Push this folder to a GitHub repository (`.env` is git-ignored).
2. Go to https://share.streamlit.io → **New app** → pick the repo, main file `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```
   GOOGLE_API_KEY = "..."
   TAVILY_API_KEY = "..."
   ```
