"""Central configuration loaded from environment variables / .env file."""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "4"))
RESULTS_PER_QUERY = int(os.getenv("RESULTS_PER_QUERY", "3"))
MAX_REVIEW_LOOPS = int(os.getenv("MAX_REVIEW_LOOPS", "2"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "6000"))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
