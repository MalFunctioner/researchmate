"""Agent memory: ChromaDB for research notes (RAG) and SQLite for report history."""
import os
import sqlite3
import uuid
from datetime import datetime

import config


class NoteMemory:
    """Stores extracted notes in a vector DB so the writer can retrieve the most
    relevant notes for each section. Falls back to keyword matching if ChromaDB's
    embedding model can't be loaded (e.g. offline)."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._fallback: list[dict] = []
        self.collection = None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=os.path.join(config.DATA_DIR, "chroma"))
            self.collection = client.get_or_create_collection(f"notes_{self.run_id}")
        except Exception:
            self.collection = None

    def add(self, text: str, metadata: dict):
        self._fallback.append({"text": text, "metadata": metadata})
        if self.collection is not None:
            try:
                self.collection.add(
                    ids=[uuid.uuid4().hex], documents=[text], metadatas=[metadata]
                )
            except Exception:
                self.collection = None  # embedding failed -> keyword fallback

    def search(self, query: str, k: int = 6) -> list[dict]:
        if self.collection is not None:
            try:
                n = min(k, self.collection.count())
                if n:
                    res = self.collection.query(query_texts=[query], n_results=n)
                    return [
                        {"text": d, "metadata": m}
                        for d, m in zip(res["documents"][0], res["metadatas"][0])
                    ]
            except Exception:
                pass
        words = set(query.lower().split())
        scored = sorted(
            self._fallback,
            key=lambda n: -len(words & set(n["text"].lower().split())),
        )
        return scored[:k]


# ---------------- Report history (SQLite) ----------------
DB_PATH = os.path.join(config.DATA_DIR, "history.db")


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT, report TEXT, n_sources INTEGER, created_at TEXT)"""
    )
    return con


def save_report(topic: str, report: str, n_sources: int) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO reports (topic, report, n_sources, created_at) VALUES (?,?,?,?)",
            (topic, report, n_sources, datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        return cur.lastrowid


def list_reports() -> list[tuple]:
    with _conn() as con:
        return con.execute(
            "SELECT id, topic, n_sources, created_at FROM reports ORDER BY id DESC"
        ).fetchall()


def get_report(report_id: int) -> tuple | None:
    with _conn() as con:
        return con.execute(
            "SELECT topic, report, created_at FROM reports WHERE id=?", (report_id,)
        ).fetchone()


def delete_report(report_id: int):
    with _conn() as con:
        con.execute("DELETE FROM reports WHERE id=?", (report_id,))
