"""
src/memory/episodic/store.py
─────────────────────────────
Episodic memory store — SQLite backend.

Two retrieval modes:
  1. naive_retrieve   — last N messages by timestamp (our Baseline-1)
  2. (coming later)   — semantic retrieval via ChromaDB embeddings

Why SQLite:
  - Zero setup, single file, runs on any machine
  - Perfect for structured queries: "get all turns from session X"
  - ChromaDB handles the vector search side; SQLite handles metadata
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from src.memory.models import EpisodicMemory


# ─── Schema ──────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memories (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    turn_id          TEXT,
    speaker          TEXT,
    text             TEXT NOT NULL,
    summary          TEXT,
    created_at       TEXT NOT NULL,
    importance_score REAL DEFAULT 0.0,
    recency_score    REAL DEFAULT 1.0,
    valence_score    REAL DEFAULT 0.0,
    emotion_labels   TEXT DEFAULT '[]',
    topic_tags       TEXT DEFAULT '[]'
);
"""

# Index on created_at so ORDER BY created_at DESC is fast
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_created_at
ON episodic_memories (created_at DESC);
"""


# ─── Store ───────────────────────────────────────────────────────────────────

class EpisodicStore:
    """
    SQLite-backed episodic memory store.

    Usage:
        store = EpisodicStore("data/processed/episodic.db")
        store.add(memory)
        memories = store.naive_retrieve(limit=10)
    """

    def __init__(self, db_path: str = "data/processed/episodic.db"):
        # Create parent directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the table and index if they don't exist yet."""
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEX_SQL)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with row_factory so rows behave like dicts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────────

    def add(self, memory: EpisodicMemory) -> None:
        """
        Insert one episodic memory into the database.
        Skips silently if a memory with the same ID already exists.
        """
        import json
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO episodic_memories
                    (id, session_id, turn_id, speaker, text, summary,
                     created_at, importance_score, recency_score,
                     valence_score, emotion_labels, topic_tags)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.session_id,
                    memory.turn_id,
                    memory.speaker,
                    memory.text,
                    memory.summary,
                    memory.created_at.isoformat()
                    if isinstance(memory.created_at, datetime)
                    else memory.created_at,
                    memory.importance_score,
                    memory.recency_score,
                    memory.valence_score,
                    json.dumps(memory.emotion_labels),
                    json.dumps(memory.topic_tags),
                ),
            )

    def add_batch(self, memories: list[EpisodicMemory]) -> None:
        """Insert multiple memories efficiently in one transaction."""
        for memory in memories:
            self.add(memory)

    # ── Read ──────────────────────────────────────────────────────────────

    def naive_retrieve(self, limit: int = 10) -> list[EpisodicMemory]:
        """
        BASELINE-1: Return the most recent N memories.
        No ranking. No semantic search. Just recency.

        This is the zero-line everything else must beat.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM episodic_memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_by_id(self, memory_id: str) -> EpisodicMemory | None:
        """Fetch one memory by its ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodic_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def count(self) -> int:
        """How many memories are stored?"""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM episodic_memories"
            ).fetchone()[0]

    def clear(self) -> None:
        """Delete all memories. Useful for test teardown."""
        with self._connect() as conn:
            conn.execute("DELETE FROM episodic_memories")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _row_to_memory(self, row: sqlite3.Row) -> EpisodicMemory:
        """Convert a database row back into an EpisodicMemory object."""
        import json
        return EpisodicMemory(
            id=row["id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"] or "",
            speaker=row["speaker"] or "",
            text=row["text"],
            summary=row["summary"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            importance_score=row["importance_score"],
            recency_score=row["recency_score"],
            valence_score=row["valence_score"],
            emotion_labels=json.loads(row["emotion_labels"]),
            topic_tags=json.loads(row["topic_tags"]),
        )
