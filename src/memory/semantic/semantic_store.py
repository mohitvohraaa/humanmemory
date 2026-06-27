"""
src/memory/semantic/semantic_store.py
────────────────────────────────────────
Layer 3 — Semantic Memory storage.

Stores durable facts extracted from episodic clusters (consolidation.py).
Detects contradictions between new and existing facts; recency wins,
but old facts are flagged stale rather than deleted, preserving history.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.memory.models import SemanticFact, FactCategory

load_dotenv()


CONTRADICTION_PROMPT = """Compare these two statements about the same user.

Existing fact: "{old_fact}"
New fact: "{new_fact}"

Do they genuinely contradict each other (not just add detail, but
conflict)? Answer with exactly one word: YES or NO."""


class SemanticStore:
    """
    Stores durable facts per user, with contradiction detection.

    Usage:
        store = SemanticStore(user_id="user_001")
        store.add_fact("User prefers exercising in the morning.", "preference",
                        source_episode_ids=["ep1", "ep2"])
        facts = store.get_facts(category="preference")
    """

    def __init__(self, user_id: str = "default_user",
                 storage_dir: str = "data/processed/semantic"):
        self.user_id = user_id
        self.storage_path = Path(storage_dir) / f"{user_id}.json"
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self._facts: list[dict] = []  # stored as plain dicts on disk
        self._load()

    # ── Write ─────────────────────────────────────────────────────────────

    def add_fact(
        self,
        fact_text: str,
        category: str,
        source_episode_ids: list[str],
    ) -> SemanticFact:
        """
        Add a new fact. Checks active facts in the SAME category
        for contradictions before saving.
        """
        # Step 1 — check existing active facts in same category for conflict
        same_category_facts = [
            f for f in self._facts
            if f["category"] == category and not f["is_stale"]
        ]

        for existing in same_category_facts:
            if self._is_contradiction(existing["fact_text"], fact_text):
                # Recency wins — mark the OLD fact stale, keep the new one
                existing["is_stale"] = True
                existing["last_contradicted_at"] = datetime.utcnow().isoformat()

        # Step 2 — build the new fact record
        new_fact = {
            "id": str(uuid.uuid4()),
            "user_id": self.user_id,
            "category": category,
            "fact_text": fact_text,
            "source_episode_ids": source_episode_ids,
            "confidence": 1.0,
            "is_stale": False,
            "first_observed_at": datetime.utcnow().isoformat(),
            "last_confirmed_at": datetime.utcnow().isoformat(),
        }
        self._facts.append(new_fact)
        self._save()
        return self._to_semantic_fact(new_fact)

    def _is_contradiction(self, old_fact: str, new_fact: str) -> bool:
        """LLM call: does the new fact genuinely conflict with the old one?"""
        prompt = CONTRADICTION_PROMPT.format(old_fact=old_fact, new_fact=new_fact)
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")

    # ── Read ──────────────────────────────────────────────────────────────

    def get_facts(self, category: str | None = None,
                   include_stale: bool = False) -> list[SemanticFact]:
        """Get facts, optionally filtered by category, excluding stale by default."""
        results = self._facts
        if category:
            results = [f for f in results if f["category"] == category]
        if not include_stale:
            results = [f for f in results if not f["is_stale"]]
        return [self._to_semantic_fact(f) for f in results]

    def to_context(self, max_facts: int = 5) -> str:
        """Assemble active facts into a working-memory injection string."""
        facts = self.get_facts()
        top = facts[:max_facts]
        if not top:
            return ""
        lines = ["[USER PROFILE]"] + [f"- {f.fact_text}" for f in top]
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "total_facts": len(self._facts),
            "active_facts": len([f for f in self._facts if not f["is_stale"]]),
            "stale_facts": len([f for f in self._facts if f["is_stale"]]),
        }

    def clear(self) -> None:
        self._facts = []
        self._save()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _to_semantic_fact(self, raw: dict) -> SemanticFact:
        return SemanticFact(
            id=raw["id"], user_id=raw["user_id"],
            category=FactCategory(raw["category"]),
            fact_text=raw["fact_text"],
            source_episode_ids=raw["source_episode_ids"],
            confidence=raw["confidence"], is_stale=raw["is_stale"],
            first_observed_at=datetime.fromisoformat(raw["first_observed_at"]),
            last_confirmed_at=datetime.fromisoformat(raw["last_confirmed_at"]),
        )

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump({"user_id": self.user_id, "facts": self._facts}, f, indent=2)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            self._facts = data.get("facts", [])
        except Exception:
            self._facts = []
