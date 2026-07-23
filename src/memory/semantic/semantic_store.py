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

from src.memory.models import SemanticFact, FactCategory

load_dotenv()


FACT_CHECK_PROMPT = """You are managing a user profile for a companion AI.

Existing facts about the user (in this category):
{existing_facts}

New fact to evaluate: "{new_fact}"

Classify the new fact as exactly one of:
  DUPLICATE    - the new fact says the same thing as an existing fact
                 (same meaning, possibly different words)
  CONTRADICTION - the new fact directly conflicts with an existing fact
                 (not just adds detail, but genuinely opposes it)
  NEW          - the new fact is genuinely different and non-conflicting

If CONTRADICTION, also state which existing fact it contradicts by
writing: CONTRADICTION: <the exact existing fact text>

Answer on a single line. Examples:
  DUPLICATE
  NEW
  CONTRADICTION: User enjoys their current job and feels fulfilled.
"""


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
                 storage_dir: str = "data/processed/semantic",
                 groq_client=None):
        self.user_id = user_id
        self.storage_path = Path(storage_dir) / f"{user_id}.json"
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        if groq_client is None:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = groq_client

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
        # Step 1 — filter existing active facts in same category
        same_category_facts = [
            f for f in self._facts
            if f["category"] == category and not f["is_stale"]
        ]

        # ONE LLM call: handles duplicate + contradiction simultaneously
        verdict = self._check_against_existing(same_category_facts, fact_text)

        if verdict == "DUPLICATE":
            # Already stored — return the matching existing fact
            return self._to_semantic_fact(same_category_facts[0])

        if verdict.startswith("CONTRADICTION:"):
            # Find the specific fact being contradicted and mark it stale
            contradicted_text = verdict.replace("CONTRADICTION:", "").strip()
            for existing in same_category_facts:
                if existing["fact_text"].strip().lower() == contradicted_text.strip().lower():
                    existing["is_stale"] = True
                    existing["last_contradicted_at"] = datetime.utcnow().isoformat()
                    break
            # If exact match not found, mark the most recent one stale
            if not any(f.get("last_contradicted_at") for f in same_category_facts):
                same_category_facts[-1]["is_stale"] = True
                same_category_facts[-1]["last_contradicted_at"] = datetime.utcnow().isoformat()

        # NEW (or after contradiction) — store the fact
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

    def _check_against_existing(
        self,
        same_category_facts: list[dict],
        new_fact: str,
    ) -> str:
        """
        ONE LLM call that handles both duplicate detection AND
        contradiction detection simultaneously.

        Returns one of:
          "DUPLICATE"        → new fact is same meaning as existing one
          "NEW"              → genuinely new, store it
          "CONTRADICTION:<text>" → conflicts with a specific existing fact
        """
        if not same_category_facts:
            return "NEW"

        existing_text = "\n".join(
            f"- {f['fact_text']}" for f in same_category_facts
        )
        prompt = FACT_CHECK_PROMPT.format(
            existing_facts=existing_text,
            new_fact=new_fact,
        )
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

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
