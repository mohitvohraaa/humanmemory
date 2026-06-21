"""
src/memory/affective/affective_store.py
─────────────────────────────────────────
Stores and updates topic→emotion vectors for one user.

This is Layer 4 — the novel contribution of HumanMemory.
Absent from: MemGPT, Generative Agents, A-MEM, Mem0, Claude.

How it works:
  1. Every session end → EmotionClassifier + TopicTagger run on all turns
  2. For each turn: update(topic, emotion_scores) via EMA
  3. At inference: get(topic) → inject emotional context into prompt

EMA formula:
  new_vector = (1 - alpha) * old_vector + alpha * new_signal
  alpha = 0.1 → effective memory window ~10 turns
  Low alpha = slow adaptation = stable emotional profile

Storage:
  JSON file per user (simple, human-readable, no Redis needed for now)
  Redis upgrade path documented in future work
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.memory.models import AffectiveRecord, EmotionVector


# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_ALPHA = 0.1          # EMA learning rate
MIN_INTENSITY = 0.2          # skip update if emotion is too weak
MIN_MENTIONS  = 3            # min turns before topic appears in context
HIGH_VALENCE_THRESHOLD = 0.3 # intensity above this = high valence topic


# ─── Store ───────────────────────────────────────────────────────────────────

class AffectiveStore:
    """
    Stores topic→emotion vectors for one user.

    Usage:
        store = AffectiveStore(user_id="user_001")

        # Update after each turn
        store.update(
            topic="career",
            group_scores={"fear": 0.8, "neutral": 0.2, ...},
            intensity=0.8,
        )

        # Retrieve at inference time
        record = store.get("career")
        print(record.to_prompt_string())
        # "User feels strongly fearful about 'career'."

        # Get full context string for working memory
        context = store.to_context()
        # "[EMOTIONAL CONTEXT]
        #  User feels strongly fearful about 'career'.
        #  User feels somewhat guilty about 'health'."
    """

    def __init__(
        self,
        user_id: str = "default_user",
        storage_dir: str = "data/processed/affective",
        alpha: float = DEFAULT_ALPHA,
    ):
        self.user_id = user_id
        self.alpha = alpha
        self.storage_path = Path(storage_dir) / f"{user_id}.json"
        Path(storage_dir).mkdir(parents=True, exist_ok=True)

        # In-memory store: {topic: {emotion: score}}
        self._vectors: dict[str, dict[str, float]] = {}
        # Mention counts: {topic: int}
        self._mentions: dict[str, int] = {}

        # Load existing data if available
        self._load()

    # ── Write ─────────────────────────────────────────────────────────────

    def update(
        self,
        topic: str,
        group_scores: dict[str, float],
        intensity: float,
    ) -> None:
        """
        Update topic emotion vector via EMA.

        Args:
            topic:        domain name e.g. "career"
            group_scores: {emotion: score} from EmotionClassifier
            intensity:    how emotional was this turn (0-1)

        Skip if intensity < MIN_INTENSITY — neutral turns add noise.
        """
        # Skip low-intensity turns
        if intensity < MIN_INTENSITY:
            return

        # Initialize vector if first time seeing this topic
        if topic not in self._vectors:
            self._vectors[topic] = {
                "joy":     0.0,
                "sadness": 0.0,
                "fear":    0.0,
                "anger":   0.0,
                "guilt":   0.0,
                "neutral": 1.0,  # start neutral
            }
            self._mentions[topic] = 0

        # EMA update for each emotion dimension
        old_vector = self._vectors[topic]
        new_vector = {}

        for emotion in old_vector:
            old_score = old_vector[emotion]
            new_signal = group_scores.get(emotion, 0.0)
            # EMA: new = (1-alpha) * old + alpha * new_signal
            new_vector[emotion] = (
                (1 - self.alpha) * old_score +
                self.alpha * new_signal
            )

        self._vectors[topic] = new_vector
        self._mentions[topic] = self._mentions.get(topic, 0) + 1

        # Save to disk after each update
        self._save()

    def update_from_turn(
        self,
        topics: list[str],
        group_scores: dict[str, float],
        intensity: float,
    ) -> None:
        """
        Update all detected topics from one turn.
        Called by the session consolidation pipeline.
        """
        for topic in topics:
            self.update(topic, group_scores, intensity)

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, topic: str) -> AffectiveRecord | None:
        """
        Get the affective record for one topic.
        Returns None if topic not seen enough times.
        """
        if topic not in self._vectors:
            return None
        if self._mentions.get(topic, 0) < MIN_MENTIONS:
            return None

        vector = self._vectors[topic]
        emotion_vec = EmotionVector(
            joy=vector.get("joy", 0.0),
            sadness=vector.get("sadness", 0.0),
            fear=vector.get("fear", 0.0),
            anger=vector.get("anger", 0.0),
            guilt=vector.get("guilt", 0.0),
            neutral=vector.get("neutral", 1.0),
        )

        return AffectiveRecord(
            user_id=self.user_id,
            topic=topic,
            emotions=emotion_vec,
            mention_count=self._mentions.get(topic, 0),
            last_mentioned_at=datetime.utcnow(),
        )

    def get_all(self) -> list[AffectiveRecord]:
        """Get all topics that have enough mentions."""
        records = []
        for topic in self._vectors:
            record = self.get(topic)
            if record is not None:
                records.append(record)
        return records

    def to_context(self, max_topics: int = 5) -> str:
        """
        Assemble affective context string for working memory injection.

        Returns only high-valence topics — neutral topics add no value.
        Sorted by intensity (most emotional first).

        Example output:
            [EMOTIONAL CONTEXT]
            User feels strongly fearful about 'career'.
            User feels somewhat guilty about 'health'.
        """
        records = self.get_all()

        # Filter to high-valence topics only
        high_valence = [
            r for r in records
            if r.emotions.intensity() >= HIGH_VALENCE_THRESHOLD
        ]

        # Sort by intensity descending
        high_valence.sort(
            key=lambda r: r.emotions.intensity(),
            reverse=True,
        )

        # Take top N
        top = high_valence[:max_topics]

        if not top:
            return ""

        lines = ["[EMOTIONAL CONTEXT]"]
        for record in top:
            line = record.to_prompt_string()
            if line:
                lines.append(line)

        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary of current affective state."""
        return {
            "user_id": self.user_id,
            "topics_tracked": len(self._vectors),
            "topics_with_enough_mentions": len(self.get_all()),
            "topic_mentions": self._mentions,
        }

    def clear(self) -> None:
        """Reset all affective memory."""
        self._vectors = {}
        self._mentions = {}
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save vectors and mentions to JSON."""
        data = {
            "user_id": self.user_id,
            "alpha": self.alpha,
            "vectors": self._vectors,
            "mentions": self._mentions,
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load vectors and mentions from JSON if file exists."""
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            self._vectors  = data.get("vectors", {})
            self._mentions = data.get("mentions", {})
        except Exception as e:
            print(f"AffectiveStore load error: {e}")
            self._vectors  = {}
            self._mentions = {}
