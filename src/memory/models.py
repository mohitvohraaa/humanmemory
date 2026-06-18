"""
src/memory/models.py
────────────────────
Single source of truth for ALL memory data structures.
Every layer imports from here. Never define schemas elsewhere.

Paper grounding:
  - EpisodicMemory: Park et al. (2023) recency+importance+relevance scoring
  - topic_tags field: foreign key that powers the Affective layer (our novel contribution)
  - SemanticFact: mirrors Claude's 5-category profile schema
  - AffectiveRecord: novel layer — topic→emotion map absent from all prior work
  - ProceduralRule: CoALA procedural memory as explicit rules
  - WorkingContext: CoALA working memory — assembled context passed to LLM
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ─── Enums ───────────────────────────────────────────────────────────────────

class EmotionClass(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    GUILT = "guilt"
    NEUTRAL = "neutral"


class FactCategory(str, Enum):
    IDENTITY = "identity"
    PREFERENCE = "preference"
    CONCERN = "concern"
    RELATIONSHIP = "relationship"
    GOAL = "goal"


# ─── Layer 1: Working Memory ─────────────────────────────────────────────────

@dataclass
class WorkingContext:
    """
    The assembled context string passed to the LLM at inference time.
    Built by the ContextAssembler from all lower layers.
    CoALA: working memory holds the agent's current state.
    """
    session_id: str
    turn_id: str
    current_message: str

    episodic_context: str = ""
    semantic_context: str = ""
    affective_context: str = ""
    procedural_context: str = ""

    total_tokens_used: int = 0
    assembly_latency_ms: float = 0.0
    assembled_at: datetime = field(default_factory=datetime.utcnow)

    def to_prompt_string(self) -> str:
        parts = []
        if self.semantic_context:
            parts.append(f"[USER PROFILE]\n{self.semantic_context}")
        if self.affective_context:
            parts.append(f"[EMOTIONAL CONTEXT]\n{self.affective_context}")
        if self.procedural_context:
            parts.append(f"[INTERACTION RULES]\n{self.procedural_context}")
        if self.episodic_context:
            parts.append(f"[RELEVANT MEMORIES]\n{self.episodic_context}")
        return "\n\n".join(parts)


# ─── Layer 2: Episodic Memory ─────────────────────────────────────────────────

@dataclass
class EpisodicMemory:
    """
    A single episodic memory — a specific past event with full context.

    Paper grounding:
      - Tulving (1972): episodic = instance-specific, temporally tagged
      - Park et al. (2023): recency + importance + relevance composite scoring
      - Our addition: topic_tags as FK to affective layer
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    turn_id: str = ""
    speaker: str = ""
    text: str = ""
    summary: str = ""

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None

    # Park et al. (2023) scoring
    importance_score: float = 0.0
    recency_score: float = 1.0
    relevance_score: float = 0.0

    # Our novel addition — emotional context
    valence_score: float = 0.0
    emotion_labels: list[str] = field(default_factory=list)

    # FK to AffectiveRecord — connects episodic to affective layer
    topic_tags: list[str] = field(default_factory=list)

    embedding_id: str = ""

    @property
    def composite_retrieval_score(self) -> float:
        """
        Park et al. (2023) retrieval formula.
        recency + importance + relevance, with emotional boost.
        """
        return (
            self.relevance_score  * 0.40 +
            self.recency_score    * 0.25 +
            self.importance_score * 0.20 +
            abs(self.valence_score) * 0.15
        )


# ─── Layer 3: Semantic Memory ─────────────────────────────────────────────────

@dataclass
class SemanticFact:
    """
    A generalised fact about the user, distilled from episodic clusters.
    Not tied to any specific event — abstracted truth.

    Tulving (1972): semantic = facts without episodic context.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""

    category: FactCategory = FactCategory.IDENTITY
    fact_text: str = ""

    source_episode_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    is_stale: bool = False

    first_observed_at: datetime = field(default_factory=datetime.utcnow)
    last_confirmed_at: datetime = field(default_factory=datetime.utcnow)


# ─── Layer 4: Affective Memory ────────────────────────────────────────────────

@dataclass
class EmotionVector:
    """Emotion scores for a single topic — matches EmotionClassifier's 6 groups."""
    joy: float = 0.0
    sadness: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    guilt: float = 0.0
    neutral: float = 1.0

    def dominant_emotion(self) -> str:
        scores = {
            "joy": self.joy, "sadness": self.sadness,
            "fear": self.fear, "anger": self.anger,
            "guilt": self.guilt, "neutral": self.neutral,
        }
        return max(scores, key=scores.get)

    def intensity(self) -> float:
        """How emotionally charged? 0 = flat, 1 = extreme."""
        return 1.0 - self.neutral


@dataclass
class AffectiveRecord:
    """
    Topic → emotion association map for one user.

    THIS IS OUR NOVEL LAYER.
    Absent from: MemGPT, Generative Agents, A-MEM, Mem0, Claude.
    This is what makes Ira feel emotionally aware.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    topic: str = ""
    emotions: EmotionVector = field(default_factory=EmotionVector)
    mention_count: int = 0
    last_mentioned_at: Optional[datetime] = None

    def is_high_valence(self, threshold: float = 0.6) -> bool:
        return self.emotions.intensity() >= threshold

    def to_prompt_string(self) -> str:
        dominant = self.emotions.dominant_emotion()
        intensity = self.emotions.intensity()
        if intensity < 0.2:
            return ""
        level = "strongly" if intensity > 0.7 else "somewhat"
        adjective_map = {
            "joy":     "joyful",
            "sadness": "sad",
            "fear":    "fearful",
            "anger":   "angry",
            "guilt":   "guilty",
            "neutral": "neutral",
        }
        dominant_adj = adjective_map.get(dominant, dominant)
        return f"User feels {level} {dominant_adj} about '{self.topic}'."


# ─── Layer 5: Procedural Memory ──────────────────────────────────────────────

@dataclass
class ProceduralRule:
    """
    A behavioural rule for how to interact with this user.
    CoALA: procedural memory = skills and rules that shape behaviour.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    condition: str = ""
    action: str = ""
    rule_text: str = ""
    confidence: float = 1.0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_prompt_string(self) -> str:
        return f"- {self.rule_text}"
