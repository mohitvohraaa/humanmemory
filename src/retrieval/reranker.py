"""
src/retrieval/reranker.py
──────────────────────────
Reranks ChromaDB candidates using Park et al. (2023) formula.

Formula:
  final_score = 0.40 × semantic_similarity
              + 0.25 × recency_decay
              + 0.20 × importance_score
              + 0.15 × valence_boost

Today we implement recency_decay.
importance_score and valence_boost are placeholders (Week 2).

Paper grounding:
  Park et al. (2023) Generative Agents:
    retrieval_score = α×recency + β×importance + γ×relevance
    all normalized to [0,1], equal weights α=β=γ=1/3
  
  We use slightly different weights based on our ablation plan.
  Ebbinghaus (1885): R(t) = e^(-t/τ) forgetting curve.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional


# ─── Weights (from config — hardcoded here for clarity) ──────────────────────

W_SEMANTIC   = 0.40
W_RECENCY    = 0.25
W_IMPORTANCE = 0.20
W_VALENCE    = 0.15

# Decay constant — 14 days half-life
# Tune this via ablation (try 7, 14, 30)
TAU_DAYS = 14.0


# ─── Individual scoring functions ────────────────────────────────────────────

def ebbinghaus_decay(created_at: str, tau_days: float = TAU_DAYS) -> float:
    """
    Compute recency score using Ebbinghaus forgetting curve.
    
    R(t) = e^(-t / τ)
    
    Args:
        created_at: ISO format timestamp string
        tau_days:   decay constant in days
    
    Returns:
        float between 0 and 1
        1.0 = created right now
        0.37 = created tau_days ago
        0.0+ = very old memory (never exactly 0)
    
    Example:
        ebbinghaus_decay("2024-01-01", tau_days=14)
        → depends on how long ago Jan 1 was
    """
    try:
        # Parse the timestamp
        if isinstance(created_at, str):
            created_dt = datetime.fromisoformat(created_at)
        else:
            created_dt = created_at

        # Make timezone-aware if needed
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - created_dt).total_seconds() / 86400.0

        # Ebbinghaus formula
        return math.exp(-age_days / tau_days)

    except Exception:
        return 0.5  # safe fallback


def importance_score(valence: float, emotion_labels: list[str]) -> float:
    """
    Compute importance score from emotional signals.
    
    High emotion = high importance.
    Placeholder for now — will be improved in Week 2
    when we have the fine-tuned emotion classifier.
    
    Args:
        valence:       -1 to +1 emotional valence
        emotion_labels: list of emotion strings
    
    Returns:
        float between 0 and 1
    """
    # For now: absolute valence = importance
    # Strongly positive OR strongly negative = important
    base = abs(valence)

    # Small boost if emotion was explicitly detected
    if emotion_labels and emotion_labels != ["neutral"]:
        base = min(1.0, base + 0.1)

    return base


def valence_boost(valence: float) -> float:
    """
    Boost score for emotionally charged memories.
    
    Rationale: if the current query has emotional content,
    emotionally significant past memories are more relevant.
    
    For now returns absolute valence.
    Week 2: will be conditioned on current query emotion.
    
    Args:
        valence: -1 to +1
    
    Returns:
        float between 0 and 1
    """
    return abs(valence)


# ─── Main reranker ────────────────────────────────────────────────────────────

def rerank(
    candidates: list[dict],
    tau_days: float = TAU_DAYS,
    weights: Optional[dict] = None,
) -> list[dict]:
    """
    Rerank ChromaDB candidates using Park et al. (2023) formula.
    
    Args:
        candidates: list of dicts from SemanticEpisodicStore.semantic_retrieve()
                    each has: id, text, score, metadata
        tau_days:   Ebbinghaus decay constant
        weights:    optional override for scoring weights
    
    Returns:
        same list, reranked, with 'final_score' added to each item
    
    Example:
        candidates = chroma_store.semantic_retrieve(query, limit=20)
        reranked   = rerank(candidates)
        top_5      = reranked[:5]
    """
    if weights is None:
        weights = {
            "semantic":   W_SEMANTIC,
            "recency":    W_RECENCY,
            "importance": W_IMPORTANCE,
            "valence":    W_VALENCE,
        }

    for candidate in candidates:
        metadata = candidate.get("metadata", {})

        # 1. Semantic similarity (already computed by ChromaDB)
        sem_score = candidate.get("score", 0.0)

        # 2. Recency decay
        created_at = metadata.get("created_at", "")
        rec_score = ebbinghaus_decay(created_at, tau_days=tau_days)

        # 3. Importance score
        valence = float(metadata.get("valence_score", 0.0))
        emotion_labels = metadata.get("emotion_labels", "").split(",") \
            if metadata.get("emotion_labels") else []
        imp_score = importance_score(valence, emotion_labels)

        # 4. Valence boost
        val_boost = valence_boost(valence)

        # 5. Final weighted score
        final = (
            weights["semantic"]   * sem_score +
            weights["recency"]    * rec_score +
            weights["importance"] * imp_score +
            weights["valence"]    * val_boost
        )

        # Store all component scores for inspection / ablation
        candidate["recency_score"]    = rec_score
        candidate["importance_score"] = imp_score
        candidate["valence_boost"]    = val_boost
        candidate["final_score"]      = final

    # Sort by final score descending
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates
