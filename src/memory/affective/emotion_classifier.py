"""
src/memory/affective/emotion_classifier.py
───────────────────────────────────────────
Emotion classifier for episodic memory turns.

Uses GoEmotions 28-class model (monologg/bert-base-cased-goemotions-original)
mapped into 6 companion-AI-relevant emotion groups.

Why 28-class over 6-class:
  6-class model maps "I miss my dad" → anger (wrong)
  28-class model maps it → disappointment → sadness (correct)

Why we group into 6:
  28 classes are too granular for affective memory vectors
  6 groups capture what matters for companion AI:
  joy, sadness, fear, anger, guilt, neutral

Paper grounding:
  Demszky et al. (2020) — GoEmotions dataset
  58k Reddit comments, 28 emotion labels, Google Research
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline


# ─── Emotion grouping ────────────────────────────────────────────────────────

EMOTION_GROUPS: dict[str, list[str]] = {
    "joy": [
        "joy", "excitement", "amusement", "pride",
        "optimism", "gratitude", "admiration", "love",
        "relief", "caring",
    ],
    "sadness": [
        "sadness", "disappointment", "grief",
        "remorse", "embarrassment", "longing",
    ],
    "fear": [
        "fear", "nervousness",
    ],
    "anger": [
        "anger", "annoyance", "disapproval",
        "disgust", "contempt",
    ],
    "guilt": [
        "guilt", "shame", "regret",
    ],
    "neutral": [
        "neutral", "realization", "surprise",
        "confusion", "curiosity", "desire",
    ],
}

# Reverse map: raw label → group
LABEL_TO_GROUP: dict[str, str] = {}
for group, labels in EMOTION_GROUPS.items():
    for label in labels:
        LABEL_TO_GROUP[label] = group


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class EmotionResult:
    """
    Emotion classification result for one text.

    dominant_emotion: the strongest emotion group
    group_scores:     scores for all 6 groups (sum to ~1.0)
    valence_score:    -1 to +1 (negative to positive)
    intensity:        0 to 1 (how emotional vs neutral)
    raw_top3:         top 3 raw 28-class labels for debugging
    """
    dominant_emotion: str
    group_scores: dict[str, float]
    valence_score: float
    intensity: float
    raw_top3: list[dict]

    def to_prompt_string(self) -> str:
        """Human-readable summary for debugging."""
        scores = sorted(
            self.group_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top = [(k, round(v, 2)) for k, v in scores if v > 0.05]
        return f"{self.dominant_emotion} ({', '.join(f'{k}:{v}' for k,v in top[:3])})"


# ─── Classifier ──────────────────────────────────────────────────────────────

# Valence map — is this emotion positive or negative?
VALENCE_MAP = {
    "joy":     +1.0,
    "sadness": -0.7,
    "fear":    -0.8,
    "anger":   -0.6,
    "guilt":   -0.5,
    "neutral":  0.0,
}


class EmotionClassifier:
    """
    Wraps GoEmotions 28-class model with grouping and valence scoring.

    Usage:
        clf = EmotionClassifier()
        result = clf.classify("I feel stuck at work")
        print(result.dominant_emotion)  # sadness
        print(result.valence_score)     # -0.49
        print(result.intensity)         # 0.31
    """

    def __init__(
        self,
        model_name: str = "monologg/bert-base-cased-goemotions-original",
    ):
        print(f"Loading emotion classifier: {model_name}")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
        )
        print("Emotion classifier ready")

    def classify(self, text: str) -> EmotionResult:
        """
        Classify the emotion in a text.

        Steps:
          1. Run 28-class GoEmotions model
          2. Group scores into 6 groups
          3. Compute valence and intensity
          4. Return EmotionResult
        """
        # Step 1 — run model
        raw_output = self.classifier(text)
        raw_scores = raw_output[0] if isinstance(raw_output[0], list) else raw_output
        raw_scores.sort(key=lambda x: x["score"], reverse=True)

        # Step 2 — aggregate into 6 groups
        group_scores: dict[str, float] = {g: 0.0 for g in EMOTION_GROUPS}
        for item in raw_scores:
            label = item["label"]
            score = item["score"]
            group = LABEL_TO_GROUP.get(label, "neutral")
            group_scores[group] += score

        # Normalize group scores to sum to 1
        total = sum(group_scores.values())
        if total > 0:
            group_scores = {k: v / total for k, v in group_scores.items()}

        # Step 3 — find dominant emotion
        dominant = max(group_scores, key=group_scores.get)

        # Step 4 — compute valence score
        # weighted sum of valence × group score
        valence = sum(
            VALENCE_MAP[group] * score
            for group, score in group_scores.items()
        )

        # Step 5 — compute intensity
        # how far from neutral? 0 = completely neutral, 1 = extreme emotion
        intensity = 1.0 - group_scores.get("neutral", 0.0)

        return EmotionResult(
            dominant_emotion=dominant,
            group_scores=group_scores,
            valence_score=round(valence, 3),
            intensity=round(intensity, 3),
            raw_top3=raw_scores[:3],
        )

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        """Classify multiple texts efficiently."""
        return [self.classify(text) for text in texts]
