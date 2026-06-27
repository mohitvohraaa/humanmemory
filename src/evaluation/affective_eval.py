"""
src/evaluation/affective_eval.py
───────────────────────────────────
Evaluates whether AffectiveStore correctly identifies the
dominant emotion for a topic, given known ground-truth turns.

Separate from Recall@K — that measures retrieval, this measures
whether Layer 4's emotional interpretation is accurate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.memory.affective.affective_store import AffectiveStore
from src.memory.affective.emotion_classifier import EmotionClassifier
from src.memory.affective.topic_tagger import TopicTagger


@dataclass
class AffectiveEvalResult:
    topic: str
    predicted_emotion: str
    expected_emotion: str
    correct: bool
    mention_count: int
    intensity: float


def compute_expected_emotion(turns: list[str], emotion_clf: EmotionClassifier) -> str:
    """
    Ground truth = majority vote of per-turn dominant emotions.
    Independent of EMA — this is what we're checking the store against.
    """
    emotions = [emotion_clf.classify(t).dominant_emotion for t in turns]
    counts = Counter(emotions)
    return counts.most_common(1)[0][0]


def evaluate_affective_accuracy(
    topic_turns: dict[str, list[str]],
    affective_store: AffectiveStore,
    emotion_clf: EmotionClassifier,
) -> list[AffectiveEvalResult]:
    """
    Args:
        topic_turns: {"career": [turn1, turn2, ...], "health": [...]}
                     ground-truth turns already known to belong to each topic
        affective_store: already populated via update_from_turn() calls
        emotion_clf: same classifier used during population

    Returns: one AffectiveEvalResult per topic
    """
    results = []

    for topic, turns in topic_turns.items():
        expected = compute_expected_emotion(turns, emotion_clf)
        record = affective_store.get(topic)

        if record is None:
            # not enough mentions yet — counts as incorrect, but tracked separately
            results.append(AffectiveEvalResult(
                topic=topic,
                predicted_emotion="none",
                expected_emotion=expected,
                correct=False,
                mention_count=0,
                intensity=0.0,
            ))
            continue

        predicted = record.emotions.dominant_emotion()

        # Handle "mixed:emotion1+emotion2" format — count as correct
        # if the expected emotion is one of the two named in the mix.
        # A mixed result isn't wrong just because it's not a single
        # clean label; it's wrong only if the expected emotion isn't
        # represented at all.
        if predicted.startswith("mixed:"):
            mixed_emotions = predicted.replace("mixed:", "").split("+")
            is_correct = expected in mixed_emotions
        else:
            is_correct = (predicted == expected)

        results.append(AffectiveEvalResult(
            topic=topic,
            predicted_emotion=predicted,
            expected_emotion=expected,
            correct=is_correct,
            mention_count=record.mention_count,
            intensity=record.emotions.intensity(),
        ))

    return results


def accuracy_summary(results: list[AffectiveEvalResult]) -> dict:
    """Aggregate accuracy across all evaluated topics."""
    if not results:
        return {"accuracy": 0.0, "n_topics": 0}
    correct = sum(1 for r in results if r.correct)
    return {
        "accuracy": correct / len(results),
        "n_topics": len(results),
        "n_correct": correct,
    }
