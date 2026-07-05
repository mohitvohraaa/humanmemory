"""
scripts/run_session_end.py
────────────────────────────
The unified session-end pipeline. ONE pass per turn:
  1. Classify emotion
  2. Tag topics
  3. Update affective store (EMA vectors)
  4. Attach topics to turn dict
Then ONE consolidation pass:
  5. Semantic consolidation using pre-tagged turns

This eliminates the redundant TopicTagger call that existed when
consolidate_session() ran its own tagging internally.
"""

import sys
sys.path.insert(0, '.')

from src.memory.affective.emotion_classifier import EmotionClassifier
from src.memory.affective.topic_tagger import TopicTagger
from src.memory.affective.affective_store import AffectiveStore
from src.memory.semantic.semantic_store import SemanticStore
from src.memory.semantic.run_consolidation import consolidate_session


def run_session_end(
    raw_turns: list[dict],
    emotion_clf: EmotionClassifier,
    topic_tagger: TopicTagger,
    affective_store: AffectiveStore,
    semantic_store: SemanticStore,
) -> dict:
    """
    Single pass over session turns — tags topics and classifies
    emotions ONCE, updates both affective and semantic layers.

    Args:
        raw_turns: [{"text": str, "turn_id": str}]

    Returns:
        summary dict with what was updated in each layer
    """
    tagged_turns = []

    print(f"Processing {len(raw_turns)} turns...")

    for turn in raw_turns:
        text = turn["text"]

        # ONE emotion classification per turn
        emotion_result = emotion_clf.classify(text)

        # ONE topic tagging per turn
        topics = topic_tagger.tag(text)

        # Update affective layer
        affective_store.update_from_turn(
            topics=topics,
            group_scores=emotion_result.group_scores,
            intensity=emotion_result.intensity,
        )

        # Attach topics to turn dict for consolidation reuse
        tagged_turns.append({
            "text": text,
            "turn_id": turn["turn_id"],
            "topics": topics,
        })

        print(f"  [{', '.join(topics) or 'no topic'}] "
              f"dominant={emotion_result.dominant_emotion} "
              f"intensity={emotion_result.intensity:.2f} "
              f"| {text[:50]}")

    # Semantic consolidation using pre-tagged turns (no extra API calls)
    added_facts = consolidate_session(tagged_turns, semantic_store)

    return {
        "turns_processed": len(raw_turns),
        "affective_context": affective_store.to_context(),
        "semantic_context": semantic_store.to_context(),
        "facts_added": added_facts,
    }


if __name__ == "__main__":
    emotion_clf = EmotionClassifier()
    topic_tagger = TopicTagger()
    affective_store = AffectiveStore(user_id="pipeline_test")
    semantic_store = SemanticStore(user_id="pipeline_test")

    affective_store.clear()
    semantic_store.clear()

    # Two sessions — simulates real cross-session production usage
    session_0 = [
        {"text": "I feel stuck. Everyone else seems to know what they want to do.", "turn_id": "s0t1"},
        {"text": "I am thinking about a career change but I am scared of failing.", "turn_id": "s0t2"},
        {"text": "My manager said I am up for a promotion but I am terrified.", "turn_id": "s0t3"},
        {"text": "My parents are visiting on Saturday.", "turn_id": "s0t4"},
    ]

    session_1 = [
        {"text": "I just got passed over for a promotion and I am trying not to panic.", "turn_id": "s1t1"},
        {"text": "I am worried I am not good enough for my current role.", "turn_id": "s1t2"},
        {"text": "Had a long call with my dad. We do not talk enough.", "turn_id": "s1t3"},
    ]

    print("=== SESSION 0 ===")
    result_0 = run_session_end(session_0, emotion_clf, topic_tagger, affective_store, semantic_store)
    print("\nAffective context after session 0:")
    print(result_0["affective_context"] or "(empty)")
    print("\nSemantic context after session 0:")
    print(result_0["semantic_context"] or "(empty)")
    print("\nFacts added:", result_0["facts_added"])

    print("\n=== SESSION 1 ===")
    result_1 = run_session_end(session_1, emotion_clf, topic_tagger, affective_store, semantic_store)
    print("\nAffective context after session 1:")
    print(result_1["affective_context"] or "(empty)")
    print("\nSemantic context after session 1:")
    print(result_1["semantic_context"] or "(empty)")
    print("\nFacts added:", result_1["facts_added"])
