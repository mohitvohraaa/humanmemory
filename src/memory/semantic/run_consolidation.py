"""
src/memory/semantic/run_consolidation.py
───────────────────────────────────────────
Connects topic tagging, fact extraction, and semantic storage
into one function that runs at session end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.memory.affective.topic_tagger import TopicTagger
from src.memory.semantic.consolidation import extract_fact
from src.memory.semantic.semantic_store import SemanticStore

MIN_TURNS_FOR_CONSOLIDATION = 2  # need at least this many same-topic turns


def consolidate_session(
    turns: list[dict],  # [{"text": str, "turn_id": str}, ...]
    semantic_store: SemanticStore,
    topic_tagger: TopicTagger | None = None,
    groq_client=None,
) -> list[dict]:
    """
    Groups session turns by topic, extracts facts per topic cluster,
    stores any extracted facts. Returns list of what was added.

    If turns contain a "topics" key, those are used directly (pre-tagged).
    Otherwise, topic_tagger must be provided to tag turns on the fly.
    """
    # Step 1 — group turns by topic
    topic_clusters: dict[str, list[dict]] = {}
    for turn in turns:
        if "topics" in turn:
            topics = turn["topics"]
        elif topic_tagger is not None:
            topics = topic_tagger.tag(turn["text"])
        else:
            continue
        for topic in topics:
            topic_clusters.setdefault(topic, []).append(turn)

    # Step 2 — extract + store a fact per topic with enough turns
    added_facts = []
    for topic, cluster_turns in topic_clusters.items():
        if len(cluster_turns) < MIN_TURNS_FOR_CONSOLIDATION:
            continue

        texts = [t["text"] for t in cluster_turns]
        result = extract_fact(texts, groq_client=groq_client)
        if result is None:
            continue

        episode_ids = [t["turn_id"] for t in cluster_turns]
        fact = semantic_store.add_fact(
            fact_text=result["fact"],
            category=result["category"],
            source_episode_ids=episode_ids,
        )
        added_facts.append({"topic": topic, "fact": fact.fact_text,
                             "category": fact.category.value})

    return added_facts
