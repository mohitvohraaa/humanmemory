"""
scripts/run_baseline.py
────────────────────────
Runs two evaluations and prints a comparison table:
  1. Baseline-1: naive last-N retrieval     (expects ~0.0)
  2. Baseline-2: semantic ChromaDB retrieval (expects ~0.4-0.6)

Run with:
    python3 scripts/run_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.harness import aggregate_metrics, LatencyTimer
from src.memory.episodic.store import EpisodicStore
from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.models import EpisodicMemory


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_turns_into_stores(
    dataset: dict,
    sqlite_store: EpisodicStore,
    chroma_store: SemanticEpisodicStore,
) -> None:
    """Load all turns into both stores in one pass."""
    memories = []
    for session in dataset["sessions"]:
        for turn in session["turns"]:
            mem = EpisodicMemory(
                id=turn["turn_id"],
                session_id=turn["session_id"],
                turn_id=turn["turn_id"],
                text=turn["text"],
                topic_tags=turn["topic_tags"],
                emotion_labels=turn["emotion_labels"],
                valence_score=turn["valence_score"],
                created_at=datetime.fromisoformat(turn["created_at"]),
            )
            memories.append(mem)
            sqlite_store.add(mem)

    # Batch encode all memories at once — much faster than one by one
    print(f"Encoding {len(memories)} memories into ChromaDB...")
    chroma_store.add_batch(memories)


def run_naive(dataset: dict, store: EpisodicStore) -> dict:
    """Baseline-1: retrieve last 10 messages, ignore query entirely."""
    timer = LatencyTimer("naive")
    results = []

    for qid, relevant_ids in dataset["ground_truth"].items():
        with timer:
            retrieved = store.naive_retrieve(limit=10)
        results.append({
            "retrieved_ids": [m.id for m in retrieved],
            "relevant_ids": set(relevant_ids),
        })

    metrics = aggregate_metrics(results, config_name="baseline_naive")
    metrics.latency_p95_ms = timer.p95()
    return metrics


def run_semantic(dataset: dict, store: SemanticEpisodicStore) -> dict:
    """Baseline-2: encode query, find semantically similar memories."""
    timer = LatencyTimer("semantic")
    results = []
    skipped = 0

    for qid, relevant_ids in dataset["ground_truth"].items():
        # Look up query text by ID
        query_text = dataset["queries"].get(qid)
        if not query_text:
            skipped += 1
            continue

        with timer:
            retrieved = store.semantic_retrieve(query_text, limit=10)

        results.append({
            "retrieved_ids": [r["id"] for r in retrieved],
            "relevant_ids": set(relevant_ids),
        })

    if skipped > 0:
        print(f"  Skipped {skipped} queries with no text")

    metrics = aggregate_metrics(results, config_name="semantic")
    metrics.latency_p95_ms = timer.p95()
    return metrics


def print_comparison(naive, semantic):
    """Print a clean side-by-side comparison table."""
    print("\n" + "─" * 60)
    print(f"  {'Metric':<20} {'Naive':>10} {'Semantic':>10} {'Delta':>10}")
    print("─" * 60)

    fields = [
        ("recall@1",  "recall_at_1"),
        ("recall@5",  "recall_at_5"),
        ("recall@10", "recall_at_10"),
        ("precision@5","precision_at_5"),
        ("mrr",       "mrr"),
        ("p95_ms",    "latency_p95_ms"),
    ]

    for label, attr in fields:
        n = getattr(naive, attr)
        s = getattr(semantic, attr)
        delta = s - n
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        print(f"  {label:<20} {n:>10.3f} {s:>10.3f} {delta_str:>10}")

    print("─" * 60 + "\n")


if __name__ == "__main__":
    dataset_path = "data/synthetic/dataset_100.json"

    if not Path(dataset_path).exists():
        print("Dataset not found. Run: python3 data/synthetic/generator.py")
        sys.exit(1)

    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"Sessions: {len(dataset['sessions'])}")
    print(f"Queries:  {len(dataset['ground_truth'])}")

    # Initialize stores
    sqlite_store = EpisodicStore("data/processed/episodic.db")
    chroma_store = SemanticEpisodicStore(
        persist_dir="data/processed/chroma",
        collection_name="episodic_memories",
    )

    # Fresh run
    sqlite_store.clear()
    chroma_store.clear()

    # Load all turns
    print("\nLoading turns into stores...")
    load_turns_into_stores(dataset, sqlite_store, chroma_store)
    print(f"SQLite:   {sqlite_store.count()} memories")
    print(f"ChromaDB: {chroma_store.count()} memories")

    # Run evaluations
    print("\nRunning naive baseline...")
    naive_metrics = run_naive(dataset, sqlite_store)

    print("Running semantic baseline...")
    semantic_metrics = run_semantic(dataset, chroma_store)

    # Print comparison
    print_comparison(naive_metrics, semantic_metrics)
