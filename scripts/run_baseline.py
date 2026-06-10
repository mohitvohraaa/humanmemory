"""
scripts/run_baseline.py
───────────────────────
Runs naive and semantic baseline evaluations, then compares results.

Baseline-1 (naive): Last N messages by recency — our zero-line
Baseline-2 (semantic): ChromaDB cosine similarity — meaning-based retrieval

Run with:
    python3 scripts/run_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Make sure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.harness import aggregate_metrics, LatencyTimer, RetrievalMetrics
from src.memory.episodic.store import EpisodicStore
from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.models import EpisodicMemory


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_turns_into_store(
    dataset: dict,
    sqlite_store: EpisodicStore,
    semantic_store: SemanticEpisodicStore | None = None,
) -> dict[str, str]:
    """
    Insert every turn from every session into the stores.
    Returns a mapping of {turn_id: text} for quick lookup.
    """
    turn_map = {}
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
            sqlite_store.add(mem)
            memories.append(mem)
            turn_map[turn["turn_id"]] = turn["text"]

    # Batch load into ChromaDB if semantic store provided
    if semantic_store is not None:
        print(f"Encoding {len(memories)} memories into vectors...")
        semantic_store.add_batch(memories)

    return turn_map


def run_naive_baseline(dataset: dict, store: EpisodicStore) -> RetrievalMetrics:
    """
    BASELINE-1: For each ground truth query, retrieve most recent N memories.
    No semantic understanding — just recency.
    """
    ground_truth = dataset["ground_truth"]
    timer = LatencyTimer("naive_retrieve")
    per_query_results = []

    for query_id, relevant_turn_ids in ground_truth.items():
        relevant_ids = set(relevant_turn_ids)

        with timer:
            retrieved = store.naive_retrieve(limit=10)

        retrieved_ids = [m.id for m in retrieved]

        per_query_results.append({
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })

    metrics = aggregate_metrics(per_query_results, config_name="baseline_naive")
    metrics.latency_p50_ms = timer.p50()
    metrics.latency_p95_ms = timer.p95()
    metrics.latency_p99_ms = timer.p99()

    return metrics


def run_semantic_baseline(
    dataset: dict,
    semantic_store: SemanticEpisodicStore,
) -> RetrievalMetrics:
    """
    BASELINE-2: For each ground truth query, retrieve most semantically similar
    memories using ChromaDB cosine similarity.

    The dataset has two dicts:
      - ground_truth: {query_id: [turn_ids]}
      - queries: {query_text: query_id}

    We need to invert queries to get {query_id: query_text} for lookup.
    """
    ground_truth = dataset["ground_truth"]
    queries = dataset.get("queries", {})

    timer = LatencyTimer("semantic_retrieve")
    per_query_results = []

    for query_id, relevant_turn_ids in ground_truth.items():
        relevant_ids = set(relevant_turn_ids)

        # Get query text directly — queries dict is {qid: query_text}
        query_text = queries.get(query_id, None)

        # Skip if no query text found
        if query_text is None:
            continue

        with timer:
            results = semantic_store.semantic_retrieve(query_text, limit=10)

        retrieved_ids = [r["id"] for r in results]

        per_query_results.append({
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })

    metrics = aggregate_metrics(per_query_results, config_name="baseline_semantic")
    metrics.latency_p50_ms = timer.p50()
    metrics.latency_p95_ms = timer.p95()
    metrics.latency_p99_ms = timer.p99()

    return metrics


def run_semantic_baseline_relaxed(
    dataset: dict,
    semantic_store: SemanticEpisodicStore,
) -> RetrievalMetrics:
    """
    BASELINE-2 (relaxed): For each ground truth query, retrieve most semantically
    similar memories. Checks if ANY turn with the same TEXT is in results,
    not just the exact turn ID (handles duplicate turns in synthetic data).
    """
    ground_truth = dataset["ground_truth"]
    queries = dataset.get("queries", {})

    # Build turn_id -> text mapping
    turn_id_to_text = {}
    for session in dataset["sessions"]:
        for turn in session["turns"]:
            turn_id_to_text[turn["turn_id"]] = turn["text"]

    timer = LatencyTimer("semantic_retrieve")
    per_query_results = []

    for query_id, relevant_turn_ids in ground_truth.items():
        query_text = queries.get(query_id, None)
        if query_text is None:
            continue

        # Get the expected text (use first relevant turn)
        expected_text = turn_id_to_text.get(relevant_turn_ids[0], None)
        if expected_text is None:
            continue

        with timer:
            results = semantic_store.semantic_retrieve(query_text, limit=10)

        retrieved_texts = [r["text"] for r in results]

        # Relaxed match: check if any retrieved text matches expected text
        per_query_results.append({
            "retrieved_ids": retrieved_texts,  # treating texts as IDs for relaxed match
            "relevant_ids": {expected_text},
        })

    metrics = aggregate_metrics(per_query_results, config_name="baseline_semantic_relaxed")
    metrics.latency_p50_ms = timer.p50()
    metrics.latency_p95_ms = timer.p95()
    metrics.latency_p99_ms = timer.p99()

    return metrics


def print_comparison(
    naive_metrics: RetrievalMetrics,
    semantic_metrics: RetrievalMetrics,
    semantic_relaxed_metrics: RetrievalMetrics | None = None,
    naive_count: int = 0,
    semantic_count: int = 0,
    queries_count: int = 0,
) -> None:
    """Print a side-by-side comparison table."""
    print("\n" + "═" * 70)
    print("  BASELINE COMPARISON")
    print("═" * 70)

    # Header
    if semantic_relaxed_metrics:
        print(f"  {'Metric':<20} {'Naive (B1)':<15} {'Semantic':<15} {'Semantic (relaxed)':<15}")
    else:
        print(f"  {'Metric':<20} {'Naive (B1)':<15} {'Semantic (B2)':<15}")
    print("─" * 70)

    # Metrics comparison
    naive_summary = naive_metrics.summary()
    semantic_summary = semantic_metrics.summary()
    relaxed_summary = semantic_relaxed_metrics.summary() if semantic_relaxed_metrics else None

    metrics_to_compare = ["recall@1", "recall@5", "recall@10", "precision@5", "mrr"]

    for metric in metrics_to_compare:
        naive_val = naive_summary.get(metric, 0.0)
        semantic_val = semantic_summary.get(metric, 0.0)

        if semantic_relaxed_metrics:
            relaxed_val = relaxed_summary.get(metric, 0.0)
            print(f"  {metric:<20} {naive_val:<15} {semantic_val:<15} {relaxed_val:<15}")
        else:
            print(f"  {metric:<20} {naive_val:<15} {semantic_val:<15}")

    # Latency comparison
    print("─" * 70)
    print(f"  {'latency_p95_ms':<20} {naive_metrics.latency_p95_ms:<15.1f} {semantic_metrics.latency_p95_ms:<15.1f}")

    # Summary
    print("═" * 70)
    print(f"  Queries evaluated:   {queries_count}")
    print(f"  Memories (SQLite):   {naive_count}")
    print(f"  Memories (ChromaDB): {semantic_count}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    dataset_path = "data/synthetic/dataset_100.json"

    if not Path(dataset_path).exists():
        print("Dataset not found. Run this first:")
        print("  python3 data/synthetic/generator.py")
        sys.exit(1)

    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    # Initialize stores
    print("\n── Initializing stores ──")
    sqlite_store = EpisodicStore("data/processed/episodic.db")
    semantic_store = SemanticEpisodicStore(persist_dir="data/processed/chroma")

    # Clear for fresh run
    sqlite_store.clear()
    semantic_store.clear()

    # Load data into both stores
    print("\n── Loading turns into stores ──")
    turn_map = load_all_turns_into_store(dataset, sqlite_store, semantic_store)
    print(f"Loaded {sqlite_store.count()} memories into SQLite")
    print(f"Loaded {semantic_store.count()} memories into ChromaDB")

    # Run both baselines
    print("\n── Running evaluations ──")
    print("Running naive baseline (Baseline-1)...")
    naive_metrics = run_naive_baseline(dataset, sqlite_store)

    print("Running semantic baseline (Baseline-2)...")
    semantic_metrics = run_semantic_baseline(dataset, semantic_store)

    print("Running semantic baseline (relaxed match)...")
    semantic_relaxed_metrics = run_semantic_baseline_relaxed(dataset, semantic_store)

    # Print comparison
    print_comparison(
        naive_metrics,
        semantic_metrics,
        semantic_relaxed_metrics,
        naive_count=sqlite_store.count(),
        semantic_count=semantic_store.count(),
        queries_count=len(dataset["ground_truth"]),
    )
