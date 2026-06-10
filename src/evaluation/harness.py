"""
src/evaluation/harness.py
─────────────────────────
Evaluation harness — written FIRST before any model code.

What we measure:
  1. Recall@K    — did we find all relevant memories in top-K?
  2. Precision@K — how many retrieved memories were actually relevant?
  3. MRR         — average reciprocal rank across all queries
  4. Latency     — how fast is the retrieval pipeline?

Paper grounding:
  - Park et al. (2023): recency + importance + relevance scoring
    We measure the contribution of each component via ablation.
  - MemGPT: we use naive last-N retrieval as Baseline-1
  - CoALA: evaluation across episodic, semantic, procedural dimensions
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# ─── Result Dataclass ────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """
    All metrics for one retrieval configuration.
    One of these gets created per ablation config.
    """
    config_name: str = ""

    # Retrieval quality
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0        # mean across all queries — computed in aggregate_metrics()

    # Latency (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    def summary(self) -> dict:
        return {
            "config": self.config_name,
            "recall@1": round(self.recall_at_1, 3),
            "recall@5": round(self.recall_at_5, 3),
            "recall@10": round(self.recall_at_10, 3),
            "precision@5": round(self.precision_at_5, 3),
            "mrr": round(self.mrr, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
        }


# ─── Latency Timer ───────────────────────────────────────────────────────────

class LatencyTimer:
    """
    Measures how long each retrieval call takes.
    Use it as a context manager — wraps the code you want to time.

    Example:
        timer = LatencyTimer("chroma_query")
        for query in queries:
            with timer:
                result = chroma.query(query)
        print(timer.p95())
    """

    def __init__(self, name: str):
        self.name = name
        self._samples: list[float] = []
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._samples.append(elapsed_ms)

    def p50(self) -> float:
        return float(np.percentile(self._samples, 50)) if self._samples else 0.0

    def p95(self) -> float:
        return float(np.percentile(self._samples, 95)) if self._samples else 0.0

    def p99(self) -> float:
        return float(np.percentile(self._samples, 99)) if self._samples else 0.0

    def report(self) -> dict:
        return {
            f"{self.name}_p50_ms": round(self.p50(), 2),
            f"{self.name}_p95_ms": round(self.p95(), 2),
            f"{self.name}_p99_ms": round(self.p99(), 2),
            f"{self.name}_n_samples": len(self._samples),
        }


# ─── Core Metric Functions ───────────────────────────────────────────────────

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """
    What fraction of relevant memories did we find in the top-K results?

    Example:
        retrieved = ["A", "B", "C", "D", "E"]
        relevant  = {"A", "C", "F"}
        recall_at_k(retrieved, relevant, k=5) = 2/3 = 0.667
        # Found A and C but not F in top 5
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """
    What fraction of the top-K results were actually relevant?

    Example:
        retrieved = ["A", "B", "C", "D", "E"]
        relevant  = {"A", "C"}
        precision_at_k(retrieved, relevant, k=5) = 2/5 = 0.4
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    """
    1 / rank of the first relevant result. For a SINGLE query.

    Example:
        retrieved = ["B", "C", "A", "D"]
        relevant  = {"A"}
        reciprocal_rank = 1/3 = 0.333
        # First relevant result was at rank 3

    NOTE: This is RR for one query.
    MRR = mean of this across all queries — computed in aggregate_metrics().
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def aggregate_metrics(
    per_query_results: list[dict],
    config_name: str = "",
) -> RetrievalMetrics:
    """
    Takes per-query results and averages them into a single RetrievalMetrics.
    This is where MRR gets computed — mean of reciprocal ranks across all queries.

    Args:
        per_query_results: list of dicts, each with keys:
            retrieved_ids, relevant_ids
        config_name: label for this ablation config

    Returns:
        RetrievalMetrics with all fields averaged across queries
    """
    n = len(per_query_results)
    if n == 0:
        return RetrievalMetrics(config_name=config_name)

    recall_1_scores  = []
    recall_3_scores  = []
    recall_5_scores  = []
    recall_10_scores = []
    prec_5_scores    = []
    prec_10_scores   = []
    rr_scores        = []

    for result in per_query_results:
        retrieved = result["retrieved_ids"]
        relevant  = result["relevant_ids"]

        recall_1_scores.append(recall_at_k(retrieved, relevant, 1))
        recall_3_scores.append(recall_at_k(retrieved, relevant, 3))
        recall_5_scores.append(recall_at_k(retrieved, relevant, 5))
        recall_10_scores.append(recall_at_k(retrieved, relevant, 10))
        prec_5_scores.append(precision_at_k(retrieved, relevant, 5))
        prec_10_scores.append(precision_at_k(retrieved, relevant, 10))
        rr_scores.append(reciprocal_rank(retrieved, relevant))

    return RetrievalMetrics(
        config_name=config_name,
        recall_at_1=float(np.mean(recall_1_scores)),
        recall_at_3=float(np.mean(recall_3_scores)),
        recall_at_5=float(np.mean(recall_5_scores)),
        recall_at_10=float(np.mean(recall_10_scores)),
        precision_at_5=float(np.mean(prec_5_scores)),
        precision_at_10=float(np.mean(prec_10_scores)),
        mrr=float(np.mean(rr_scores)),   # ← THE ACTUAL MEAN HAPPENS HERE
    )
