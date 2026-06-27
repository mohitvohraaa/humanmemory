# HumanMemory

A 5-layer biologically-inspired memory architecture for companion AI — built as a direct implementation of the research problem behind long-term, emotionally-aware AI memory.

🔗 **W&B Dashboard:** https://wandb.ai/mohitvohraaa-netaji-subhas-university-of-technology/humanmemory

---

## The problem

Existing memory frameworks (MemGPT, Generative Agents, A-MEM, Mem0) remember *what happened*. None of them track *how the user felt about it* as a structured, queryable signal. HumanMemory adds that missing layer.

## Architecture

| Layer | Status | Key Component |
|---|---|---|
| 1 — Working Memory | ✅ | Context assembler, token-budget aware |
| 2 — Episodic Memory | ✅ | SQLite + ChromaDB, hybrid retrieval + adaptive reranking |
| 3 — Semantic Memory | ✅ | Durable facts from episodic clusters, contradiction detection |
| 4 — Affective Memory ★ | ✅ | Topic→emotion EMA vectors — the novel layer |
| 5 — Procedural Memory | ❌ | Not yet built (Week 4) |

## Results so far

### Retrieval (Week 1)

| Metric | Naive | Semantic |
|---|---|---|
| Recall@10 | 0.019 | 0.068 |
| MRR | 0.002 | 0.012 |
| p95 latency | 0.2ms | 10.6ms |

### Affective Layer (Week 2)

| Metric | Result |
|---|---|
| Accuracy (5 topics) | 1.0 |
| Margin threshold | 0.1 |
| Min mentions required | 3 |

**Affective impact finding (LLM-judge evaluation):** affective context's
value is conditional on episodic memory's emotional sparsity — redundant
when retrieved memories already contain emotional language (episodic-only
won 3/3 in that case), but adds genuine signal when episodic memories are
factually flat (episodic+affective won 3/3 in that case).

### Semantic Memory (Layer 3)

Consolidation pipeline: groups session turns by topic, extracts durable
facts via LLM (with contradiction detection against existing facts in the
same category). Recency wins on conflict; old facts marked stale, not
deleted, preserving full history.

**Known limitation:** turns tagged into multiple topics can generate
near-duplicate facts across categories — not yet deduplicated.

## Blog series

- [Week 1: Everything That Broke](#) — retrieval baselines, synthetic data failures
- [Week 2: Why My AI Companion Thought Fear Was Neutral](#) — affective memory, the neutral-vs-fear bug

## Setup

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Actively built in public, day by day. Week 4 (procedural memory) in progress.
