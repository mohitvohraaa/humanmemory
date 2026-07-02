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

**Evaluation methodology:** built an independent evaluator that computes
expected emotion via majority vote over raw classifications — no EMA, no
accumulation, no smoothing. Agreement between the store's EMA-based
prediction and this independent ground truth validates that the affective
layer is internally correct, not just self-consistent.

**Mixed emotion stress test:** career topic with 4 fear + 4 joy turns
(0.246 vs 0.266 — 2% margin). Without a margin threshold, the system
confidently returned "joyful" despite near-tied scores. Adding
`MARGIN_THRESHOLD=0.1` now correctly returns `mixed:joy+fear`, which
formats as *"User has mixed feelings about career, feeling both joyful
and fearful."* — one representation, one source of truth.

**Affective impact finding (LLM-judge evaluation):** affective context's
value is conditional on episodic memory's emotional sparsity — redundant
when retrieved memories already contain emotional language (episodic-only
won 3/3 in that case), but adds genuine signal when episodic memories are
factually flat (episodic+affective won 3/3 in that case). The judge
parser was validated for positional bias: same winner regardless of
response order.

### Semantic Memory (Layer 3)

Semantic memory asks a harder question than episodic: not *what happened*,
but *what should the AI permanently learn about this person?*

**Topic ≠ Category:** TopicTagger outputs "Career", but semantic knowledge
categories are Goal / Preference / Concern / Identity. Reusing topic labels
would have silently built the wrong user profile — the architecture's most
dangerous failure mode.

**Single LLM call design:** fact extraction and category prediction are
collapsed into one structured output (`{"fact": "...", "category": "..."}`)
for consistency — two independent calls could eventually disagree.

**Contradiction detection:** new facts are compared against active facts
within the same category. On conflict, recency wins: old facts are marked
stale (not deleted), preserving full history.

**Known limitation:** turns tagged into multiple topics can generate
near-duplicate facts across categories — not yet deduplicated.

## Blog series

- [Week 1: Everything That Broke](#) — retrieval baselines, synthetic data failures
- [Week 2: Why My AI Companion Thought Fear Was Neutral](#) — affective memory, the neutral-vs-fear bug
- [Week 3 Part 1: Building the Evaluator](https://mohitvohraaa.substack.com/p/building-humanmemory-week-3part-1?r=2y4jgp) — independent ground truth, mixed emotion handling
- [Week 3 Part 2: Does Affective Memory Help?](https://mohitvohraaa.substack.com/p/building-humanmemory-week-3-does?r=2y4jgp) — A/B impact testing, conditional value finding
- [Week 3 Part 3: Building Semantic Memory](https://mohitvohraaa.substack.com/p/building-humanmemory-week-3part-3?r=2y4jgp) — fact extraction, contradiction detection

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
