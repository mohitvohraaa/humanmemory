# HumanMemory

A 5-layer biologically-inspired memory service for LLM applications — built as a direct implementation of the research problem behind long-term, emotionally-aware AI memory.

The service **never generates responses**. It provides context retrieval, turn storage, and session consolidation. Your LLM handles generation.

W&B Dashboard: https://wandb.ai/mohitvohraaa-netaji-subhas-university-of-technology/humanmemory

---

## Quick Start (SDK)

```bash
pip install -e ".[api,dev]"
export GROQ_API_KEY="gsk_..."  # for internal memory operations
```

```python
from humanmemory import MemoryClient

mem = MemoryClient(user_id="alice")

# 1. Get context for incoming message
context = mem.get_context("How's my mom doing?")

# 2. Call your own LLM with the context
response = your_llm.chat(
    system=context.to_prompt_string(),
    user="How's my mom doing?"
)

# 3. Store the conversation turn
mem.store(message="How's my mom doing?", response=response)

# 4. At session end, consolidate
mem.consolidate()
```

### How `get_context()` works

When you call `mem.get_context("How's my mom doing?")`, here's what happens internally:

1. **Classify query** — Groq determines if this is a `recent`, `long_term`, or `specific` query (~5 tokens, ~60ms)
2. **Retrieve candidates** — ChromaDB vector search finds 20 nearest memories by semantic similarity
3. **Rerank** — Park et al. formula reranks candidates using query-adaptive weights (e.g., `long_term` boosts importance over recency)
4. **Pull affective context** — topic→emotion vectors (e.g., "User feels fearful about 'family'")
5. **Pull semantic context** — durable user profile facts (e.g., "User is close to their mother")
6. **Assemble** — combines all three into a single prompt string via `to_prompt_string()`

**One Groq call total.** The rest is vector search + math. No LLM generation, no writing — pure retrieval.

```
store()        → writes to episodic, affective, semantic stores
consolidate()  → extracts facts, writes to semantic store
get_context()  → reads from all three stores, assembles context
```

---

## Architecture

```
Client App
    │
    ├── POST /context ──→ Memory Orchestration ──→ Retrieval + Storage
    │       returns context                            │
    │                                                  ▼
    ├── (client calls their own LLM with context)
    │
    ├── POST /store ──→ Memory Orchestration ──→ Storage + Affective + Indexing
    │       stores turn
    │
    └── POST /consolidate ──→ Consolidation Pipeline ──→ Semantic + Affective Updates
            end of session
```

**Internal pipeline uses Groq** for query classification, topic tagging, and fact extraction. **Generation uses the user's own LLM** — completely external, never touched by our service.

| Layer | Status | Key Component |
|---|---|---|
| 1 — Working Memory | ✅ | Context assembler, 3-layer prompt assembly |
| 2 — Episodic Memory | ✅ | SQLite + ChromaDB, hybrid retrieval + adaptive reranking |
| 3 — Semantic Memory | ✅ | Durable facts from episodic clusters, contradiction detection |
| 4 — Affective Memory ★ | ✅ | Topic→emotion EMA vectors — the novel layer |
| 5 — Procedural Memory | ❌ | Not yet built |

---

## API Server

```bash
uvicorn humanmemory.api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/context` | POST | Retrieve assembled memory context for a message |
| `/store` | POST | Store a completed conversation turn |
| `/consolidate` | POST | End-of-session consolidation (extract facts, update memories) |
| `/session/end` | POST | Alias for `/consolidate` (backward compatible) |
| `/memory/{user_id}` | GET | Inspect semantic facts, affective profile, episodic count |
| `/memory/{user_id}` | DELETE | Clear all memory for a user |
| `/health` | GET | Health check |

### Client Flow

```
1. POST /context      → get assembled memory context
2. (call your own LLM with the context)
3. POST /store        → store the conversation turn
4. POST /consolidate  → end-of-session processing
```

---

## Results

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

---

## Blog Series

- [Week 1: Everything That Broke](#) — retrieval baselines, synthetic data failures
- [Week 2: Why My AI Companion Thought Fear Was Neutral](#) — affective memory, the neutral-vs-fear bug
- [Week 3 Part 1: Building the Evaluator](https://mohitvohraaa.substack.com/p/building-humanmemory-week-3part-1?r=2y4jgp) — independent ground truth, mixed emotion handling
- [Week 3 Part 2: Does Affective Memory Help?](https://mohitvohraaa.substack.com/p/building-humanmemory-week-3-does?r=2y4jgp) — A/B impact testing, conditional value finding
- [Week 3 Part 3: Building Semantic Memory](https://mohitvohraaa.substack.com/p/building-humanmemory-week-3part-3?r=2y4jgp) — fact extraction, contradiction detection

---

## Setup

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"
export GROQ_API_KEY="gsk_..."
```

---

## Testing

```bash
python -m pytest tests/ -v
```

28 tests: 13 unit, 3 integration, 12 SDK.

---

Actively built in public, day by day. Week 4 (procedural memory) in progress.
