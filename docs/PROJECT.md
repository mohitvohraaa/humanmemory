# HumanMemory — 5-Layer Cognitive Memory Architecture

A research project implementing a cognitive memory system for LLMs, inspired by
Tulving (1972), Park et al. (2023), and the CoALA framework. The system is
designed for **"Ira"** — an emotionally-aware conversational AI with persistent
memory across sessions.

---

## Quick Start

### Prerequisites

- Python 3.11.9 (see `.python-version`)
- pip

### Installation

```bash
# Clone and enter the repo
git clone <repo-url> && cd humanmemory

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Synthetic Dataset

```bash
python3 data/synthetic/generator.py
# Creates data/synthetic/dataset_100.json (100 sessions, ~2000 turns)
```

### Run Baseline Evaluation

```bash
python3 scripts/run_baseline.py
```

Expected output:

```
───────────────────────────────────────────────────────
  BASELINE RESULTS (naive last-N retrieval)
───────────────────────────────────────────────────────
  config               baseline_naive
  recall@1             0.0
  recall@5             0.0
  recall@10            0.0
  precision@5          0.0
  mrr                  0.0
  latency_p95_ms       0.2
───────────────────────────────────────────────────────
  Queries evaluated:   43
  Memories in store:   371
───────────────────────────────────────────────────────
```

**The zero-line is 0.0 across all metrics.** This is expected — naive "last 10
messages" retrieval fails because ground truth queries reference older memories.
Every future improvement must beat this baseline.

---

## Architecture Overview

The system implements a **5-layer cognitive memory architecture** modeled after
human memory systems:

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKING MEMORY (L1)                       │
│  Assembled context string passed to LLM at inference time   │
│  Token budget: 3000 total (500 current + 800 episodic +     │
│  600 semantic + 300 affective + 400 procedural + 400 sys)   │
└──────────────────────────┬──────────────────────────────────┘
                           │ assembles from ↓
┌──────────────────────────┼──────────────────────────────────┐
│                          │                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  EPISODIC   │  │  SEMANTIC   │  │  AFFECTIVE  │         │
│  │  MEMORY     │  │  MEMORY     │  │  MEMORY     │         │
│  │  (L2)       │  │  (L3)       │  │  (L4)       │         │
│  │             │  │             │  │             │         │
│  │  "What     │  │  "What is   │  │  "How does  │         │
│  │  happened"  │  │  true about │  │  the user   │         │
│  │             │  │  the user"  │  │  feel about │         │
│  │  SQLite +   │  │             │  │  topics"    │         │
│  │  ChromaDB   │  │  Facts      │  │             │         │
│  └─────────────┘  └─────────────┘  │  Topic →    │         │
│                                     │  Emotion    │         │
│  ┌─────────────┐                   │  mapping    │         │
│  │ PROCEDURAL  │                   └─────────────┘         │
│  │ MEMORY (L5) │                                            │
│  │             │                                            │
│  │ "How to    │                                            │
│  │  behave"   │                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Paper Grounding

| Layer | Paper | Key Concept |
|-------|-------|-------------|
| Episodic | Tulving (1972) | Instance-specific, temporally tagged memories |
| Episodic | Park et al. (2023) | Recency + importance + relevance composite scoring |
| Semantic | Tulving (1972) | Generalized facts without episodic context |
| Procedural | CoALA | Skills and rules that shape behavior |
| Working | CoALA | Agent's current state, assembled context |

### Novel Contribution

**The Affective Layer (L4)** is absent from all prior work:
- MemGPT — no emotional context
- Generative Agents — no topic-emotion mapping
- A-MEM — no affective component
- Mem0 — no emotional awareness
- Claude — no structured emotion tracking

This is what makes Ira emotionally aware — it tracks how the user feels about
specific topics over time.

---

## Data Models

All models are defined in a single file: `src/memory/models.py`. This is the
**single source of truth** — every other module imports from here.

### EpisodicMemory

A single past event with full context and scoring metadata.

```python
@dataclass
class EpisodicMemory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    turn_id: str = ""
    speaker: str = ""
    text: str = ""
    summary: str = ""

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None

    # Park et al. (2023) scoring
    importance_score: float = 0.0
    recency_score: float = 1.0
    relevance_score: float = 0.0

    # Our novel addition — emotional context
    valence_score: float = 0.0
    emotion_labels: list[str] = field(default_factory=list)

    # FK to AffectiveRecord — connects episodic to affective layer
    topic_tags: list[str] = field(default_factory=list)

    embedding_id: str = ""
```

#### Composite Retrieval Score

The retrieval scoring formula combines four signals (Park et al. 2023, extended
with emotional boost):

```python
@property
def composite_retrieval_score(self) -> float:
    """
    Park et al. (2023) retrieval formula.
    recency + importance + relevance, with emotional boost.
    """
    return (
        self.relevance_score  * 0.40 +  # semantic similarity
        self.recency_score    * 0.25 +  # time decay
        self.importance_score * 0.20 +  # significance
        abs(self.valence_score) * 0.15  # emotional intensity
    )
```

### SemanticFact

A generalized fact about the user, distilled from episodic clusters. Not tied
to any specific event — abstracted truth.

```python
@dataclass
class SemanticFact:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""

    category: FactCategory = FactCategory.IDENTITY
    fact_text: str = ""

    source_episode_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    is_stale: bool = False

    first_observed_at: datetime = field(default_factory=datetime.utcnow)
    last_confirmed_at: datetime = field(default_factory=datetime.utcnow)
```

Categories: `IDENTITY`, `PREFERENCE`, `CONCERN`, `RELATIONSHIP`, `GOAL`

### AffectiveRecord (Novel)

Topic → emotion association map for one user. **This is the novel layer absent
from all prior work.**

```python
@dataclass
class EmotionVector:
    """Emotion scores for a single topic."""
    joy: float = 0.0
    sadness: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    surprise: float = 0.0
    neutral: float = 1.0

    def dominant_emotion(self) -> str:
        scores = {
            "joy": self.joy, "sadness": self.sadness,
            "fear": self.fear, "anger": self.anger,
            "surprise": self.surprise, "neutral": self.neutral,
        }
        return max(scores, key=scores.get)

    def intensity(self) -> float:
        """How emotionally charged? 0 = flat, 1 = extreme."""
        return 1.0 - self.neutral


@dataclass
class AffectiveRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    topic: str = ""
    emotions: EmotionVector = field(default_factory=EmotionVector)
    mention_count: int = 0
    last_mentioned_at: Optional[datetime] = None

    def to_prompt_string(self) -> str:
        dominant = self.emotions.dominant_emotion()
        intensity = self.emotions.intensity()
        if intensity < 0.2:
            return ""
        level = "strongly" if intensity > 0.7 else "somewhat"
        return f"User feels {level} {dominant} about '{self.topic}'."
```

### ProceduralRule

Behavioral rules for how to interact with the user.

```python
@dataclass
class ProceduralRule:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    condition: str = ""
    action: str = ""
    rule_text: str = ""
    confidence: float = 1.0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_prompt_string(self) -> str:
        return f"- {self.rule_text}"
```

### WorkingContext

The assembled context string passed to the LLM. Built by the ContextAssembler
from all lower layers.

```python
@dataclass
class WorkingContext:
    session_id: str
    turn_id: str
    current_message: str

    episodic_context: str = ""
    semantic_context: str = ""
    affective_context: str = ""
    procedural_context: str = ""

    total_tokens_used: int = 0
    assembly_latency_ms: float = 0.0
    assembled_at: datetime = field(default_factory=datetime.utcnow)

    def to_prompt_string(self) -> str:
        parts = []
        if self.semantic_context:
            parts.append(f"[USER PROFILE]\n{self.semantic_context}")
        if self.affective_context:
            parts.append(f"[EMOTIONAL CONTEXT]\n{self.affective_context}")
        if self.procedural_context:
            parts.append(f"[INTERACTION RULES]\n{self.procedural_context}")
        if self.episodic_context:
            parts.append(f"[RELEVANT MEMORIES]\n{self.episodic_context}")
        return "\n\n".join(parts)
```

---

## Episodic Store

SQLite-backed persistence for episodic memories. File: `src/memory/episodic/store.py`

### Schema

```sql
CREATE TABLE IF NOT EXISTS episodic_memories (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    turn_id          TEXT,
    speaker          TEXT,
    text             TEXT NOT NULL,
    summary          TEXT,
    created_at       TEXT NOT NULL,
    importance_score REAL DEFAULT 0.0,
    recency_score    REAL DEFAULT 1.0,
    valence_score    REAL DEFAULT 0.0,
    emotion_labels   TEXT DEFAULT '[]',
    topic_tags       TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_created_at
ON episodic_memories (created_at DESC);
```

### EpisodicStore Class

```python
class EpisodicStore:
    def __init__(self, db_path: str = "data/processed/episodic.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def add(self, memory: EpisodicMemory) -> None:
        """Insert one memory. Skips if ID already exists."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO episodic_memories ...",
                (memory.id, memory.session_id, ...)
            )

    def naive_retrieve(self, limit: int = 10) -> list[EpisodicMemory]:
        """
        BASELINE-1: Return the most recent N memories.
        No ranking. No semantic search. Just recency.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodic_memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]
```

### Why SQLite?

- **Zero setup** — single file, runs on any machine
- **Structured queries** — "get all turns from session X"
- **Fast** — `ORDER BY created_at DESC` with index
- **ChromaDB handles vector search** — SQLite handles metadata

---

## Evaluation Framework

Written FIRST before any model code. File: `src/evaluation/harness.py`

### Metrics

| Metric | What It Measures |
|--------|------------------|
| Recall@K | What fraction of relevant memories did we find in top-K? |
| Precision@K | What fraction of top-K results were actually relevant? |
| MRR | Mean Reciprocal Rank — average 1/rank of first relevant result |
| Latency | p50, p95, p99 retrieval time in milliseconds |

### Core Functions

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of relevant memories did we find in the top-K?"""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of the top-K results were actually relevant?"""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant result."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0
```

### LatencyTimer

Context manager for timing retrieval calls:

```python
timer = LatencyTimer("naive_retrieve")
for query in queries:
    with timer:
        result = store.naive_retrieve(limit=10)
print(timer.p95())  # 95th percentile latency in ms
```

### Aggregate Metrics

```python
def aggregate_metrics(per_query_results: list[dict], config_name: str = "") -> RetrievalMetrics:
    """Averages per-query metrics into a single RetrievalMetrics."""
    # Computes mean of recall@K, precision@K, and MRR across all queries
    ...
```

---

## Synthetic Data Generator

File: `data/synthetic/generator.py`

### Conversation Templates

5 topics with associated emotions:

| Topic | Emotion | Example Turn |
|-------|---------|--------------|
| career | anxiety | "I've been thinking about whether I should change jobs" |
| family | warmth | "Talked to my mom this morning" |
| health | guilt | "Skipped the gym again" |
| work | stress | "Had three back-to-back meetings today" |
| relationships | longing | "Long distance is harder than I thought" |

### Persona Facts

Planted every 10 sessions for long-horizon testing:

- "I work as a product manager at a fintech startup in Bangalore"
- "I graduated from NSUT with a degree in computer science"
- "I want to transition into AI research within the next year"
- "I prefer direct advice over just being asked how I feel"
- "I worry a lot about whether I am smart enough"
- "My closest friend is someone I met in college"

### Dataset Structure

```json
{
  "sessions": [
    {
      "session_id": "session-001",
      "turns": [
        {
          "turn_id": "turn-001-001",
          "session_id": "session-001",
          "text": "...",
          "topic_tags": ["career"],
          "emotion_labels": ["anxiety"],
          "valence_score": -0.3,
          "created_at": "2024-01-01T10:00:00"
        }
      ]
    }
  ],
  "ground_truth": {
    "query-001": ["turn-001-001", "turn-002-003"]
  },
  "long_horizon_tests": {
    "persona-001": {
      "fact_text": "...",
      "target_turn_id": "...",
      "session_gaps": [10, 50, 100]
    }
  }
}
```

### Generation

- Deterministic (seed=42)
- 100 sessions, ~20 turns each
- ~10% of turns get ground-truth queries
- Every 10th session plants a persona fact

---

## Configuration

File: `configs/config.yaml`

### Key Sections

```yaml
project:
  name: humanmemory
  wandb_project: humanmemory

model:
  base_llm: microsoft/Phi-3-mini-4k-instruct
  embedding_model: all-mpnet-base-v2
  emotion_classifier: distilbert-base-uncased-emotion
  device: cpu
  dtype: float32

memory:
  working:
    max_tokens_total: 3000
    current_turn: 500
    episodic: 800
    semantic: 600
    affective: 300
    procedural: 400
    system_prompt: 400

  episodic:
    db_path: data/processed/episodic.db
    chroma_collection: episodic_memories
    max_memories_retrieved: 10
    importance_threshold: 0.3
    decay_half_life_days: 14

retrieval:
  weights:
    semantic: 0.40
    recency: 0.25
    importance: 0.20
    affective: 0.15
  top_k: 10
  rerank_top_n: 5

evaluation:
  recall_at_k: [1, 3, 5, 10]
  test_session_gaps: [1, 5, 10, 25, 50, 100]
  num_synthetic_sessions: 100
  turns_per_session: 20
  target_latency_ms: 80
```

---

## Baseline Results

The naive baseline scores **0.0 across all metrics**:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| recall@1 | 0.0 | Never finds relevant memory at rank 1 |
| recall@5 | 0.0 | Never finds relevant memory in top 5 |
| recall@10 | 0.0 | Never finds relevant memory in top 10 |
| precision@5 | 0.0 | No relevant memories in top 5 |
| mrr | 0.0 | First relevant result never appears |
| latency_p95 | 0.2ms | Very fast (but useless) |

### Why 0.0?

The dataset has 371 memories. Ground truth queries reference specific older
memories. Naive retrieval returns only the 10 most recent — which don't include
the relevant ones. This is the **zero-line** that semantic retrieval must beat.

---

## Git History

```
58b6775 feat: add episodic SQLite store with naive retrieval baseline
115eedc feat: add evaluation harness with Recall@K, Precision@K, RR and aggregate MRR
e9e6877 feat: add 5-layer memory data models
c99ac6e init: project folder structure and module init files
1c11f31 init: project foundation
```

Deliberate build order: foundation → folder structure → data models → evaluation
framework → episodic store + baseline script.

---

## Roadmap

### Completed

- [x] 5-layer data models (`src/memory/models.py`)
- [x] Episodic SQLite store (`src/memory/episodic/store.py`)
- [x] Evaluation harness (`src/evaluation/harness.py`)
- [x] Synthetic data generator (`data/synthetic/generator.py`)
- [x] Baseline runner (`scripts/run_baseline.py`)
- [x] Project configuration (`configs/config.yaml`)

### In Progress

- [ ] Semantic retrieval (ChromaDB + sentence-transformers)

### Planned

- [ ] Semantic memory layer — fact extraction, storage, contradiction detection
- [ ] Affective memory layer — emotion classification, topic-emotion mapping
- [ ] Procedural memory layer — rule extraction, condition-action pairs
- [ ] Working memory — context assembly with 3000-token budget
- [ ] Memory consolidation — episodic → semantic promotion
- [ ] API serving (FastAPI)
- [ ] Unit tests
- [ ] Integration tests

---

## Project Structure

```
humanmemory/
├── configs/
│   └── config.yaml              # Central configuration
├── data/
│   ├── processed/               # Runtime data (gitignored)
│   │   ├── episodic.db
│   │   └── test_episodic.db
│   └── synthetic/
│       ├── generator.py         # Synthetic data generator
│       └── dataset_100.json     # Generated dataset
├── docs/
│   └── PROJECT.md               # This file
├── scripts/
│   └── run_baseline.py          # Baseline evaluation runner
├── src/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── harness.py           # Recall@K, Precision@K, MRR, latency
│   └── memory/
│       ├── models.py            # ALL data models (single source of truth)
│       ├── episodic/
│       │   └── store.py         # SQLite episodic store
│       ├── affective/           # Planned
│       ├── procedural/          # Planned
│       ├── semantic/            # Planned
│       └── working/             # Planned
├── tests/
│   ├── unit/                    # Planned
│   └── integration/             # Planned
├── requirements.txt
└── .python-version              # 3.11.9
```

---

## Dependencies

| Category | Packages |
|----------|----------|
| Core ML | torch, transformers, sentence-transformers, datasets |
| Vector Store | chromadb |
| Database | sqlalchemy |
| API | fastapi, uvicorn |
| Evaluation | numpy, scipy, scikit-learn, wandb |
| NLP | keybert, nltk |
| Dev | pytest, python-dotenv, tqdm, loguru, pyyaml |
