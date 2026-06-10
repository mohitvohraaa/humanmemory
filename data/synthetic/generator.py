"""
data/synthetic/generator.py
────────────────────────────
Generates synthetic multi-session conversations with pre-labelled ground truth.

What it produces:
  - N sessions of M turns each
  - ground_truth dict: {query_id: [relevant_turn_ids]}
  - long_horizon_tests: same facts tested at session +10, +50, +100

Why synthetic:
  - We need ground truth to compute Recall@K automatically
  - We plant facts at generation time and record the answer key
  - Standard practice: Park et al. (2023), MemGPT both use synthetic data

Bugs fixed:
  - query_text now includes qid to avoid duplicate key overwrites
  - persona fact queries now saved to dataset.queries
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


# ─── Conversation Templates ──────────────────────────────────────────────────

TEMPLATES = [
    {
        "topic": "career",
        "emotion": "anxiety",
        "turns": [
            "I've been thinking about whether I should change jobs. It keeps me up at night.",
            "My manager told me I'm up for a promotion but I'm not sure I even want it.",
            "Got a call from a recruiter today. I don't know if I should take the interview.",
            "I feel stuck. Everyone else seems to know what they want to do.",
            "I've been thinking about moving into AI research but I don't know if I'm good enough.",
        ],
    },
    {
        "topic": "family",
        "emotion": "warmth",
        "turns": [
            "Talked to my mom this morning. She keeps asking when I'm coming home.",
            "My sister just had a baby. It's strange being an uncle now.",
            "Family WhatsApp group is going crazy with wedding planning again.",
            "Had a long call with my dad last night. We don't talk enough.",
            "My parents are getting older and I worry about them being far away.",
        ],
    },
    {
        "topic": "health",
        "emotion": "guilt",
        "turns": [
            "Went for a run today, first time in two weeks. Felt good actually.",
            "I've been trying to fix my sleep schedule. Not easy with late calls.",
            "Skipped the gym again. I keep telling myself I'll go tomorrow.",
            "Haven't been eating well lately. Too much ordering in.",
            "I think the stress is affecting my sleep. I wake up at 3am sometimes.",
        ],
    },
    {
        "topic": "work",
        "emotion": "stress",
        "turns": [
            "Had three back-to-back meetings today and still have a deadline tonight.",
            "My team is behind on the sprint and I have to tell the stakeholders.",
            "I actually shipped something today. First time in weeks I feel useful.",
            "The codebase is a mess and nobody wants to fix it.",
            "I got some really good feedback from a senior engineer today.",
        ],
    },
    {
        "topic": "relationships",
        "emotion": "longing",
        "turns": [
            "Long distance is harder than I thought it would be.",
            "We had a small argument over text. Hard to resolve things that way.",
            "Had a really good call last night. Felt close again.",
            "I've been feeling lonely even when I'm around people.",
            "I don't know how to tell them that I need more support right now.",
        ],
    },
]

PERSONA_FACTS = [
    ("identity",     "I work as a product manager at a fintech startup in Bangalore"),
    ("identity",     "I graduated from NSUT with a degree in computer science"),
    ("goal",         "I want to transition into AI research within the next year"),
    ("preference",   "I prefer direct advice over just being asked how I feel"),
    ("concern",      "I worry a lot about whether I am smart enough for the things I want"),
    ("relationship", "My closest friend is someone I met in college, we talk every week"),
]


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SyntheticTurn:
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    session_number: int = 0
    turn_number: int = 0
    speaker: str = "user"
    text: str = ""
    topic_tags: list[str] = field(default_factory=list)
    emotion_labels: list[str] = field(default_factory=list)
    valence_score: float = 0.0
    created_at: str = ""
    # Ground truth: this turn is the correct answer for these query IDs
    relevant_for_queries: list[str] = field(default_factory=list)


@dataclass
class SyntheticSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_number: int = 0
    turns: list[SyntheticTurn] = field(default_factory=list)
    created_at: str = ""


@dataclass
class SyntheticDataset:
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sessions: list[SyntheticSession] = field(default_factory=list)
    # {query_id: [turn_ids that are relevant]}
    ground_truth: dict[str, list[str]] = field(default_factory=dict)
    # {query_text: query_id} — so we can look up queries by text
    queries: dict[str, str] = field(default_factory=dict)
    # long horizon: {session_number: {query_id: [turn_ids]}}
    long_horizon_tests: dict[int, dict[str, list[str]]] = field(default_factory=dict)


# ─── Generator ───────────────────────────────────────────────────────────────

class SyntheticGenerator:
    """
    Generates synthetic conversations with pre-labelled ground truth.

    Usage:
        gen = SyntheticGenerator(seed=42)
        dataset = gen.generate(n_sessions=100, turns_per_session=20)
        gen.save(dataset, "data/synthetic/dataset_100.json")
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._query_counter = 0

    def _new_query_id(self) -> str:
        self._query_counter += 1
        return f"Q{self._query_counter:04d}"

    def generate(
        self,
        n_sessions: int = 100,
        turns_per_session: int = 20,
    ) -> SyntheticDataset:

        dataset = SyntheticDataset()
        start_date = datetime.utcnow() - timedelta(days=n_sessions)

        for s_idx in range(n_sessions):
            session_date = start_date + timedelta(days=s_idx)
            session = SyntheticSession(
                session_number=s_idx,
                created_at=session_date.isoformat(),
            )

            # Pick 2-3 random templates for this session
            selected_templates = random.sample(TEMPLATES, k=random.randint(2, 3))

            turn_idx = 0
            for template in selected_templates:
                if turn_idx >= turns_per_session:
                    break

                # Pick 1-2 turns from this template
                selected_turns = random.sample(
                    template["turns"],
                    k=random.randint(1, 2)
                )

                for turn_text in selected_turns:
                    if turn_idx >= turns_per_session:
                        break

                    turn_date = session_date + timedelta(minutes=turn_idx * 3)
                    turn = SyntheticTurn(
                        session_id=session.session_id,
                        session_number=s_idx,
                        turn_number=turn_idx,
                        text=turn_text,
                        topic_tags=[template["topic"]],
                        emotion_labels=[template["emotion"]],
                        # negative valence for anxiety/stress, positive for warmth
                        valence_score=random.uniform(-0.8, -0.2)
                        if template["emotion"] in ["anxiety", "stress", "guilt", "longing"]
                        else random.uniform(0.2, 0.8),
                        created_at=turn_date.isoformat(),
                    )
                    session.turns.append(turn)
                    turn_idx += 1

                    # Plant a ground-truth query for ~10% of turns
                    if random.random() < 0.10:
                        qid = self._new_query_id()
                        # Include qid in query_text to avoid duplicate key overwrites
                        query_text = f"What has the user said about {template['topic']}? [{qid}]"
                        dataset.ground_truth[qid] = [turn.turn_id]
                        dataset.queries[query_text] = qid
                        turn.relevant_for_queries.append(qid)

            # Every 10 sessions — plant a persona fact for long-horizon testing
            if s_idx % 10 == 0:
                category, fact_text = random.choice(PERSONA_FACTS)
                fact_turn = SyntheticTurn(
                    session_id=session.session_id,
                    session_number=s_idx,
                    turn_number=turn_idx,
                    text=f"By the way, {fact_text}.",
                    topic_tags=[category],
                    emotion_labels=["neutral"],
                    valence_score=0.0,
                    created_at=(
                        session_date + timedelta(minutes=turn_idx * 3)
                    ).isoformat(),
                )
                session.turns.append(fact_turn)

                # Create the query for this fact
                qid = self._new_query_id()
                query_text = f"What did the user say about their {category}? [{qid}]"
                dataset.ground_truth[qid] = [fact_turn.turn_id]
                dataset.queries[query_text] = qid
                fact_turn.relevant_for_queries.append(qid)

                # Register as long-horizon test at +10, +50, +100 sessions
                for gap in [10, 50, 100]:
                    target = s_idx + gap
                    if target < n_sessions:
                        if target not in dataset.long_horizon_tests:
                            dataset.long_horizon_tests[target] = {}
                        dataset.long_horizon_tests[target][qid] = [fact_turn.turn_id]

            dataset.sessions.append(session)

        return dataset

    def save(self, dataset: SyntheticDataset, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(dataset), f, indent=2, default=str)

        # Print a summary so we know what was generated
        total_turns = sum(len(s.turns) for s in dataset.sessions)
        long_horizon_total = sum(
            len(v) for v in dataset.long_horizon_tests.values()
        )
        print(f"Sessions:              {len(dataset.sessions)}")
        print(f"Total turns:           {total_turns}")
        print(f"Ground truth queries:  {len(dataset.ground_truth)}")
        print(f"Long-horizon queries:  {long_horizon_total}")
        print(f"Saved to:              {path}")


if __name__ == "__main__":
    gen = SyntheticGenerator(seed=42)
    dataset = gen.generate(n_sessions=100, turns_per_session=20)
    gen.save(dataset, "data/synthetic/dataset_100.json")
