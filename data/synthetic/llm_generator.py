"""
data/synthetic/llm_generator.py
─────────────────────────────────
Generates unique synthetic turns using Groq LLM.

Two steps:
  1. generate_sentence_pool() — calls Groq to create 100 unique turns per topic
  2. build_dataset()          — samples from pool to create sessions with ground truth

Why LLM-generated vs templates:
  - Templates: 25 sentences × 12 repetitions = identical vectors = retrieval fails
  - LLM-generated: every turn is unique prose = meaningful vectors = real evaluation
  - This is what Park et al. (2023) did in Generative Agents
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────

TOPICS = {
    "career": {
        "emotion": "anxiety",
        "prompt": """Generate 100 unique, realistic things a person might say to
their AI companion about career stress, job uncertainty, promotion anxiety,
career transitions, or work-life balance.

Rules:
- Each must be a single sentence or two, first person
- Each must be genuinely different from the others
- Sound like real casual conversation, not formal writing
- Mix different situations: job interviews, promotions, career changes,
  feeling stuck, comparing to peers, imposter syndrome
- Return as a JSON array of strings, nothing else
- No numbering, no bullet points, just the JSON array"""
    },
    "family": {
        "emotion": "warmth",
        "prompt": """Generate 100 unique, realistic things a person might say to
their AI companion about family — parents, siblings, extended family,
family obligations, missing home, family conflicts, or family celebrations.

Rules:
- Each must be a single sentence or two, first person
- Each must be genuinely different from the others
- Sound like real casual conversation
- Mix warmth, guilt, obligation, love, frustration
- Return as a JSON array of strings, nothing else"""
    },
    "health": {
        "emotion": "guilt",
        "prompt": """Generate 100 unique, realistic things a person might say to
their AI companion about health habits — exercise, sleep, eating, stress,
mental health, skipping workouts, or trying to build better habits.

Rules:
- Each must be a single sentence or two, first person
- Each must be genuinely different from the others
- Sound like real casual conversation
- Mix motivation, guilt, progress, setbacks
- Return as a JSON array of strings, nothing else"""
    },
    "work": {
        "emotion": "stress",
        "prompt": """Generate 100 unique, realistic things a person might say to
their AI companion about day-to-day work stress — deadlines, meetings,
colleagues, code reviews, shipping features, or feeling productive/unproductive.

Rules:
- Each must be a single sentence or two, first person
- Each must be genuinely different from the others
- Sound like real casual conversation
- Mix frustration, small wins, overwhelm, boredom
- Return as a JSON array of strings, nothing else"""
    },
    "relationships": {
        "emotion": "longing",
        "prompt": """Generate 100 unique, realistic things a person might say to
their AI companion about relationships — romantic partners, friendships,
loneliness, connection, arguments, missing people, or feeling understood.

Rules:
- Each must be a single sentence or two, first person
- Each must be genuinely different from the others
- Sound like real casual conversation
- Mix longing, joy, conflict, comfort, loneliness
- Return as a JSON array of strings, nothing else"""
    },
}

PERSONA_FACTS = [
    ("identity",     "I work as a product manager at a fintech startup in Bangalore"),
    ("identity",     "I graduated from NSUT with a degree in computer science"),
    ("goal",         "I want to transition into AI research within the next year"),
    ("preference",   "I prefer direct advice over just being asked how I feel"),
    ("concern",      "I worry a lot about whether I am smart enough for the things I want"),
    ("relationship", "My closest friend is someone I met in college, we talk every week"),
    ("identity",     "I am 24 years old and living alone for the first time"),
    ("goal",         "I want to run a half marathon before the end of the year"),
    ("concern",      "I sometimes feel like I am falling behind my peers"),
    ("preference",   "I find it easier to open up in writing than in conversation"),
]


# ─── Data Structures (same as before) ───────────────────────────────────────

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
    ground_truth: dict[str, list[str]] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    long_horizon_tests: dict[int, dict[str, list[str]]] = field(default_factory=dict)


# ─── Step 1: Generate sentence pool ─────────────────────────────────────────

def generate_sentence_pool(
    output_path: str = "data/synthetic/sentence_pool.json",
    sentences_per_topic: int = 100,
) -> dict[str, list[str]]:
    """
    Call Groq once per topic to generate unique sentences.
    Saves to sentence_pool.json so we don't regenerate every run.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    pool = {}

    for topic, config in TOPICS.items():
        print(f"Generating {sentences_per_topic} sentences for topic: {topic}...")

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates realistic synthetic conversation data. Always return valid JSON arrays only, with no additional text before or after."
                    },
                    {
                        "role": "user",
                        "content": config["prompt"]
                    }
                ],
                max_tokens=4000,
                temperature=0.9,
            )

            raw = response.choices[0].message.content.strip()

            # Parse JSON array
            # Sometimes LLM adds markdown code blocks — strip them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            sentences = json.loads(raw)

            # Validate — must be a list of strings
            if not isinstance(sentences, list):
                raise ValueError(f"Expected list, got {type(sentences)}")

            sentences = [s for s in sentences if isinstance(s, str) and len(s) > 10]
            pool[topic] = sentences[:sentences_per_topic]
            print(f"  Got {len(pool[topic])} unique sentences")

            # Small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  Error for topic {topic}: {e}")
            print("  Using fallback sentences")
            pool[topic] = [f"I have been thinking about {topic} a lot lately. [{i}]"
                          for i in range(sentences_per_topic)]

    # Save pool to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pool, f, indent=2)

    total = sum(len(v) for v in pool.values())
    print(f"\nSentence pool saved: {total} unique sentences → {output_path}")
    return pool


# ─── Step 2: Build dataset from pool ────────────────────────────────────────
def get_query_for_turn(pool: dict, topic: str, exclude_text: str) -> str:
    """
    Pick a different sentence from the same topic as the query.
    Mirrors real production use — user says something new,
    system searches for related past memories.
    """
    candidates = [s for s in pool[topic] if s != exclude_text]
    if not candidates:
        return exclude_text  # fallback if pool is tiny
    return random.choice(candidates)
def build_dataset(
    pool: dict[str, list[str]],
    n_sessions: int = 100,
    turns_per_session: int = 20,
    seed: int = 42,
) -> SyntheticDataset:
    """
    Build sessions by sampling from the sentence pool.
    Each sentence used at most once — no duplicates.
    """
    random.seed(seed)

    # Shuffle each topic's sentences
    shuffled = {topic: list(sentences) for topic, sentences in pool.items()}
    for topic in shuffled:
        random.shuffle(shuffled[topic])

    # Pointers — track how far into each topic's list we are
    pointers = {topic: 0 for topic in shuffled}

    def next_sentence(topic: str) -> str:
        """Get the next unused sentence for this topic."""
        idx = pointers[topic] % len(shuffled[topic])
        pointers[topic] += 1
        return shuffled[topic][idx]

    dataset = SyntheticDataset()
    start_date = datetime.utcnow() - timedelta(days=n_sessions)
    query_counter = 0

    def new_qid() -> str:
        nonlocal query_counter
        query_counter += 1
        return f"Q{query_counter:04d}"

    topics = list(TOPICS.keys())

    for s_idx in range(n_sessions):
        session_date = start_date + timedelta(days=s_idx)
        session = SyntheticSession(
            session_number=s_idx,
            created_at=session_date.isoformat(),
        )

        selected_topics = random.sample(topics, k=random.randint(2, 3))

        turn_idx = 0
        for topic in selected_topics:
            if turn_idx >= turns_per_session:
                break

            n_turns = random.randint(1, 3)
            for _ in range(n_turns):
                if turn_idx >= turns_per_session:
                    break

                turn_date = session_date + timedelta(minutes=turn_idx * 3)
                emotion = TOPICS[topic]["emotion"]

                turn = SyntheticTurn(
                    session_id=session.session_id,
                    session_number=s_idx,
                    turn_number=turn_idx,
                    text=next_sentence(topic),
                    topic_tags=[topic],
                    emotion_labels=[emotion],
                    valence_score=random.uniform(-0.8, -0.2)
                    if emotion in ["anxiety", "stress", "guilt", "longing"]
                    else random.uniform(0.2, 0.8),
                    created_at=turn_date.isoformat(),
                )
                session.turns.append(turn)
                turn_idx += 1

                # Plant ground truth query for ~10% of turns
                if random.random() < 0.10:
                    qid = new_qid()
                    #query_text = f"What has the user said about {topic}?"
                    query_text = get_query_for_turn(pool, topic, exclude_text=turn.text)
                    dataset.ground_truth[qid] = [turn.turn_id]
                    dataset.queries[qid] = query_text
                    turn.relevant_for_queries.append(qid)

        # Every 10 sessions — plant persona fact
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

            qid = new_qid()
            query_text = f"What did the user say about their {category}?"
            dataset.ground_truth[qid] = [fact_turn.turn_id]
            dataset.queries[qid] = query_text
            fact_turn.relevant_for_queries.append(qid)

            for gap in [10, 50, 100]:
                target = s_idx + gap
                if target < n_sessions:
                    if target not in dataset.long_horizon_tests:
                        dataset.long_horizon_tests[target] = {}
                    dataset.long_horizon_tests[target][qid] = [fact_turn.turn_id]

        dataset.sessions.append(session)

    return dataset


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pool_path = "data/synthetic/sentence_pool.json"

    # Step 1 — generate or load sentence pool
    if Path(pool_path).exists():
        print(f"Loading existing sentence pool from {pool_path}")
        with open(pool_path) as f:
            pool = json.load(f)
        total = sum(len(v) for v in pool.values())
        print(f"Loaded {total} sentences across {len(pool)} topics")
    else:
        print("Generating sentence pool via Groq API...")
        pool = generate_sentence_pool(pool_path)

    # Check for duplicates
    all_sentences = []
    for sentences in pool.values():
        all_sentences.extend(sentences)
    unique = len(set(all_sentences))
    print(f"\nTotal sentences: {len(all_sentences)}, Unique: {unique}")

    # Step 2 — build dataset
    print("\nBuilding dataset from pool...")
    dataset = build_dataset(pool, n_sessions=100, turns_per_session=20)

    # Save
    output_path = "data/synthetic/dataset_100.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(dataset), f, indent=2, default=str)

    total_turns = sum(len(s.turns) for s in dataset.sessions)
    print(f"\nSessions:             {len(dataset.sessions)}")
    print(f"Total turns:          {total_turns}")
    print(f"Ground truth queries: {len(dataset.ground_truth)}")
    print(f"Queries with text:    {len(dataset.queries)}")
    print(f"Queries missing text: {len(set(dataset.ground_truth.keys()) - set(dataset.queries.keys()))}")
    print(f"Saved to:             {output_path}")
