"""
scripts/affective_impact_test_v2.py
─────────────────────────────────────
Fairer test: uses a deliberately FLAT/factual episodic memory,
paired with strong affective history the episodic retrieval
would NOT surface on its own. Tests whether affective context
adds value when episodic context alone is emotionally neutral.
"""

import sys
sys.path.insert(0, '.')

import os
import random
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_llm_response(system_context: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a companion AI. Use this context if relevant:\n\n{system_context}"},
            {"role": "user", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content


def judge_responses(query_text: str, response_a: str, response_b: str) -> dict:
    flip = random.random() < 0.5
    first, second = (response_a, response_b) if not flip else (response_b, response_a)
    first_label, second_label = ("A", "B") if not flip else ("B", "A")

    judge_prompt = f"""A user said: "{query_text}"

Response 1: {first}

Response 2: {second}

Which response demonstrates better awareness of the user's likely
emotional state and history with this topic? Answer with just
"Response 1" or "Response 2", followed by one sentence explaining why."""

    judge_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=80,
        temperature=0.0,
    )
    verdict_raw = judge_response.choices[0].message.content.strip()
    winner = first_label if verdict_raw.startswith("Response 1") else second_label

    return {"winner": winner, "reasoning": verdict_raw}


def run_fair_comparison(query_text: str, flat_episodic_memory: str, affective_context: str):
    # Version A: episodic only, but DELIBERATELY FLAT/FACTUAL
    context_a = f"[RELEVANT MEMORIES]\n- {flat_episodic_memory}"

    # Version B: same flat episodic memory + affective context
    context_b = f"{affective_context}\n\n[RELEVANT MEMORIES]\n- {flat_episodic_memory}"

    response_a = get_llm_response(context_a, query_text)
    response_b = get_llm_response(context_b, query_text)

    print("=" * 70)
    print(f"QUERY: {query_text}")
    print(f"FLAT EPISODIC MEMORY (no emotional language): {flat_episodic_memory}")
    print("=" * 70)
    print("\n--- Response A (flat episodic only) ---")
    print(response_a)
    print("\n--- Response B (flat episodic + affective context) ---")
    print(response_b)

    verdict = judge_responses(query_text, response_a, response_b)
    print("\n--- Judge verdict ---")
    print(f"Winner: Response {verdict['winner']}")
    print(f"Reasoning: {verdict['reasoning']}")
    print()
    return verdict


if __name__ == "__main__":
    test_cases = [
        {
            "query": "What should I focus on this week regarding my career?",
            "flat_memory": "I have a meeting about the promotion next Tuesday.",
            "affective": "[EMOTIONAL CONTEXT]\nUser feels strongly fearful about 'career'.",
        },
        {
            "query": "Any thoughts on how my week at work went?",
            "flat_memory": "I finished the quarterly report on Thursday.",
            "affective": "[EMOTIONAL CONTEXT]\nUser feels strongly guilty about 'work'.",
        },
        {
            "query": "I'm seeing my family this weekend, anything I should keep in mind?",
            "flat_memory": "My parents are visiting on Saturday.",
            "affective": "[EMOTIONAL CONTEXT]\nUser feels somewhat fearful about 'family'.",
        },
    ]

    wins = {"A": 0, "B": 0}
    for case in test_cases:
        verdict = run_fair_comparison(case["query"], case["flat_memory"], case["affective"])
        wins[verdict["winner"]] += 1

    print("=" * 70)
    print(f"FINAL TALLY across {len(test_cases)} queries: {wins}")
    print("(A = flat episodic only, B = flat episodic + affective)")
