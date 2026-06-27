"""
scripts/affective_impact_test.py
─────────────────────────────────
A/B test: does affective context improve LLM companion responses?
Compares episodic-only vs episodic + affective context.
"""

import os
import sys
import random

from groq import Groq

sys.path.insert(0, ".")

from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.affective.affective_store import AffectiveStore
from src.memory.working.context_assembler import assemble_context

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def get_llm_response(context: str, query_text: str) -> str:
    """Generate a companion response with memory context."""
    system_msg = "You are a supportive AI companion."
    if context:
        system_msg += f"\n\nUse this context about the user:\n{context}"

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query_text},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def judge_responses(query_text: str, response_a: str, response_b: str) -> dict:
    """
    LLM judge — randomizes order to reduce position bias.
    Returns verdict + reasoning. Treated as a qualitative signal,
    not a validated metric — no ground truth exists for "better
    emotional awareness," so this supplements manual inspection
    rather than replacing it.
    """
    flip = random.random() < 0.5
    first, second = (response_a, response_b) if not flip else (response_b, response_a)
    first_label, second_label = ("A", "B") if not flip else ("B", "A")

    judge_prompt = f"""A user said: "{query_text}"

Two AI companion responses were generated:

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

    # Map back to actual A/B since we randomized order
    # startswith, not "in" — avoids substring bug where "Response 1"
    # can appear inside reasoning text even when verdict is "Response 2"
    winner = first_label if verdict_raw.startswith("Response 1") else second_label

    return {
        "winner": winner,  # "A" or "B"
        "reasoning": verdict_raw,
        "order_flipped": flip,
    }


def run_comparison(query_text: str, episodic_store, affective_store):
    result_a = assemble_context(query_text, episodic_store, affective_store)
    context_a = "[RELEVANT MEMORIES]\n" + result_a.episodic_context
    context_b = result_a.to_prompt_string()

    response_a = get_llm_response(context_a, query_text)
    response_b = get_llm_response(context_b, query_text)

    print("=" * 70)
    print(f"QUERY: {query_text}")
    print("=" * 70)
    print("\n--- Response A (episodic only) ---")
    print(response_a)
    print("\n--- Response B (episodic + affective) ---")
    print(response_b)

    verdict = judge_responses(query_text, response_a, response_b)
    print("\n--- Judge verdict ---")
    print(f"Winner: Response {verdict['winner']}")
    print(f"Reasoning: {verdict['reasoning']}")
    print()
    return verdict


if __name__ == "__main__":
    episodic_store = SemanticEpisodicStore(
        persist_dir='data/processed/chroma',
        collection_name='episodic_memories',
    )
    affective_store = AffectiveStore(user_id='consistency_test')

    queries = [
        "I've been thinking about my career direction lately",
        "Should I take the new job offer?",
        "How do you think I am doing overall?",
    ]

    wins = {"A": 0, "B": 0}
    for q in queries:
        verdict = run_comparison(q, episodic_store, affective_store)
        wins[verdict["winner"]] += 1

    print("=" * 70)
    print(f"FINAL TALLY across {len(queries)} queries: {wins}")
    print("(A = episodic only, B = episodic + affective)")
