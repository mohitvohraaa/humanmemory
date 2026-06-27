"""
src/memory/semantic/consolidation.py
───────────────────────────────────────
Layer 3 — Semantic Memory consolidation.

Turns clusters of episodic memories into durable, generalized facts.
Runs at session end, not per-turn (unlike affective memory).
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

import json

SYSTEM_PROMPT = """You are extracting durable facts about a user from
their conversation history with a companion AI.

Given several related statements from the user, extract ONE concise,
general fact that captures the pattern — written in third person,
present tense, as a standalone statement about the user.

Also classify the fact into exactly one category:
  identity      - who the user is, their role, life situation
  preference    - how they like to communicate, what they enjoy/avoid
  concern       - recurring anxieties, stressors, worries
  relationship  - people they mention and the nature of that connection
  goal          - something they are working toward

Rules:
- Only extract a fact if there is a genuine PATTERN across multiple
  statements, not a one-off event
- If the statements do not reveal a clear pattern, respond with
  exactly: NONE
- Keep the fact under 15 words
- Use third person ("User prefers..." not "I prefer...")
- Respond ONLY with valid JSON, no other text:
  {"fact": "...", "category": "..."}
  or
  {"fact": null, "category": null}

Examples:
  Input: "I enjoy morning runs." / "Went jogging before work." /
         "Did my run again before heading to the office."
  Output: {"fact": "User prefers exercising in the morning.", "category": "preference"}

  Input: "I am thinking about a career change but scared of failing." /
         "I feel stuck, everyone else knows what they want to do."
  Output: {"fact": "User feels uncertain about their career direction.", "category": "concern"}

  Input: "Had pizza for dinner." / "The weather was nice today."
  Output: {"fact": null, "category": null}
"""


def extract_fact(turns: list[str]) -> dict | None:
    """
    Given a cluster of related episodic turns, extract a single
    durable fact AND its category in one LLM call.

    Returns:
        {"fact": str, "category": str} or None if no clear pattern
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    turns_text = "\n".join(f"- {t}" for t in turns)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": turns_text},
        ],
        max_tokens=80,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not parsed.get("fact"):
        return None

    return {"fact": parsed["fact"], "category": parsed["category"]}
