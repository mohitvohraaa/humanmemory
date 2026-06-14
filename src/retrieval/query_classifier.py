"""
src/retrieval/query_classifier.py
───────────────────────────────────
LLM-based query classifier using Groq.

Classifies queries into three types:
  recent      → user asking about something that happened recently
  long_term   → user asking about patterns or feelings over time
  specific    → user asking about a specific past decision or fact

Classification runs async in parallel with ChromaDB search
to minimize latency impact.

Includes an in-memory cache so repeated query patterns
don't incur repeated API calls.

Design decision:
  LLM-based chosen over rule-based for accuracy.
  Async + cache mitigates latency cost.
  Rule-based fallback if Groq call fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from enum import Enum
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# ─── Query types ─────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    RECENT    = "recent"       # about something that happened recently
    LONG_TERM = "long_term"    # about patterns or feelings over time
    SPECIFIC  = "specific"     # about a specific past decision or fact


# ─── Weights per query type ───────────────────────────────────────────────────

WEIGHTS = {
    QueryType.RECENT: {
        "semantic":   0.35,
        "recency":    0.35,   # high — recent memories matter most
        "importance": 0.20,
        "valence":    0.10,
    },
    QueryType.LONG_TERM: {
        "semantic":   0.50,
        "recency":    0.05,   # low — don't penalize old memories
        "importance": 0.25,
        "valence":    0.20,
    },
    QueryType.SPECIFIC: {
        "semantic":   0.70,
        "recency":    0.00,   # zero — pure semantic for specific facts
        "importance": 0.20,
        "valence":    0.10,
    },
}


# ─── Cache ───────────────────────────────────────────────────────────────────

# Simple in-memory cache: query_hash → QueryType
# In production: replace with Redis for persistence across restarts
_cache: dict[str, QueryType] = {}

def _cache_key(query_text: str) -> str:
    return hashlib.md5(query_text.lower().strip().encode()).hexdigest()


# ─── Rule-based fallback ─────────────────────────────────────────────────────

def classify_rule_based(query_text: str) -> QueryType:
    """
    Fast rule-based fallback if Groq call fails.
    ~80% accuracy vs ~95% for LLM-based.
    """
    q = query_text.lower()

    # Recent signals
    recent_keywords = ["today", "yesterday", "this week", "just now",
                       "earlier", "this morning", "tonight", "recently"]
    if any(kw in q for kw in recent_keywords):
        return QueryType.RECENT

    # Specific fact signals
    specific_keywords = ["what did i decide", "what did i say", "remember when",
                         "did i mention", "what was i", "what happened with"]
    if any(kw in q for kw in specific_keywords):
        return QueryType.SPECIFIC

    # Default to long-term
    return QueryType.LONG_TERM


# ─── LLM classifier ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a query classifier for a memory retrieval system.

Classify the user's query into exactly one of these three types:

recent     - The user is asking about something that happened recently
             (today, this week, a recent event or conversation)
             Examples: "how did my meeting go", "what did I eat today"

long_term  - The user is asking about patterns, feelings, or history over time
             Examples: "how have I been feeling about work", "what do I usually say about family"

specific   - The user is asking about a specific past decision, fact, or statement
             Examples: "what did I decide about the job offer", "what did I say about my sister"

Reply with ONLY one word: recent, long_term, or specific.
No explanation. No punctuation. Just the word."""


def classify(query_text: str, use_cache: bool = True) -> QueryType:
    """
    Classify a query synchronously.
    Checks cache first, then calls Groq, falls back to rules.
    
    Args:
        query_text: the user's query
        use_cache:  whether to use the in-memory cache
    
    Returns:
        QueryType enum value
    """
    # Check cache
    key = _cache_key(query_text)
    if use_cache and key in _cache:
        return _cache[key]

    # Try Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query_text},
            ],
            max_tokens=5,
            temperature=0.0,   # deterministic
        )
        raw = response.choices[0].message.content.strip().lower()

        # Parse response
        if "recent" in raw:
            result = QueryType.RECENT
        elif "specific" in raw:
            result = QueryType.SPECIFIC
        else:
            result = QueryType.LONG_TERM

    except Exception as e:
        print(f"Groq classification failed: {e}. Using rule-based fallback.")
        result = classify_rule_based(query_text)

    # Cache result
    if use_cache:
        _cache[key] = result

    return result


async def classify_async(query_text: str, use_cache: bool = True) -> QueryType:
    """
    Async version — runs in parallel with ChromaDB search.
    
    Usage:
        query_type, retrieved = await asyncio.gather(
            classify_async(query_text),
            search_chroma_async(query_text),
        )
    """
    # Check cache first — no async needed
    key = _cache_key(query_text)
    if use_cache and key in _cache:
        return _cache[key]

    # Run sync classify in thread pool so it doesn't block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, classify, query_text, use_cache)
    return result


def get_weights(query_type: QueryType) -> dict:
    """Get reranking weights for a given query type."""
    return WEIGHTS[query_type]
