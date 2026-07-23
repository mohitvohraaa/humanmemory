"""
humanmemory/llm.py
──────────────────
Groq client factory for internal memory operations.

The memory service uses Groq for:
  - Query classification (recent / long_term / specific)
  - Topic tagging (career, family, health, work, relationships)
  - Fact extraction (durable fact from episodic clusters)

Generation is handled by the user's own LLM — we never touch it.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def create_groq_client(api_key: str | None = None):
    """
    Create a Groq client for internal memory operations.

    Args:
        api_key: Groq API key. Falls back to GROQ_API_KEY env var.

    Returns:
        groq.Groq client instance.
    """
    from groq import Groq
    return Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
