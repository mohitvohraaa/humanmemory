"""
humanmemory
───────────
5-layer biologically-inspired memory service for LLM applications.

Usage:
    from humanmemory import MemoryClient

    mem = MemoryClient(user_id="alice")
    context = mem.get_context("How's my mom doing?")
    # ... call your own LLM with context ...
    mem.store(message="How's my mom doing?", response="She's doing well!")
    mem.consolidate()
"""

from .llm import create_groq_client
from .system import MemoryClient

__all__ = ["MemoryClient", "create_groq_client"]
