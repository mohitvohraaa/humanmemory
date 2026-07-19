"""
src/memory/working/context_assembler.py
─────────────────────────────────────────
Layer 1 — Working Memory.

The single function that gets called per user query.
Pulls from episodic (Layer 2) and affective (Layer 4) memory,
assembles the final context string injected into the LLM prompt.

This is intentionally minimal — semantic (Layer 3) and procedural
(Layer 5) integration come later. Today we wire exactly what we've
built so far: episodic retrieval + reranking + affective context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.memory.affective.affective_store import AffectiveStore
from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.semantic.semantic_store import SemanticStore
from src.retrieval.query_classifier import classify, get_weights
from src.retrieval.reranker import rerank


@dataclass
class AssembledContext:
    """Result of assembling working memory context for one query."""
    query_text: str
    episodic_context: str = ""
    semantic_context: str = ""
    affective_context: str = ""
    retrieved_memory_ids: list[str] = field(default_factory=list)
    query_type: str = ""

    def to_prompt_string(self) -> str:
        """
        Final string injected into the LLM system prompt.
        Order matters: profile facts first (stable), then emotional
        state (dynamic), then specific memories (most specific).
        """
        parts = []
        if self.semantic_context:
            parts.append(self.semantic_context)
        if self.affective_context:
            parts.append(self.affective_context)
        if self.episodic_context:
            parts.append("[RELEVANT MEMORIES]\n" + self.episodic_context)
        return "\n\n".join(parts)


def assemble_context(
    query_text: str,
    episodic_store: SemanticEpisodicStore,
    affective_store: AffectiveStore,
    semantic_store: SemanticStore | None = None,
    topics: list[str] | None = None,
    top_k: int = 5,
) -> AssembledContext:
    """
    Build the full working memory context for one query.

    Args:
        query_text:      the user's current message
        episodic_store:  ChromaDB-backed episodic store
        affective_store: per-user affective store
        topics:          topics relevant to this query (from TopicTagger,
                         passed in rather than computed here to avoid
                         redundant Groq calls in the hot path)
        top_k:           how many reranked memories to include

    Returns:
        AssembledContext with both episodic and affective strings filled in
    """
    # Step 1 — classify query type (recent / long_term / specific)
    query_type = classify(query_text)
    weights = get_weights(query_type)

    # Step 2 — retrieve candidates from episodic memory
    candidates = episodic_store.semantic_retrieve(query_text, limit=20)

    # Step 3 — rerank using query-adaptive weights
    reranked = rerank(candidates, weights=weights)
    top_memories = reranked[:top_k]

    episodic_lines = [m["text"] for m in top_memories]
    episodic_context = "\n".join(f"- {line}" for line in episodic_lines)

    # Step 4 — pull affective context for relevant topics
    affective_context = affective_store.to_context()

    # Step 5 — pull semantic facts ([USER PROFILE])
    semantic_context = semantic_store.to_context() if semantic_store else ""

    return AssembledContext(
        query_text=query_text,
        episodic_context=episodic_context,
        semantic_context=semantic_context,
        affective_context=affective_context,
        retrieved_memory_ids=[m["id"] for m in top_memories],
        query_type=query_type.value,
    )
