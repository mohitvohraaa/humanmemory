"""
tests/unit/test_context_assembler.py
─────────────────────────────────────
Unit tests for assemble_context() — Layer 1 Working Memory.

All external dependencies (ChromaDB, Groq) are mocked.
Tests verify pure logic: prompt assembly, ordering, top_k, query types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from src.memory.working.context_assembler import (
    AssembledContext,
    assemble_context,
)
from src.retrieval.query_classifier import QueryType


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_episodic_store(memories: list[dict] | None = None):
    """Create a mock SemanticEpisodicStore with controlled retrieval results."""
    store = MagicMock()
    store.semantic_retrieve.return_value = memories or []
    return store


def _make_affective_store(context_str: str = ""):
    """Create a mock AffectiveStore."""
    store = MagicMock()
    store.to_context.return_value = context_str
    return store


def _make_semantic_store(context_str: str = ""):
    """Create a mock SemanticStore."""
    store = MagicMock()
    store.to_context.return_value = context_str
    return store


def _make_memory(text: str, score: float = 0.8, created_at: str = "2026-07-19T10:00:00"):
    """Create a candidate dict matching SemanticEpisodicStore.semantic_retrieve() output."""
    return {
        "id": f"mem_{hash(text) % 10000}",
        "text": text,
        "score": score,
        "metadata": {
            "session_id": "s1",
            "topic_tags": "career",
            "valence_score": -0.3,
            "importance_score": 0.6,
            "created_at": created_at,
        },
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestEmptyStores:
    """All stores return empty → no crash, empty context strings."""

    @patch("src.memory.working.context_assembler.classify")
    def test_empty_stores(self, mock_classify):
        mock_classify.return_value = QueryType.RECENT

        result = assemble_context(
            query_text="hello",
            episodic_store=_make_episodic_store([]),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
        )

        assert result.episodic_context == ""
        assert result.affective_context == ""
        assert result.semantic_context == ""
        assert result.retrieved_memory_ids == []
        assert result.query_type == "recent"

    @patch("src.memory.working.context_assembler.classify")
    def test_empty_stores_no_semantic_store(self, mock_classify):
        mock_classify.return_value = QueryType.RECENT

        result = assemble_context(
            query_text="hello",
            episodic_store=_make_episodic_store([]),
            affective_store=_make_affective_store(""),
            semantic_store=None,
        )

        assert result.semantic_context == ""


class TestEpisodicOnly:
    """Only episodic store has data → [RELEVANT MEMORIES] section populated."""

    @patch("src.memory.working.context_assembler.classify")
    def test_episodic_only(self, mock_classify):
        mock_classify.return_value = QueryType.RECENT

        memories = [
            _make_memory("I am worried about my job interview tomorrow"),
            _make_memory("Had a stressful meeting with my manager"),
        ]

        result = assemble_context(
            query_text="how is work going",
            episodic_store=_make_episodic_store(memories),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
        )

        assert "job interview" in result.episodic_context
        assert "stressful meeting" in result.episodic_context
        assert len(result.retrieved_memory_ids) == 2

        # [RELEVANT MEMORIES] header is added by to_prompt_string()
        prompt = result.to_prompt_string()
        assert "[RELEVANT MEMORIES]" in prompt


class TestAffectiveOnly:
    """Only affective store has data → [EMOTIONAL CONTEXT] section populated."""

    @patch("src.memory.working.context_assembler.classify")
    def test_affective_only(self, mock_classify):
        mock_classify.return_value = QueryType.LONG_TERM

        affective_ctx = "[EMOTIONAL CONTEXT]\nUser feels strongly fearful about 'career'."

        result = assemble_context(
            query_text="how have I been feeling about work",
            episodic_store=_make_episodic_store([]),
            affective_store=_make_affective_store(affective_ctx),
            semantic_store=_make_semantic_store(""),
        )

        assert result.affective_context == affective_ctx
        assert result.episodic_context == ""


class TestSemanticOnly:
    """Only semantic store has data → [USER PROFILE] section populated."""

    @patch("src.memory.working.context_assembler.classify")
    def test_semantic_only(self, mock_classify):
        mock_classify.return_value = QueryType.SPECIFIC

        semantic_ctx = "[USER PROFILE]\n- User prefers working from home\n- User is introverted"

        result = assemble_context(
            query_text="what do I prefer",
            episodic_store=_make_episodic_store([]),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(semantic_ctx),
        )

        assert result.semantic_context == semantic_ctx


class TestFullAssembly:
    """All three stores populated → prompt has all three sections in correct order."""

    @patch("src.memory.working.context_assembler.classify")
    def test_full_assembly(self, mock_classify):
        mock_classify.return_value = QueryType.LONG_TERM

        memories = [
            _make_memory("I was excited about the new project"),
            _make_memory("My team celebrated a win today"),
        ]
        affective_ctx = "[EMOTIONAL CONTEXT]\nUser feels somewhat joyful about 'career'."
        semantic_ctx = "[USER PROFILE]\n- User is a software engineer"

        result = assemble_context(
            query_text="tell me about my career",
            episodic_store=_make_episodic_store(memories),
            affective_store=_make_affective_store(affective_ctx),
            semantic_store=_make_semantic_store(semantic_ctx),
            top_k=2,
        )

        # All sections present
        assert result.episodic_context != ""
        assert result.affective_context != ""
        assert result.semantic_context != ""

        # Verify prompt string
        prompt = result.to_prompt_string()
        assert "[USER PROFILE]" in prompt
        assert "[EMOTIONAL CONTEXT]" in prompt
        assert "[RELEVANT MEMORIES]" in prompt

        # Order: semantic → affective → episodic
        profile_pos = prompt.index("[USER PROFILE]")
        emotion_pos = prompt.index("[EMOTIONAL CONTEXT]")
        memory_pos = prompt.index("[RELEVANT MEMORIES]")
        assert profile_pos < emotion_pos < memory_pos


class TestToPromptStringOrder:
    """Verify the exact order in to_prompt_string()."""

    @patch("src.memory.working.context_assembler.classify")
    def test_order_semantic_before_affective_before_episodic(self, mock_classify):
        mock_classify.return_value = QueryType.LONG_TERM

        result = AssembledContext(
            query_text="test",
            semantic_context="[USER PROFILE]\n- fact1",
            affective_context="[EMOTIONAL CONTEXT]\n- emotion1",
            episodic_context="- memory1\n- memory2",
        )

        prompt = result.to_prompt_string()
        lines = prompt.split("\n\n")

        # First section: semantic
        assert lines[0].startswith("[USER PROFILE]")
        # Second section: affective
        assert lines[1].startswith("[EMOTIONAL CONTEXT]")
        # Third section: episodic
        assert lines[2].startswith("[RELEVANT MEMORIES]")

    @patch("src.memory.working.context_assembler.classify")
    def test_skip_empty_sections(self, mock_classify):
        mock_classify.return_value = QueryType.RECENT

        result = AssembledContext(
            query_text="test",
            semantic_context="",
            affective_context="[EMOTIONAL CONTEXT]\n- sad",
            episodic_context="",
        )

        prompt = result.to_prompt_string()
        # Only affective section present
        assert prompt == "[EMOTIONAL CONTEXT]\n- sad"


class TestQueryTypePassthrough:
    """Mock classifier returns a QueryType → verify it's captured in result."""

    @patch("src.memory.working.context_assembler.classify")
    def test_specific_query_type(self, mock_classify):
        mock_classify.return_value = QueryType.SPECIFIC

        result = assemble_context(
            query_text="what did I decide about the job",
            episodic_store=_make_episodic_store([]),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
        )

        assert result.query_type == "specific"

    @patch("src.memory.working.context_assembler.classify")
    def test_long_term_query_type(self, mock_classify):
        mock_classify.return_value = QueryType.LONG_TERM

        result = assemble_context(
            query_text="how have I been feeling generally",
            episodic_store=_make_episodic_store([]),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
        )

        assert result.query_type == "long_term"


class TestTopKLimitsResults:
    """Insert 10 memories, set top_k=3 → only 3 returned."""

    @patch("src.memory.working.context_assembler.classify")
    def test_top_k_limits(self, mock_classify):
        mock_classify.return_value = QueryType.RECENT

        memories = [_make_memory(f"memory {i}", score=0.9 - i * 0.05) for i in range(10)]

        result = assemble_context(
            query_text="recent things",
            episodic_store=_make_episodic_store(memories),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
            top_k=3,
        )

        assert len(result.retrieved_memory_ids) == 3
        # Only 3 lines in episodic context
        lines = [l for l in result.episodic_context.split("\n") if l.strip()]
        assert len(lines) == 3


class TestRerankerWeightsApplied:
    """Different query types produce different reranking via weights."""

    @patch("src.memory.working.context_assembler.classify")
    def test_long_term_low_recency_weight(self, mock_classify):
        """LONG_TERM query → recency weight=0.05, so old memories still score well."""
        mock_classify.return_value = QueryType.LONG_TERM

        # Old memory with high semantic match
        old_memory = _make_memory("deep reflection on career", score=0.9,
                                  created_at="2026-01-01T00:00:00")
        # Recent memory with low semantic match
        recent_memory = _make_memory("quick chat about weather", score=0.3,
                                     created_at="2026-07-19T10:00:00")

        result = assemble_context(
            query_text="how do I feel about my career over time",
            episodic_store=_make_episodic_store([recent_memory, old_memory]),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
            top_k=2,
        )

        # Old memory with high semantic should rank first for LONG_TERM
        # (recency weight is only 0.05, semantic is 0.50)
        assert result.retrieved_memory_ids[0] == old_memory["id"]

    @patch("src.memory.working.context_assembler.classify")
    def test_recent_high_recency_weight(self, mock_classify):
        """RECENT query → recency weight=0.35, so recent memories rank higher."""
        mock_classify.return_value = QueryType.RECENT

        # Old memory with high semantic match
        old_memory = _make_memory("deep reflection on career", score=0.9,
                                  created_at="2026-01-01T00:00:00")
        # Recent memory with low semantic match
        recent_memory = _make_memory("quick chat about weather", score=0.3,
                                     created_at="2026-07-19T10:00:00")

        result = assemble_context(
            query_text="what happened today",
            episodic_store=_make_episodic_store([old_memory, recent_memory]),
            affective_store=_make_affective_store(""),
            semantic_store=_make_semantic_store(""),
            top_k=2,
        )

        # Recent memory should rank first for RECENT query
        assert result.retrieved_memory_ids[0] == recent_memory["id"]
