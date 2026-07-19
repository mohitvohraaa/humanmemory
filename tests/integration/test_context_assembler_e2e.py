"""
tests/integration/test_context_assembler_e2e.py
───────────────────────────────────────────────
End-to-end integration test for assemble_context().

Real ChromaDB (in tmp_path), real AffectiveStore (in tmp_path).
Only Groq API calls are mocked (classifier + semantic fact check).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.affective.affective_store import AffectiveStore
from src.memory.models import EpisodicMemory, EmotionVector, AffectiveRecord
from src.memory.working.context_assembler import assemble_context
from src.retrieval.query_classifier import QueryType


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def chroma_store(tmp_path):
    """Real SemanticEpisodicStore backed by temp ChromaDB."""
    store = SemanticEpisodicStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_episodic",
    )
    yield store
    store.clear()


@pytest.fixture
def affective_store(tmp_path):
    """Real AffectiveStore backed by temp JSON."""
    store = AffectiveStore(
        user_id="test_user",
        storage_dir=str(tmp_path / "affective"),
    )
    yield store
    store.clear()


@pytest.fixture
def sample_memories():
    """Pre-built EpisodicMemory objects for testing."""
    now = datetime(2026, 7, 19, 10, 0, 0, tzinfo=timezone.utc)
    return [
        EpisodicMemory(
            id="mem_001",
            session_id="s1",
            text="I am really worried about my upcoming job interview",
            topic_tags=["career"],
            valence_score=-0.6,
            importance_score=0.8,
            created_at=now,
        ),
        EpisodicMemory(
            id="mem_002",
            session_id="s1",
            text="Had a great conversation with my mom today",
            topic_tags=["family"],
            valence_score=0.7,
            importance_score=0.5,
            created_at=now,
        ),
        EpisodicMemory(
            id="mem_003",
            session_id="s1",
            text="I feel stuck in my current role and need a change",
            topic_tags=["career"],
            valence_score=-0.5,
            importance_score=0.9,
            created_at=now,
        ),
        EpisodicMemory(
            id="mem_004",
            session_id="s2",
            text="The weather was nice for a walk in the park",
            topic_tags=["daily_life"],
            valence_score=0.2,
            importance_score=0.1,
            created_at=now,
        ),
    ]


# ─── Integration test ───────────────────────────────────────────────────────

class TestContextAssemblerE2E:
    """End-to-end test with real stores and mocked Groq calls."""

    @patch("src.memory.working.context_assembler.classify")
    def test_full_assembly_e2e(
        self,
        mock_classify,
        chroma_store,
        affective_store,
        sample_memories,
    ):
        """
        Add real memories + affective data, assemble context,
        verify the full prompt string is correctly formatted.
        """
        # Mock Groq classifier
        mock_classify.return_value = QueryType.LONG_TERM

        # Step 1 — Add real memories to ChromaDB
        chroma_store.add_batch(sample_memories)
        assert chroma_store.count() == 4

        # Step 2 — Add affective data (simulate career fear)
        for _ in range(5):
            affective_store.update(
                topic="career",
                group_scores={
                    "joy": 0.0, "sadness": 0.2, "fear": 0.7,
                    "anger": 0.0, "guilt": 0.1, "neutral": 0.0,
                },
                intensity=0.8,
            )

        # Step 3 — Assemble context with mocked semantic store
        mock_semantic = MagicMock()
        mock_semantic.to_context.return_value = (
            "[USER PROFILE]\n- User is a software engineer\n- User prefers remote work"
        )

        result = assemble_context(
            query_text="how do I feel about my career",
            episodic_store=chroma_store,
            affective_store=affective_store,
            semantic_store=mock_semantic,
            top_k=3,
        )

        # Step 4 — Verify result structure
        assert result.query_text == "how do I feel about my career"
        assert result.query_type == "long_term"
        assert len(result.retrieved_memory_ids) == 3

        # Step 5 — Verify all context sections are present
        assert result.semantic_context.startswith("[USER PROFILE]")
        assert result.affective_context.startswith("[EMOTIONAL CONTEXT]")
        assert result.episodic_context != ""

        # Step 6 — Verify full prompt string ordering
        prompt = result.to_prompt_string()
        assert "[USER PROFILE]" in prompt
        assert "[EMOTIONAL CONTEXT]" in prompt
        assert "[RELEVANT MEMORIES]" in prompt

        # Order: semantic → affective → episodic
        profile_pos = prompt.index("[USER PROFILE]")
        emotion_pos = prompt.index("[EMOTIONAL CONTEXT]")
        memory_pos = prompt.index("[RELEVANT MEMORIES]")
        assert profile_pos < emotion_pos < memory_pos

        # Step 7 — Verify career-related memories are retrieved
        # mem_001 and mem_003 are career-related, should be in top results
        retrieved_texts = []
        for mid in result.retrieved_memory_ids:
            # Find the memory text from our sample
            for m in sample_memories:
                if m.id == mid:
                    retrieved_texts.append(m.text)
        assert any("job interview" in t for t in retrieved_texts)
        assert any("stuck" in t for t in retrieved_texts)

    @patch("src.memory.working.context_assembler.classify")
    def test_empty_e2e(
        self,
        mock_classify,
        chroma_store,
        affective_store,
    ):
        """Empty stores → empty context, no crash."""
        mock_classify.return_value = QueryType.RECENT

        mock_semantic = MagicMock()
        mock_semantic.to_context.return_value = ""

        result = assemble_context(
            query_text="hello",
            episodic_store=chroma_store,
            affective_store=affective_store,
            semantic_store=mock_semantic,
        )

        assert result.episodic_context == ""
        assert result.affective_context == ""
        assert result.semantic_context == ""
        assert result.to_prompt_string() == ""

    @patch("src.memory.working.context_assembler.classify")
    def test_affective_high_valence_only(
        self,
        mock_classify,
        chroma_store,
        affective_store,
    ):
        """Affective store with high-valence topic → emotional context appears."""
        mock_classify.return_value = QueryType.LONG_TERM

        # Build up enough mentions (>MIN_MENTIONS=3) for career topic
        for _ in range(5):
            affective_store.update(
                topic="career",
                group_scores={
                    "joy": 0.0, "sadness": 0.0, "fear": 0.9,
                    "anger": 0.0, "guilt": 0.0, "neutral": 0.0,
                },
                intensity=0.9,
            )

        # Low-valence topic (below threshold) — should NOT appear
        for _ in range(5):
            affective_store.update(
                topic="weather",
                group_scores={
                    "joy": 0.1, "sadness": 0.0, "fear": 0.0,
                    "anger": 0.0, "guilt": 0.0, "neutral": 0.9,
                },
                intensity=0.2,
            )

        mock_semantic = MagicMock()
        mock_semantic.to_context.return_value = ""

        result = assemble_context(
            query_text="how do I feel about work",
            episodic_store=chroma_store,
            affective_store=affective_store,
            semantic_store=mock_semantic,
        )

        # Career should appear (high valence), weather should not
        assert "career" in result.affective_context.lower()
        assert "weather" not in result.affective_context.lower()
