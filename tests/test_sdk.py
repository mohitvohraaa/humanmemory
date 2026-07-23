"""
tests/test_sdk.py
─────────────────
Integration tests for MemoryClient (humanmemory SDK).

Tests the 3-line integration flow:
  get_context → store → consolidate
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_groq():
    """Mock Groq client that returns predictable responses."""
    client = MagicMock()

    # classify returns "recent"
    # tag returns ["work"]
    # extract_fact returns a fact
    def chat_create(**kwargs):
        messages = kwargs.get("messages", [])
        content = messages[-1]["content"] if messages else ""

        resp = MagicMock()
        resp.choices = [MagicMock()]

        # Query classifier
        if "query classifier" in str(messages).lower():
            resp.choices[0].message.content = "recent"
        # Topic tagger
        elif "topic classifier" in str(messages).lower():
            resp.choices[0].message.content = "work"
        # Fact extraction
        elif "extracting durable facts" in str(messages).lower():
            resp.choices[0].message.content = '{"fact": "User works in tech", "category": "identity"}'
        # Fact check
        elif "managing a user profile" in str(messages).lower():
            resp.choices[0].message.content = "NEW"
        else:
            resp.choices[0].message.content = "recent"

        return resp

    client.chat.completions.create = chat_create
    return client


@pytest.fixture
def memory_client(tmp_path, mock_groq):
    """MemoryClient with mocked Groq and temp storage."""
    from humanmemory.system import MemoryClient

    client = MemoryClient(
        user_id="test_user",
        storage_dir=str(tmp_path),
    )
    # Inject mock Groq client into all stores
    client._groq = mock_groq
    client._semantic.client = mock_groq
    client._topic_tagger.client = mock_groq

    yield client
    client.clear_memory()


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestMemoryClientInit:
    """MemoryClient initializes all stores correctly."""

    def test_creates_stores(self, memory_client):
        assert memory_client._user_id == "test_user"
        assert memory_client._episodic_sqlite is not None
        assert memory_client._episodic_chroma is not None
        assert memory_client._affective is not None
        assert memory_client._semantic is not None


class TestGetContext:
    """get_context() returns assembled context from all layers."""

    @patch("src.memory.working.context_assembler.classify")
    def test_get_context_returns_assembled_context(self, mock_classify, memory_client):
        from src.retrieval.query_classifier import QueryType
        mock_classify.return_value = QueryType.RECENT

        ctx = memory_client.get_context("How is work going?")

        assert ctx.query_text == "How is work going?"
        assert ctx.query_type == "recent"
        assert isinstance(ctx.to_prompt_string(), str)

    @patch("src.memory.working.context_assembler.classify")
    def test_get_context_empty_stores(self, mock_classify, memory_client):
        from src.retrieval.query_classifier import QueryType
        mock_classify.return_value = QueryType.RECENT

        ctx = memory_client.get_context("hello")

        assert ctx.episodic_context == ""
        assert ctx.semantic_context == ""


class TestStore:
    """store() persists turns and runs lightweight processing."""

    def test_store_returns_turn_id_and_topics(self, memory_client):
        result = memory_client.store(
            message="I had a stressful meeting today",
            response="That sounds tough.",
            session_id="s1",
        )

        assert "turn_id" in result
        assert "topics" in result
        assert "emotions" in result

    def test_store_persists_to_episodic(self, memory_client):
        memory_client.store(
            message="I love morning runs",
            response="Great habit!",
            session_id="s1",
        )

        count = memory_client._episodic_sqlite.count()
        assert count >= 2  # user + assistant turn

    def test_store_buffers_for_consolidation(self, memory_client):
        memory_client.store(
            message="Work is stressful",
            response="I understand.",
            session_id="s1",
        )

        assert "s1" in memory_client._session_turns
        assert len(memory_client._session_turns["s1"]) == 1


class TestConsolidate:
    """consolidate() extracts facts and clears session buffer."""

    def test_consolidate_requires_turns(self, memory_client):
        with pytest.raises(KeyError):
            memory_client.consolidate(session_id="nonexistent")

    def test_consolidate_processes_turns(self, memory_client):
        # Store some turns first
        for msg in ["Work is stressful", "I feel stuck at my job"]:
            memory_client.store(
                message=msg, response="I see.", session_id="s1"
            )

        result = memory_client.consolidate(session_id="s1")

        assert result["turns_processed"] == 2
        assert "facts_added" in result
        assert "s1" not in memory_client._session_turns


class TestMemoryState:
    """get_memory_state() and clear_memory() work correctly."""

    def test_get_memory_state_empty(self, memory_client):
        state = memory_client.get_memory_state()

        assert state["user_id"] == "test_user"
        assert state["semantic_facts"] == []
        assert state["episodic_count"] == 0

    def test_clear_memory(self, memory_client):
        memory_client.store(
            message="test", response="test", session_id="s1"
        )
        assert memory_client._episodic_sqlite.count() > 0

        memory_client.clear_memory()
        assert memory_client._episodic_sqlite.count() == 0


class TestSDKImports:
    """Verify the public API imports work correctly."""

    def test_import_memory_client(self):
        from humanmemory import MemoryClient
        assert MemoryClient is not None

    def test_import_create_groq_client(self):
        from humanmemory import create_groq_client
        assert callable(create_groq_client)
