"""
humanmemory/system.py
─────────────────────
MemoryClient — the 3-line integration point for any LLM application.

Usage:
    from humanmemory import MemoryClient

    mem = MemoryClient(user_id="alice")
    context = mem.get_context("How's my mom doing?")
    # ... call your own LLM with context ...
    mem.store(message="How's my mom doing?", response="She's doing well!")
    mem.consolidate()
"""

from __future__ import annotations

import uuid
from datetime import datetime

from src.memory.affective.affective_store import AffectiveStore
from src.memory.affective.emotion_classifier import EmotionClassifier
from src.memory.affective.topic_tagger import TopicTagger
from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.episodic.store import EpisodicStore
from src.memory.models import EpisodicMemory
from src.memory.semantic.run_consolidation import consolidate_session
from src.memory.semantic.semantic_store import SemanticStore
from src.memory.working.context_assembler import AssembledContext, assemble_context

from .llm import create_groq_client


class MemoryClient:
    """
    High-level memory service for LLM applications.

    Provides context retrieval, turn storage, and session consolidation
    without generating any responses — the user's own LLM handles generation.

    Args:
        user_id:      unique user identifier
        groq_api_key: Groq API key for internal operations. Falls back to
                      GROQ_API_KEY env var.
        storage_dir:  base directory for persistent storage
    """

    def __init__(
        self,
        user_id: str,
        groq_api_key: str | None = None,
        storage_dir: str = "data/processed",
    ):
        self._user_id = user_id
        self._storage_dir = storage_dir

        # Shared Groq client for all internal LLM operations
        self._groq = create_groq_client(groq_api_key)

        # Memory stores
        self._episodic_sqlite = EpisodicStore(
            db_path=f"{storage_dir}/{user_id}_episodic.db"
        )
        self._episodic_chroma = SemanticEpisodicStore(
            persist_dir=f"{storage_dir}/chroma",
            collection_name=f"episodic_{user_id}",
        )
        self._affective = AffectiveStore(
            user_id=user_id,
            storage_dir=f"{storage_dir}/affective",
        )
        self._semantic = SemanticStore(
            user_id=user_id,
            storage_dir=f"{storage_dir}/semantic",
            groq_client=self._groq,
        )

        # Internal classifiers
        self._emotion_clf = EmotionClassifier()
        self._topic_tagger = TopicTagger(groq_client=self._groq)

        # Session turn buffer: {session_id: [turn_dict, ...]}
        self._session_turns: dict[str, list[dict]] = {}

    def get_context(self, message: str, top_k: int = 10) -> AssembledContext:
        """
        Retrieve relevant memories for an incoming message.

        Assembles context from all 3 built layers:
          - Semantic (durable user profile facts)
          - Affective (topic-specific emotional state)
          - Episodic (most relevant past conversations)

        Args:
            message: the user's current message
            top_k:   max number of episodic memories to include

        Returns:
            AssembledContext with context strings and metadata.
            Call .to_prompt_string() for the final injection string.
        """
        return assemble_context(
            query_text=message,
            episodic_store=self._episodic_chroma,
            affective_store=self._affective,
            semantic_store=self._semantic,
            groq_client=self._groq,
            top_k=top_k,
        )

    def store(
        self,
        message: str,
        response: str,
        session_id: str = "default",
    ) -> dict:
        """
        Store a completed conversation turn.

        Immediately:
          - Persists both user and assistant turns to episodic memory
          - Classifies emotions and tags topics
          - Updates affective store

        Buffers the turn for later consolidation via consolidate().

        Args:
            message:    the user's message
            response:   the LLM's response (from user's own LLM)
            session_id: session identifier for grouping turns

        Returns:
            dict with turn_id, topics, emotions
        """
        turn_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Store user turn in episodic memory
        user_memory = EpisodicMemory(
            id=turn_id,
            session_id=session_id,
            turn_id=turn_id,
            speaker="user",
            text=message,
            created_at=now,
        )
        self._episodic_sqlite.add(user_memory)
        self._episodic_chroma.add(user_memory)

        # Store assistant turn in episodic memory
        assistant_turn_id = str(uuid.uuid4())
        assistant_memory = EpisodicMemory(
            id=assistant_turn_id,
            session_id=session_id,
            turn_id=assistant_turn_id,
            speaker="assistant",
            text=response,
            created_at=now,
        )
        self._episodic_sqlite.add(assistant_memory)
        self._episodic_chroma.add(assistant_memory)

        # Classify emotions on user message
        emotion_result = self._emotion_clf.classify(message)

        # Tag topics on user message
        topics = self._topic_tagger.tag(message)

        # Update affective store
        self._affective.update_from_turn(
            topics=topics,
            group_scores=emotion_result.group_scores,
            intensity=emotion_result.intensity,
        )

        # Buffer for consolidation
        if session_id not in self._session_turns:
            self._session_turns[session_id] = []
        self._session_turns[session_id].append({
            "text": message,
            "turn_id": turn_id,
            "topics": topics,
        })

        return {
            "turn_id": turn_id,
            "topics": topics,
            "emotions": emotion_result.group_scores,
        }

    def consolidate(self, session_id: str = "default") -> dict:
        """
        End-of-session consolidation.

        Extracts durable facts from episodic turn clusters,
        updates the semantic store, and clears the session buffer.

        Args:
            session_id: session to consolidate

        Returns:
            dict with turns_processed, facts_added, summaries

        Raises:
            KeyError: if session_id has no buffered turns
        """
        turns = self._session_turns.get(session_id, [])
        if not turns:
            raise KeyError(f"No turns found for session {session_id}")

        # Run semantic fact extraction
        facts_added = consolidate_session(
            turns=turns,
            semantic_store=self._semantic,
            topic_tagger=self._topic_tagger,
            groq_client=self._groq,
        )

        # Clear session buffer
        del self._session_turns[session_id]

        return {
            "turns_processed": len(turns),
            "facts_added": facts_added,
            "affective_summary": self._affective.to_context(),
            "semantic_summary": self._semantic.to_context(),
        }

    def get_memory_state(self) -> dict:
        """
        Inspect the current memory state for this user.

        Returns:
            dict with semantic_facts, affective_profile, episodic_count
        """
        facts = self._semantic.get_facts()
        affective_records = self._affective.get_all()

        return {
            "user_id": self._user_id,
            "semantic_facts": [f.fact_text for f in facts],
            "affective_profile": {
                r.topic: r.emotions.dominant_emotion()
                for r in affective_records
            },
            "episodic_count": self._episodic_sqlite.count(),
        }

    def clear_memory(self) -> None:
        """Clear all memory for this user. Irreversible."""
        self._affective.clear()
        self._semantic.clear()
        self._episodic_sqlite.clear()
        self._episodic_chroma.clear()
        self._session_turns.clear()
