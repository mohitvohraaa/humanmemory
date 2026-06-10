"""
src/memory/episodic/semantic_store.py
──────────────────────────────────────
Semantic retrieval using ChromaDB + sentence-transformers.

How it works:
  1. When we store a memory → convert text to vector → save in ChromaDB
  2. When query comes in   → convert query to vector → find closest vectors
  3. Return memories whose meaning is closest to the query

This is Baseline-2. It will beat naive retrieval (0.0) significantly
because it understands meaning, not just recency.

ChromaDB stores:
  - the vector (384 numbers from all-MiniLM-L6-v2)
  - the memory ID (links back to SQLite for full metadata)
  - basic metadata (topic_tags, valence_score, created_at)
"""

from __future__ import annotations

from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from src.memory.models import EpisodicMemory


class SemanticEpisodicStore:
    """
    ChromaDB-backed semantic memory store.

    Works alongside EpisodicStore (SQLite) — they share the same memory IDs.
    SQLite stores full metadata. ChromaDB stores vectors for similarity search.

    Usage:
        store = SemanticEpisodicStore()
        store.add(memory)
        results = store.semantic_retrieve("job stress", limit=10)
    """

    def __init__(
        self,
        persist_dir: str = "data/processed/chroma",
        collection_name: str = "episodic_memories",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        # Load embedding model
        # all-MiniLM-L6-v2: fast, 80MB, 384 dimensions, good quality
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB with persistent storage
        # persistent = survives between runs, stored on disk
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create collection
        # A collection is like a table in SQL — holds related vectors
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            # hnsw:space=cosine means ChromaDB uses cosine similarity
            # other options: "l2" (euclidean), "ip" (inner product)
        )
        print(f"ChromaDB collection: {collection_name}")
        print(f"Memories in collection: {self.collection.count()}")

    # ── Write ─────────────────────────────────────────────────────────────

    def add(self, memory: EpisodicMemory) -> None:
        """
        Convert memory text to vector and store in ChromaDB.
        Also stores metadata so we can filter and sort later.
        """
        # Step 1 — convert text to vector
        vector = self.model.encode(memory.text).tolist()

        # Step 2 — store in ChromaDB
        # document = original text (ChromaDB keeps this for reference)
        # embedding = the vector
        # metadata = structured fields we can filter on
        # id = links back to SQLite record
        self.collection.add(
            documents=[memory.text],
            embeddings=[vector],
            metadatas=[{
                "session_id": memory.session_id,
                "topic_tags": ",".join(memory.topic_tags),
                "valence_score": memory.valence_score,
                "importance_score": memory.importance_score,
                "created_at": memory.created_at.isoformat()
                if isinstance(memory.created_at, datetime)
                else memory.created_at,
            }],
            ids=[memory.id],
        )

    def add_batch(self, memories: list[EpisodicMemory]) -> None:
        """
        Add multiple memories efficiently in one ChromaDB call.
        Much faster than calling add() in a loop for large batches.
        """
        if not memories:
            return

        # Encode all texts in one batch — much faster than one by one
        texts = [m.text for m in memories]
        vectors = self.model.encode(texts, show_progress_bar=True).tolist()

        self.collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=[{
                "session_id": m.session_id,
                "topic_tags": ",".join(m.topic_tags),
                "valence_score": m.valence_score,
                "importance_score": m.importance_score,
                "created_at": m.created_at.isoformat()
                if isinstance(m.created_at, datetime)
                else m.created_at,
            } for m in memories],
            ids=[m.id for m in memories],
        )

    # ── Read ──────────────────────────────────────────────────────────────

    def semantic_retrieve(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        BASELINE-2: Find memories most semantically similar to query.

        Steps:
          1. Convert query text to vector
          2. ChromaDB finds the limit closest vectors by cosine similarity
          3. Return results with similarity scores

        Returns list of dicts with keys:
          id, text, score, metadata
        """
        if self.collection.count() == 0:
            return []

        # Convert query to vector
        query_vector = self.model.encode(query_text).tolist()

        # Search ChromaDB
        # n_results = how many to return
        # include = what fields to include in results
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns distances (lower = more similar for cosine)
        # Convert to similarity scores (higher = more similar)
        retrieved = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # cosine distance → similarity

            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": similarity,
                "metadata": results["metadatas"][0][i],
            })

        # Sort by similarity score descending (most similar first)
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved

    # ── Utilities ─────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Delete all vectors. Useful for test teardown."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
