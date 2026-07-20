"""
src/api/main.py
────────────────
HumanMemory FastAPI server.

Endpoints:
  POST /chat           → assembles context, calls LLM, stores turn
  POST /session/end    → runs affective + semantic consolidation
  GET  /memory/{user_id} → inspect current memory state
  DELETE /memory/{user_id} → clear all memory for a user
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel

from src.memory.affective.affective_store import AffectiveStore
from src.memory.affective.emotion_classifier import EmotionClassifier
from src.memory.affective.topic_tagger import TopicTagger
from src.memory.episodic.semantic_store import SemanticEpisodicStore
from src.memory.episodic.store import EpisodicStore
from src.memory.models import EpisodicMemory
from src.memory.semantic.run_consolidation import consolidate_session
from src.memory.semantic.semantic_store import SemanticStore
from src.memory.working.context_assembler import assemble_context

load_dotenv()


# ─── Global model instances ───────────────────────────────────────────────────
# Loaded once at startup, reused across all requests.
# Loading models per-request would add 2-5 seconds of latency per call.

emotion_clf: EmotionClassifier | None = None
topic_tagger: TopicTagger | None = None
groq_client: Groq | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models at startup, release at shutdown."""
    global emotion_clf, topic_tagger, groq_client
    print("Loading models...")
    emotion_clf = EmotionClassifier()
    topic_tagger = TopicTagger()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Models ready. Server starting.")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="HumanMemory API",
    description="5-layer biologically-inspired memory for companion AI",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Per-user store factories ─────────────────────────────────────────────────
# Each user gets their own store instances.
# In production: cache these in Redis; for now, instantiate per request.

def get_stores(user_id: str) -> dict:
    return {
        "episodic_sqlite": EpisodicStore(
            db_path=f"data/processed/{user_id}_episodic.db"
        ),
        "episodic_chroma": SemanticEpisodicStore(
            persist_dir="data/processed/chroma",
            collection_name=f"episodic_{user_id}",
        ),
        "affective": AffectiveStore(user_id=user_id),
        "semantic": SemanticStore(user_id=user_id),
    }


# ─── Session turn buffer ──────────────────────────────────────────────────────
# Stores turns per session in memory until /session/end is called.
# In production: use Redis with TTL.

session_turns: dict[str, list[dict]] = {}


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    query_type: str
    context_used: dict


class SessionEndRequest(BaseModel):
    user_id: str
    session_id: str


class SessionEndResponse(BaseModel):
    turns_processed: int
    facts_added: list[dict]
    affective_summary: str
    semantic_summary: str


class MemoryState(BaseModel):
    user_id: str
    semantic_facts: list[str]
    affective_profile: dict
    episodic_count: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Assembles context from all 3 built layers, calls LLM, stores turn.
    """
    stores = get_stores(request.user_id)

    # Step 1 — assemble context from all layers
    ctx = assemble_context(
        query_text=request.message,
        episodic_store=stores["episodic_chroma"],
        affective_store=stores["affective"],
        semantic_store=stores["semantic"],
        top_k=5,
    )

    # Step 2 — call LLM with assembled context
    context_str = ctx.to_prompt_string()
    if context_str:
        system_prompt = (
            "You are a compassionate companion AI. Use the context below "
            "to respond in a way that reflects knowledge of this user's "
            "history, emotional patterns, and current situation.\n\n"
            + context_str
        )
    else:
        system_prompt = (
            "You are a compassionate companion AI. This is your first "
            "conversation with this user. You have no prior context about "
            "them. Respond warmly and ask questions to learn about them. "
            "Do NOT fabricate past conversations or assume you know "
            "anything about their history."
        )

    llm_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ],
        max_tokens=300,
        temperature=0.5,
    )
    response_text = llm_response.choices[0].message.content

    # Step 3 — store this turn in episodic memory
    turn_id = str(uuid.uuid4())
    memory = EpisodicMemory(
        id=turn_id,
        session_id=request.session_id,
        turn_id=turn_id,
        speaker="user",
        text=request.message,
        created_at=datetime.utcnow(),
    )
    stores["episodic_sqlite"].add(memory)
    stores["episodic_chroma"].add(memory)

    # Step 4 — buffer turn for session-end consolidation
    if request.session_id not in session_turns:
        session_turns[request.session_id] = []
    session_turns[request.session_id].append({
        "text": request.message,
        "turn_id": turn_id,
        "topics": [],  # filled in during /session/end
    })

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        query_type=ctx.query_type,
        context_used={
            "semantic": ctx.semantic_context,
            "affective": ctx.affective_context,
            "episodic_preview": ctx.episodic_context[:200],
        },
    )


@app.post("/session/end", response_model=SessionEndResponse)
async def session_end(request: SessionEndRequest):
    """
    Runs affective + semantic consolidation on all turns from this session.
    Call this when a conversation session ends.
    """
    turns = session_turns.get(request.session_id, [])
    if not turns:
        raise HTTPException(
            status_code=404,
            detail=f"No turns found for session {request.session_id}"
        )

    stores = get_stores(request.user_id)
    tagged_turns = []

    # Tag topics + update affective store per turn
    for turn in turns:
        result = emotion_clf.classify(turn["text"])
        topics = topic_tagger.tag(turn["text"])
        stores["affective"].update_from_turn(
            topics=topics,
            group_scores=result.group_scores,
            intensity=result.intensity,
        )
        tagged_turns.append({**turn, "topics": topics})

    # Run semantic consolidation
    facts_added = consolidate_session(tagged_turns, stores["semantic"])

    # Clear session buffer
    del session_turns[request.session_id]

    return SessionEndResponse(
        turns_processed=len(turns),
        facts_added=facts_added,
        affective_summary=stores["affective"].to_context(),
        semantic_summary=stores["semantic"].to_context(),
    )


@app.get("/memory/{user_id}", response_model=MemoryState)
async def get_memory(user_id: str):
    """Inspect the current memory state for a user."""
    stores = get_stores(user_id)

    facts = stores["semantic"].get_facts()
    affective_records = stores["affective"].get_all()

    return MemoryState(
        user_id=user_id,
        semantic_facts=[f.fact_text for f in facts],
        affective_profile={
            r.topic: r.emotions.dominant_emotion()
            for r in affective_records
        },
        episodic_count=stores["episodic_sqlite"].count(),
    )


@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    """Clear all memory for a user. Irreversible."""
    stores = get_stores(user_id)
    stores["affective"].clear()
    stores["semantic"].clear()
    stores["episodic_sqlite"].clear()
    stores["episodic_chroma"].clear()
    return {"status": "cleared", "user_id": user_id}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
