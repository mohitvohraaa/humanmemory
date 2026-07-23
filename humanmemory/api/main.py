"""
humanmemory/api/main.py
───────────────────────
FastAPI server for the HumanMemory memory service.

The service never generates responses. It provides:
  - Context retrieval for incoming messages
  - Turn storage for completed conversations
  - End-of-session consolidation

Client flow:
  1. POST /context      → get assembled memory context
  2. (call your own LLM with the context)
  3. POST /store        → store the conversation turn
  4. POST /consolidate  → end-of-session processing
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from humanmemory.system import MemoryClient


# ─── Per-user MemoryClient cache ─────────────────────────────────────────────

clients: dict[str, MemoryClient] = {}


def get_client(user_id: str) -> MemoryClient:
    if user_id not in clients:
        clients[user_id] = MemoryClient(user_id=user_id)
    return clients[user_id]


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("HumanMemory API starting.")
    yield
    print("HumanMemory API shutting down.")


app = FastAPI(
    title="HumanMemory API",
    description="5-layer biologically-inspired memory service for LLM applications",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Request / Response models ───────────────────────────────────────────────

class ContextRequest(BaseModel):
    user_id: str
    message: str


class ContextResponse(BaseModel):
    context: str
    query_type: str
    semantic_context: str
    affective_context: str
    episodic_context: str
    retrieved_memory_ids: list[str]


class StoreRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    response: str


class StoreResponse(BaseModel):
    status: str
    turn_id: str
    topics: list[str]
    emotions: dict


class ConsolidateRequest(BaseModel):
    user_id: str
    session_id: str


class ConsolidateResponse(BaseModel):
    turns_processed: int
    facts_added: list[dict]
    affective_summary: str
    semantic_summary: str


class SessionEndRequest(BaseModel):
    user_id: str
    session_id: str


class MemoryState(BaseModel):
    user_id: str
    semantic_facts: list[str]
    affective_profile: dict
    episodic_count: int


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Retrieve assembled memory context for an incoming message.

    Returns context from semantic (profile), affective (emotions),
    and episodic (relevant past conversations) layers.
    """
    client = get_client(request.user_id)
    ctx = client.get_context(request.message)

    return ContextResponse(
        context=ctx.to_prompt_string(),
        query_type=ctx.query_type,
        semantic_context=ctx.semantic_context,
        affective_context=ctx.affective_context,
        episodic_context=ctx.episodic_context,
        retrieved_memory_ids=ctx.retrieved_memory_ids,
    )


@app.post("/store", response_model=StoreResponse)
async def store_turn(request: StoreRequest):
    """
    Store a completed conversation turn.

    Persists to episodic memory, classifies emotions,
    tags topics, and updates affective store.
    """
    client = get_client(request.user_id)
    result = client.store(
        message=request.message,
        response=request.response,
        session_id=request.session_id,
    )

    return StoreResponse(
        status="stored",
        turn_id=result["turn_id"],
        topics=result["topics"],
        emotions=result["emotions"],
    )


@app.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate(request: ConsolidateRequest):
    """
    End-of-session consolidation.

    Extracts durable facts from episodic turn clusters,
    updates semantic store, and clears session buffer.
    """
    client = get_client(request.user_id)
    try:
        result = client.consolidate(session_id=request.session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No turns found for session {request.session_id}",
        )

    return ConsolidateResponse(**result)


@app.post("/session/end", response_model=ConsolidateResponse)
async def session_end(request: SessionEndRequest):
    """Backward-compatible alias for /consolidate."""
    return await consolidate(request)


@app.get("/memory/{user_id}", response_model=MemoryState)
async def get_memory(user_id: str):
    """Inspect the current memory state for a user."""
    client = get_client(user_id)
    state = client.get_memory_state()
    return MemoryState(**state)


@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    """Clear all memory for a user. Irreversible."""
    client = get_client(user_id)
    client.clear_memory()
    if user_id in clients:
        del clients[user_id]
    return {"status": "cleared", "user_id": user_id}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
