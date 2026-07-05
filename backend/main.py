"""FastAPI backend — chat, transcription, and memory endpoints."""

# load_dotenv FIRST — before any local imports that read env vars at module level
from dotenv import load_dotenv
load_dotenv()

import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import AgentRunner
from backend.memory_store import MemoryStore
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

agent: AgentRunner | None = None
memory_store: MemoryStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, memory_store
    logger.info("Initialising memory store…")
    memory_store = MemoryStore()
    await memory_store.connect()

    logger.info("Initialising agent runner…")
    agent = AgentRunner(memory_store=memory_store)

    yield

    logger.info("Shutting down…")
    await memory_store.disconnect()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Custom AI Agent with Memory",
    description="Voice/text assistant with long-term memory powered by LangChain.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    thread_id: str = "default"
    model: str = "groq/llama-3.3-70b-versatile"


class ChatResponse(BaseModel):
    response: str
    user_id: str
    thread_id: str


class TranscriptResponse(BaseModel):
    transcript: str


class MemoriesResponse(BaseModel):
    memories: list[dict]
    user_id: str


class HealthResponse(BaseModel):
    status: str
    memory_backend: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns backend status."""
    return HealthResponse(
        status="ok",
        memory_backend=os.getenv("MEMORY_BACKEND", "redis"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a text message and receive an AI response with memory."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    try:
        response = await agent.chat(
            message=req.message,
            user_id=req.user_id,
            thread_id=req.thread_id,
            model=req.model,
        )
        return ChatResponse(
            response=response,
            user_id=req.user_id,
            thread_id=req.thread_id,
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe an audio file using Groq Whisper (free) or OpenAI Whisper fallback."""
    try:
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key:
            # Groq Whisper — free, fast
            from groq import AsyncGroq
            client = AsyncGroq(api_key=groq_key)
            with open(tmp_path, "rb") as f:
                result = await client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="text",
                )
            transcript_text = result if isinstance(result, str) else result.text
        elif openai_key:
            # OpenAI Whisper fallback
            import openai
            client = openai.AsyncOpenAI(api_key=openai_key)
            with open(tmp_path, "rb") as f:
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            transcript_text = result.text
        else:
            raise ValueError("No GROQ_API_KEY or OPENAI_API_KEY found. Set at least one.")

        os.unlink(tmp_path)
        return TranscriptResponse(transcript=transcript_text)

    except Exception as exc:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memories/{user_id}", response_model=MemoriesResponse)
async def get_memories(user_id: str):
    """Retrieve all stored memories for a user."""
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialised.")

    try:
        memories = await memory_store.list_memories(user_id)
        return MemoriesResponse(memories=memories, user_id=user_id)
    except Exception as exc:
        logger.exception("Memory fetch error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/memories/{user_id}", status_code=204)
async def delete_memories(user_id: str):
    """Delete all memories for a user."""
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialised.")
    await memory_store.delete_all(user_id)
