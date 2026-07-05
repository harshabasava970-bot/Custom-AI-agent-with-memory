# Custom AI Agent with Memory

A voice/text assistant with long-term memory, built on [langchain-ai/memory-agent](https://github.com/langchain-ai/memory-agent).

The agent remembers facts about each user across conversations and threads. Talk to it by typing or by recording your voice — it transcribes speech with OpenAI Whisper and stores memories using Redis (with semantic vector search) or ChromaDB.

---

## Stack

| Layer | Technology |
|---|---|
| **Backend** | Python · FastAPI · LangGraph · LangChain |
| **AI model** | OpenAI GPT-4o-mini (default) · Anthropic Claude support |
| **Voice** | OpenAI Whisper (`whisper-1`) |
| **Memory (primary)** | Redis Stack (vector search via RedisSearch) |
| **Memory (fallback)** | ChromaDB · In-memory (dev) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Frontend** | Streamlit |
| **Free deployment** | Railway · Render · Streamlit Community Cloud |

---

## Project Structure

```
Custom-AI-agent-with-memory/
├── src/memory_agent/        # LangGraph agent (from langchain-ai/memory-agent)
│   ├── graph.py             # ReAct graph with memory tool
│   ├── tools.py             # upsert_memory tool
│   ├── state.py             # Graph state
│   ├── context.py           # Runtime context (user_id, model, prompt)
│   ├── prompts.py           # System prompt
│   └── utils.py             # load_chat_model helper
├── backend/
│   ├── main.py              # FastAPI app (chat, transcribe, memories endpoints)
│   ├── agent.py             # AgentRunner — bridges API ↔ LangGraph
│   └── memory_store.py      # Redis / ChromaDB / InMemory store adapters
├── frontend/
│   └── app.py               # Streamlit UI (voice + text chat, memory panel)
├── .streamlit/config.toml   # Streamlit theme
├── Dockerfile               # Multi-stage: api | frontend
├── docker-compose.yml       # Local full stack (Redis + API + Frontend)
├── render.yaml              # Render.com free deployment
├── railway.toml             # Railway.app free deployment
├── Procfile                 # Heroku / Railway fallback
├── requirements.txt
└── .env.example
```

---

## Quick Start (local)

### 1. Clone and set up

```bash
git clone https://github.com/harshabasava970-bot/Custom-AI-agent-with-memory.git
cd Custom-AI-agent-with-memory
cp .env.example .env
```

Edit `.env` and add your `OPENAI_API_KEY`.

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| RedisInsight | http://localhost:8001 |

### 3. Run without Docker

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# Start Redis (or set MEMORY_BACKEND=inmem in .env to skip)
# docker run -p 6379:6379 redis/redis-stack

# Terminal 1 — API
set PYTHONPATH=src              # Windows
# export PYTHONPATH=src         # Mac/Linux
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
set API_URL=http://localhost:8000
streamlit run frontend/app.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a message, get a reply |
| `POST` | `/transcribe` | Upload audio → transcript (Whisper) |
| `GET` | `/memories/{user_id}` | List all memories for a user |
| `DELETE` | `/memories/{user_id}` | Wipe all memories for a user |

**Chat request body:**
```json
{
  "message": "My name is Harsha and I love Python.",
  "user_id": "harsha-123",
  "thread_id": "thread-abc",
  "model": "openai/gpt-4o-mini"
}
```

---

## Free Deployment (1000+ requests/day)

### Option A — Render (API) + Streamlit Community Cloud (Frontend) ⭐ Recommended

**750 free hours/month on Render (enough for 24/7 with uptime pinging), unlimited on Streamlit Cloud.**

#### Step 1 — Deploy the API on Render

1. Go to [render.com](https://render.com) → Sign up (free, no credit card needed)
2. Click **New** → **Web Service** → Connect GitHub → select `harshabasava970-bot/Custom-AI-agent-with-memory`
3. Configure the service:
   - **Name:** `memory-agent-api`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
4. Scroll to **Environment Variables** → add:
   ```
   OPENAI_API_KEY=sk-...
   MEMORY_BACKEND=inmem
   PYTHONPATH=./src
   ```
   > Start with `inmem` — it works with zero setup. Switch to `redis` later (see below).
5. Click **Create Web Service** — Render builds and deploys in ~3 minutes
6. Copy your Render URL: `https://memory-agent-api.onrender.com`

**Optional: Add Redis for persistent memory**

1. In Render dashboard → **New** → **Redis** → Free plan (25 MB)
2. Copy the **Internal Redis URL**
3. In your API service → Environment → update:
   ```
   MEMORY_BACKEND=redis
   REDIS_URL=redis://red-xxxxx:6379   ← paste internal URL here
   ```

**Keep it awake (avoid 50s cold starts):**

Render free services sleep after 15 min of inactivity. Fix it for free:
1. Sign up at [uptimerobot.com](https://uptimerobot.com)
2. New monitor → HTTP(S) → URL: `https://memory-agent-api.onrender.com/health`
3. Interval: every **10 minutes** → this keeps the service warm 24/7

#### Step 2 — Deploy the Frontend on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
2. Click **New app**
3. Repo: `harshabasava970-bot/Custom-AI-agent-with-memory`
4. Branch: `main` | Main file: `frontend/app.py`
5. Click **Advanced settings** → **Secrets** → paste:
   ```toml
   API_URL = "https://memory-agent-api.onrender.com"
   ```
6. Click **Deploy** — live at `https://your-app.streamlit.app` in ~2 minutes

---

### Option B — Railway (API) + Streamlit Community Cloud (Frontend)

Railway gives ~500 free hours/month. Good alternative if Render is slow for you.

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
2. Select `harshabasava970-bot/Custom-AI-agent-with-memory`
3. Railway auto-detects `railway.toml` → builds the Docker image
4. Add a **Redis** plugin: **+ New** → **Database** → **Redis**
5. Set environment variables:
   ```
   OPENAI_API_KEY=sk-...
   MEMORY_BACKEND=redis
   REDIS_URL=${{Redis.REDIS_URL}}
   PYTHONPATH=./src
   ```
6. Copy your Railway URL → use it as `API_URL` in Streamlit Cloud secrets (same Step 2 above)

---

### Option C — Hugging Face Spaces (frontend only)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. SDK: **Streamlit**, Visibility: Public
3. Upload `frontend/app.py`, `.streamlit/config.toml`, and a minimal `requirements.txt`:
   ```
   streamlit==1.45.1
   requests==2.32.3
   ```
4. Set Space secret `API_URL` to your Railway/Render API URL

---

## Capacity — 1000+ requests/day

| Platform | Free Tier | Requests/day |
|---|---|---|
| Render (API) ⭐ | 750 hrs/mo + UptimeRobot ping = 24/7 | ~2000–5000 |
| Railway (API) | 500 hrs/mo (~16 hrs/day active) | ~1000–3000 |
| Streamlit Cloud (UI) | Unlimited | Unlimited |
| Render Redis | 25 MB free (~10k memories) | Millions of ops |
| Railway Redis | 500 MB included | Millions of ops |
| OpenAI API | Pay-per-use | Unlimited (cost ~$0.001/turn) |

GPT-4o-mini costs roughly **$0.001 per conversation turn** — 1000 requests/day = ~$1/day.

---

## Memory Backends

### Redis (recommended for production)

```env
MEMORY_BACKEND=redis
REDIS_URL=redis://localhost:6379
```

Uses `langgraph[redis]` with `AsyncRedisStore` + RedisSearch for semantic vector search.
Memories are persisted to disk and survive restarts.

### ChromaDB (good for single-server deploys)

```env
MEMORY_BACKEND=chroma
CHROMA_PERSIST_DIR=./chroma_data
```

Persists locally. No external service needed. Good for Render/Railway single-container deploys.

### In-Memory (dev only)

```env
MEMORY_BACKEND=inmem
```

No persistence — memories reset on restart. Use for local testing only.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI key (chat + Whisper + embeddings) |
| `ANTHROPIC_API_KEY` | No | — | Only needed for Anthropic models |
| `MEMORY_BACKEND` | No | `redis` | `redis` / `chroma` / `inmem` |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection string |
| `CHROMA_PERSIST_DIR` | No | `./chroma_data` | ChromaDB data directory |
| `MODEL` | No | `openai/gpt-4o-mini` | Default LLM (`provider/model`) |
| `API_URL` | No | `http://localhost:8000` | Backend URL (used by frontend) |
| `LANGSMITH_API_KEY` | No | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | — | Set to `true` to enable tracing |

---

## How Memory Works

```
User sends message
       │
       ▼
  call_model ──── Fetches top-10 relevant memories (semantic search)
       │           Injects them into system prompt
       │           Calls LLM with memory context
       │
   tool call? ──── Yes ──▶ store_memory ──▶ upsert to store ──▶ back to call_model
       │
       No
       │
       ▼
  Return reply to user
```

Each memory has:
- `content` — the fact to remember (e.g. "User's name is Harsha")
- `context` — when/why it was noted (e.g. "Mentioned during first conversation")

Memories are scoped by `user_id` and persist across threads.

---

## Credits

- Core agent architecture from [langchain-ai/memory-agent](https://github.com/langchain-ai/memory-agent) (MIT License)
- Extended with voice input, FastAPI REST layer, Redis/ChromaDB memory, Streamlit UI, and free-tier deployment configs

---

## Push to GitHub

Run these from inside `Custom-AI-agent-with-memory/`:

```bash
git init
git add .
git commit -m "feat: voice/text AI agent with long-term memory"
git remote add origin https://github.com/harshabasava970-bot/Custom-AI-agent-with-memory.git
git branch -M main
git push -u origin main
```

## Fastest Free Deployment (Render + Streamlit Cloud)

1. Push repo to GitHub (above)
2. **[render.com](https://render.com)** → New Web Service → connect repo → set:
   - `OPENAI_API_KEY=sk-...`
   - `PYTHONPATH=./src`
   - `MEMORY_BACKEND=inmem` (or `redis` if you add a Render Redis instance)
3. **[share.streamlit.io](https://share.streamlit.io)** → New app → same repo → `frontend/app.py` → Advanced settings → secret:
   ```toml
   API_URL = "https://your-service.onrender.com"
   ```
4. **[uptimerobot.com](https://uptimerobot.com)** → HTTP monitor → `https://your-service.onrender.com/health` → every 10 min
5. Live in ~10 minutes, handles 1000+ requests/day, completely free
