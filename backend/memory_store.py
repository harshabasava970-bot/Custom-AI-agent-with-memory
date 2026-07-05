"""Memory store — Redis primary, ChromaDB fallback, in-memory dev mode."""

import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ── Base ──────────────────────────────────────────────────────────────────────

class MemoryStore:
    """Unified interface over Redis, ChromaDB, or in-process memory."""

    def __init__(self):
        self._backend: "_Backend" = _pick_backend()

    async def connect(self):
        await self._backend.connect()
        logger.info("Memory backend ready: %s", type(self._backend).__name__)

    async def disconnect(self):
        await self._backend.disconnect()

    def get_langgraph_store(self):
        """Return a LangGraph-compatible BaseStore for injection into the graph."""
        return self._backend.get_langgraph_store()

    async def list_memories(self, user_id: str) -> list[dict]:
        return await self._backend.list_memories(user_id)

    async def delete_all(self, user_id: str):
        await self._backend.delete_all(user_id)


def _pick_backend() -> "_Backend":
    # Read at call time so dotenv has already loaded
    MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "inmem").lower()
    if MEMORY_BACKEND == "redis":
        return _RedisBackend()
    if MEMORY_BACKEND == "chroma":
        return _ChromaBackend()
    logger.warning("Using in-memory store — data will NOT persist across restarts.")
    return _InMemBackend()


# ── Redis backend ─────────────────────────────────────────────────────────────

class _RedisBackend:
    def __init__(self):
        self._redis = None
        self._store = None

    async def connect(self):
        import redis.asyncio as aioredis
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis = aioredis.from_url(url, decode_responses=True)
        await self._redis.ping()

        # Use LangGraph's built-in AsyncRedisStore for semantic search
        try:
            from langgraph.store.redis.aio import AsyncRedisStore
            self._store = await AsyncRedisStore.from_url(
                url,
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                },
            )
        except ImportError:
            # Fallback: plain Redis without vector search
            logger.warning("langgraph[redis] not installed — using plain Redis store.")
            self._store = _RedisPlainStore(self._redis)

    async def disconnect(self):
        if self._redis:
            await self._redis.aclose()

    def get_langgraph_store(self):
        return self._store

    async def list_memories(self, user_id: str) -> list[dict]:
        if self._redis is None:
            return []
        pattern = f"memories:{user_id}:*"
        keys = await self._redis.keys(pattern)
        memories = []
        for key in keys:
            raw = await self._redis.get(key)
            if raw:
                try:
                    mem_id = key.split(":")[-1]
                    value = json.loads(raw)
                    memories.append({"key": mem_id, "value": value})
                except json.JSONDecodeError:
                    pass
        return memories

    async def delete_all(self, user_id: str):
        if self._redis:
            pattern = f"memories:{user_id}:*"
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)


# ── ChromaDB backend ──────────────────────────────────────────────────────────

class _ChromaBackend:
    def __init__(self):
        self._client = None
        self._store = None

    async def connect(self):
        import chromadb
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        self._client = chromadb.PersistentClient(path=persist_dir)

        # Wrap ChromaDB in a LangGraph-compatible adapter
        self._store = _ChromaLangGraphStore(self._client)
        logger.info("ChromaDB ready at %s", persist_dir)

    async def disconnect(self):
        pass  # ChromaDB PersistentClient auto-persists

    def get_langgraph_store(self):
        return self._store

    async def list_memories(self, user_id: str) -> list[dict]:
        if self._client is None:
            return []
        try:
            col = self._client.get_or_create_collection(f"memories_{user_id}")
            result = col.get()
            memories = []
            for i, doc_id in enumerate(result.get("ids", [])):
                meta = result.get("metadatas", [{}])[i] or {}
                memories.append({"key": doc_id, "value": meta})
            return memories
        except Exception as exc:
            logger.warning("ChromaDB list error: %s", exc)
            return []

    async def delete_all(self, user_id: str):
        if self._client:
            try:
                self._client.delete_collection(f"memories_{user_id}")
            except Exception:
                pass


# ── In-memory fallback backend ────────────────────────────────────────────────

class _InMemBackend:
    def __init__(self):
        self._store = None

    async def connect(self):
        from langgraph.store.memory import InMemoryStore
        self._store = InMemoryStore()

    async def disconnect(self):
        pass

    def get_langgraph_store(self):
        return self._store

    async def list_memories(self, user_id: str) -> list[dict]:
        if self._store is None:
            return []
        try:
            items = await self._store.asearch(("memories", user_id), query="", limit=100)
            return [{"key": m.key, "value": m.value} for m in items]
        except Exception:
            return []

    async def delete_all(self, user_id: str):
        pass  # In-memory — cleared on restart anyway


# ── Thin adapters (used when full integration packages are absent) ─────────────

class _RedisPlainStore:
    """Minimal LangGraph BaseStore shim backed by plain Redis JSON keys."""

    def __init__(self, redis_client):
        self._r = redis_client

    async def aput(self, namespace: tuple, key: str, value: dict, **_):
        redis_key = ":".join(namespace) + ":" + key
        await self._r.set(redis_key, json.dumps(value))

    async def asearch(self, namespace: tuple, query: str, limit: int = 10, **_):
        # No vector search — return all memories (simple fallback)
        pattern = ":".join(namespace) + ":*"
        keys = await self._r.keys(pattern)
        results = []
        for k in keys[:limit]:
            raw = await self._r.get(k)
            if raw:
                results.append(_MemItem(key=k.split(":")[-1], value=json.loads(raw), score=1.0))
        return results

    async def aget(self, namespace: tuple, key: str, **_):
        redis_key = ":".join(namespace) + ":" + key
        raw = await self._r.get(redis_key)
        return json.loads(raw) if raw else None

    async def adelete(self, namespace: tuple, key: str, **_):
        redis_key = ":".join(namespace) + ":" + key
        await self._r.delete(redis_key)


class _ChromaLangGraphStore:
    """Minimal LangGraph BaseStore shim backed by ChromaDB."""

    def __init__(self, client):
        self._client = client

    def _col(self, namespace: tuple):
        name = "_".join(namespace)[:63]  # ChromaDB name limit
        return self._client.get_or_create_collection(name)

    async def aput(self, namespace: tuple, key: str, value: dict, **_):
        col = self._col(namespace)
        content_text = value.get("content", str(value))
        col.upsert(ids=[key], documents=[content_text], metadatas=[value])

    async def asearch(self, namespace: tuple, query: str, limit: int = 10, **_):
        col = self._col(namespace)
        try:
            res = col.query(query_texts=[query or " "], n_results=min(limit, col.count()))
            results = []
            ids = res.get("ids", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0]
            for i, doc_id in enumerate(ids):
                score = 1 - (distances[i] if distances else 0)
                results.append(_MemItem(key=doc_id, value=metas[i] or {}, score=score))
            return results
        except Exception:
            return []

    async def aget(self, namespace: tuple, key: str, **_):
        col = self._col(namespace)
        res = col.get(ids=[key])
        metas = res.get("metadatas", [])
        return metas[0] if metas else None

    async def adelete(self, namespace: tuple, key: str, **_):
        col = self._col(namespace)
        col.delete(ids=[key])


class _MemItem:
    """Simple memory result object matching LangGraph's SearchItem interface."""

    def __init__(self, key: str, value: Any, score: float):
        self.key = key
        self.value = value
        self.score = score
