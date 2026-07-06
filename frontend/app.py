"""Streamlit frontend — voice/text chat with long-term memory."""

import hashlib
import io
import os
import uuid

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Memory Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Session state bootstrap ───────────────────────────────────────────────────
for key, default in [
    ("user_id",        str(uuid.uuid4())),
    ("thread_id",      str(uuid.uuid4())),
    ("messages",       []),
    ("memories",       []),
    ("last_audio_hash", None),   # ← prevents audio loop
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────

def send_message(text: str) -> str:
    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={
                "message": text,
                "user_id": st.session_state.user_id,
                "thread_id": st.session_state.thread_id,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "Sorry, I could not generate a response.")
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach the backend. Make sure the API server is running."
    except Exception as exc:
        return f"⚠️ Error: {exc}"


def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        resp = requests.post(
            f"{API_URL}/transcribe",
            files={"audio": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("transcript", "")
    except Exception as exc:
        st.warning(f"Transcription error: {exc}")
        return ""


def fetch_memories() -> list[dict]:
    try:
        resp = requests.get(f"{API_URL}/memories/{st.session_state.user_id}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("memories", [])
    except Exception:
        return []


def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI Memory Assistant")
    st.markdown("---")

    st.subheader("Session")
    st.code(f"User ID:\n{st.session_state.user_id[:8]}…", language=None)
    st.code(f"Thread:\n{st.session_state.thread_id[:8]}…", language=None)

    if st.button("➕ New Thread", use_container_width=True):
        new_thread()

    if st.button("🔄 Reset User ID (wipe memories)", use_container_width=True):
        st.session_state.user_id    = str(uuid.uuid4())
        st.session_state.thread_id  = str(uuid.uuid4())
        st.session_state.messages   = []
        st.session_state.memories   = []
        st.session_state.last_audio_hash = None
        st.rerun()

    st.markdown("---")
    st.subheader("💾 Stored Memories")
    if st.button("Refresh memories", use_container_width=True):
        st.session_state.memories = fetch_memories()

    if st.session_state.memories:
        for i, mem in enumerate(st.session_state.memories, 1):
            with st.expander(f"Memory {i}", expanded=False):
                val = mem.get("value", {})
                st.markdown(f"**Content:** {val.get('content', '')}")
                st.markdown(f"**Context:** {val.get('context', '')}")
    else:
        st.info("No memories yet. Start chatting!")

    st.markdown("---")
    st.caption("Powered by LangGraph · Groq · Whisper")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Chat")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Voice input ───────────────────────────────────────────────────────────────
st.markdown("#### 🎙️ Voice Input")
audio_input = st.audio_input("Record your message", key="audio_recorder")

if audio_input is not None:
    audio_bytes = audio_input.read()

    # Hash the audio so we only process each unique recording ONCE
    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    if audio_hash != st.session_state.last_audio_hash:
        # Mark as processed immediately before any rerun can happen
        st.session_state.last_audio_hash = audio_hash

        with st.spinner("Transcribing…"):
            transcript = transcribe_audio(audio_bytes)

        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.spinner("Thinking…"):
                reply = send_message(transcript)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
        else:
            st.warning("Could not transcribe audio. Try typing instead.")

# ── Text input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Type a message…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = send_message(user_input)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
