"""
Notes audio generation.

Two providers, switchable via NOTES_AUDIO_PROVIDER env var:
  NOTES_AUDIO_PROVIDER=gemini      (default) — Gemini TTS, near-zero cost
  NOTES_AUDIO_PROVIDER=elevenlabs  — ElevenLabs TTS, ~$0.10/1k chars

Both return an audio_url string.
"""

import io
import os
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from app.utils.files import upload_bytes_to_do

# ── ElevenLabs config ────────────────────────────────────────────────────────
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_CHUNK_SIZE = 9000
ELEVENLABS_MAX_CONCURRENCY = 3
RETRY_MAX = 3
RETRY_BASE_DELAY = 1.5

# ── Gemini TTS config ────────────────────────────────────────────────────────
GEMINI_TTS_CHUNK_SIZE = 2000
GEMINI_TTS_MAX_CONCURRENCY = 5
GEMINI_WAV_SAMPLE_RATE = 24000
GEMINI_WAV_SAMPLE_WIDTH = 2  # 16-bit PCM

_GEMINI_NOTES_NARRATION_PREFIX = (
    "Read the following educational note aloud as a clear, engaging teacher narration. "
    "Speak naturally and conversationally. You may rephrase slightly for better audio flow, "
    "but preserve all key facts, terms, and examples. "
    "Output AUDIO ONLY.\n\n"
)

# ── Markdown → plain text ─────────────────────────────────────────────────────

_BLOCK_SPOKEN_PREFIXES = {
    "formula":        "Formula.",
    "shortcut":       "Quick tip.",
    "pyq_alert":      "Previous year question alert.",
    "mistake":        "Common mistake.",
    "exam_insight":   "Exam insight.",
    "solved_example": "Solved example.",
}


def _strip_markdown_to_text(markdown: str) -> str:
    """Convert Extended Markdown to clean plain text suitable for TTS narration."""

    def _replace_block(m: re.Match) -> str:
        block_type = m.group(1).strip().lower()
        content = m.group(2).strip()
        if block_type == "mcq":
            return ""  # interactive-only — skip from audio entirely
        prefix = _BLOCK_SPOKEN_PREFIXES.get(block_type, "")
        return f"{prefix} {content}" if prefix else content

    text = re.sub(
        r"^:::(\w+)[^\n]*\n(.*?)^:::\s*$",
        _replace_block,
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Replace GFM tables with a spoken placeholder
    text = re.sub(r"(?:^\|[^\n]*(?:\n|$))+", "Refer to the table in the notes. ", text, flags=re.MULTILINE)

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text_chunks(text: str, chunk_size: int) -> list[str]:
    """Split text at sentence boundaries into chunks of at most chunk_size chars."""
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind(". ", 0, chunk_size)
        if split_at == -1:
            split_at = text.rfind("\n", 0, chunk_size)
        if split_at == -1:
            split_at = text.rfind(" ", 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[: split_at + 1].strip())
        text = text[split_at + 1:].strip()
    if text:
        chunks.append(text)
    return chunks


# ── ElevenLabs provider ──────────────────────────────────────────────────────

def _elevenlabs_generate_chunk(chunk_text: str, chunk_idx: int, total_chunks: int) -> bytes:
    """Call ElevenLabs for one chunk. Returns mp3_bytes."""
    from elevenlabs.client import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    client = ElevenLabs(api_key=api_key)
    last_exc: Exception | None = None

    for attempt in range(1, RETRY_MAX + 1):
        try:
            audio_bytes = b"".join(client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=chunk_text,
                model_id=ELEVENLABS_MODEL,
                output_format="mp3_44100_128",
            ))
            logger.info(f"ElevenLabs chunk {chunk_idx + 1}/{total_chunks} done — {len(audio_bytes)} bytes")
            return audio_bytes

        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(f"ElevenLabs chunk {chunk_idx + 1} non-retryable error: {exc}") from exc
        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_MAX:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"ElevenLabs chunk {chunk_idx + 1} attempt {attempt} failed: {exc}. Retry in {delay:.1f}s")
                time.sleep(delay)

    raise RuntimeError(f"ElevenLabs chunk {chunk_idx + 1} failed after {RETRY_MAX} attempts: {last_exc}")


def _generate_with_elevenlabs(content: str, note_id: int, language: str) -> str:
    plain_text = _strip_markdown_to_text(content)
    if not plain_text:
        raise ValueError("Note content is empty after stripping markdown")

    chunks = _split_text_chunks(plain_text, ELEVENLABS_CHUNK_SIZE)
    logger.info(f"[elevenlabs] note={note_id} chunks={len(chunks)} chars={len(plain_text)} lang={language}")

    results: dict[int, bytes] = {}
    with ThreadPoolExecutor(max_workers=ELEVENLABS_MAX_CONCURRENCY) as executor:
        futures = {executor.submit(_elevenlabs_generate_chunk, chunk, i, len(chunks)): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    combined_mp3 = b"".join(results[i] for i in range(len(chunks)))
    audio_url = upload_bytes_to_do(combined_mp3, "audio.mp3", f"comp/notes/audio/{note_id}", content_type="audio/mpeg")
    logger.info(f"[elevenlabs] note={note_id} uploaded → {audio_url}")
    return audio_url


# ── Gemini TTS provider ──────────────────────────────────────────────────────

def _gemini_notes_pcm_chunk(chunk: str, chunk_idx: int, total_chunks: int) -> bytes:
    import os as _os
    import time as _time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_os.getenv("GOOGLE_API_KEY"))
    prompt = _GEMINI_NOTES_NARRATION_PREFIX + chunk

    for attempt in range(1, 6):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                        )
                    ),
                ),
            )
            candidate = response.candidates[0] if response.candidates else None
            if candidate and candidate.content and candidate.content.parts:
                pcm = candidate.content.parts[0].inline_data.data
                if pcm:
                    return pcm
            if attempt < 5:
                _time.sleep(1.5 * (2 ** (attempt - 1)))
        except Exception as exc:
            if attempt < 5:
                _time.sleep(1.5 * (2 ** (attempt - 1)))
                logger.warning(f"[gemini-tts] chunk {chunk_idx + 1} attempt {attempt} failed: {exc}")
            else:
                raise
    raise ValueError(f"Gemini TTS failed for chunk {chunk_idx + 1}")


def _generate_with_gemini(content: str, note_id: int, language: str) -> str:
    from app.services.audio_generation import _split_text_for_tts

    plain_text = _strip_markdown_to_text(content)
    if not plain_text:
        raise ValueError("Note content is empty after stripping markdown")

    chunks = _split_text_for_tts(plain_text, GEMINI_TTS_CHUNK_SIZE)
    logger.info(f"[gemini-tts] note={note_id} chunks={len(chunks)} chars={len(plain_text)} lang={language}")

    indexed_pcm: dict[int, bytes] = {}
    with ThreadPoolExecutor(max_workers=GEMINI_TTS_MAX_CONCURRENCY) as executor:
        futures = {executor.submit(_gemini_notes_pcm_chunk, chunk, i, len(chunks)): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            indexed_pcm[futures[future]] = future.result()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(GEMINI_WAV_SAMPLE_WIDTH)
        wf.setframerate(GEMINI_WAV_SAMPLE_RATE)
        for i in range(len(chunks)):
            wf.writeframes(indexed_pcm[i])
    wav_bytes = buf.getvalue()

    audio_url = upload_bytes_to_do(wav_bytes, "audio.wav", f"comp/notes/audio/{note_id}", content_type="audio/wav")
    logger.info(f"[gemini-tts] note={note_id} uploaded → {audio_url}")
    return audio_url


# ── Public API ────────────────────────────────────────────────────────────────

def generate_notes_audio(
    content: str,
    note_id: int,
    language: str = "en",
) -> str:
    """
    Generate audio for a comp note.

    Provider is controlled by NOTES_AUDIO_PROVIDER env var:
      "gemini"      — Gemini TTS (default, near-zero cost)
      "elevenlabs"  — ElevenLabs TTS (~$0.10/1k chars)

    Returns: audio_url
    """
    provider = os.getenv("NOTES_AUDIO_PROVIDER", "gemini").lower()
    logger.info(f"[notes-audio] provider={provider} note={note_id}")
    if provider == "elevenlabs":
        return _generate_with_elevenlabs(content, note_id, language)
    return _generate_with_gemini(content, note_id, language)
