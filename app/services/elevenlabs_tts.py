"""
ElevenLabs TTS service for comp notes audio generation with word-level timestamps.

Flow:
  1. Strip Extended Markdown to plain narration text
  2. Split into ≤9000-char chunks at sentence boundaries
  3. Call ElevenLabs convert_with_timestamps for each chunk (parallel, max 3)
  4. Merge character alignment arrays → word list with global time offsets
  5. Concatenate MP3 chunks, upload to DigitalOcean
  6. Return (audio_url, audio_sync_json_str)

Sync JSON format stored in DB:
  {"words": [{"w": "Pressure", "s": 0.12, "e": 0.45}, ...], "duration_sec": 1680.5}
"""

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from app.utils.files import upload_bytes_to_do

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL = "eleven_multilingual_v2"
CHUNK_SIZE = 9000
MAX_CONCURRENCY = 3
RETRY_MAX = 3
RETRY_BASE_DELAY = 1.5


def _strip_markdown_to_text(markdown: str) -> str:
    """Convert Extended Markdown to clean plain text suitable for TTS narration."""
    text = markdown

    # Remove image tags entirely
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove :::type block markers and ::: closing markers (keep inner text)
    text = re.sub(r"^:::\w+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^:::\s*$", "", text, flags=re.MULTILINE)

    # Remove heading markers (#, ##, ###) but keep heading text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold and italic markers (keep text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    # Remove table separator rows (|---|---|)
    text = re.sub(r"^\|[\s\-\|:]+\|\s*$", "", text, flags=re.MULTILINE)

    # Remove table cell pipes, keep cell text
    text = re.sub(r"\|", " ", text)

    # Remove MCQ option JSON arrays (they'd be read as literal JSON otherwise)
    text = re.sub(r'^options:\s*\[.*?\]$', "", text, flags=re.MULTILINE)

    # Remove markdown links, keep display text
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)

    # Remove inline code backticks
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Collapse multiple blank lines to single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_text_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks at sentence boundaries, each ≤ chunk_size chars."""
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
        text = text[split_at + 1 :].strip()
    if text:
        chunks.append(text)
    return chunks


def _chars_to_words(
    characters: list[str],
    start_times: list[float],
    end_times: list[float],
    time_offset: float = 0.0,
) -> list[dict]:
    """Convert parallel character arrays into a word list with absolute timestamps."""
    words = []
    current_chars: list[str] = []
    current_start: float | None = None

    for ch, st, et in zip(characters, start_times, end_times):
        if ch in (" ", "\n", "\t", "\r"):
            if current_chars:
                words.append({
                    "w": "".join(current_chars),
                    "s": round(current_start + time_offset, 3),
                    "e": round(et + time_offset, 3),
                })
                current_chars = []
                current_start = None
        else:
            if current_start is None:
                current_start = st
            current_chars.append(ch)

    if current_chars and current_start is not None:
        words.append({
            "w": "".join(current_chars),
            "s": round(current_start + time_offset, 3),
            "e": round(end_times[-1] + time_offset, 3),
        })

    return words


def _generate_chunk(chunk_text: str, chunk_idx: int, total_chunks: int) -> tuple[bytes, list[dict], float]:
    """
    Call ElevenLabs for one text chunk.
    Returns (mp3_bytes, word_list_no_offset, chunk_duration_sec).
    """
    from elevenlabs.client import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    client = ElevenLabs(api_key=api_key)
    last_exc: Exception | None = None

    for attempt in range(1, RETRY_MAX + 1):
        try:
            response = client.text_to_speech.convert_with_timestamps(
                voice_id=ELEVENLABS_VOICE_ID,
                text=chunk_text,
                model_id=ELEVENLABS_MODEL,
                output_format="mp3_44100_128",
            )

            mp3_bytes = base64.b64decode(response.audio_base64)
            alignment = response.alignment

            words = _chars_to_words(
                alignment.characters,
                alignment.character_start_times_seconds,
                alignment.character_end_times_seconds,
            )
            duration = alignment.character_end_times_seconds[-1] if alignment.character_end_times_seconds else 0.0
            logger.info(f"ElevenLabs chunk {chunk_idx + 1}/{total_chunks} done — {len(words)} words, {duration:.1f}s")
            return mp3_bytes, words, duration

        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_MAX:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"ElevenLabs chunk {chunk_idx + 1} attempt {attempt} failed: {exc}. Retry in {delay:.1f}s")
                time.sleep(delay)

    raise RuntimeError(f"ElevenLabs chunk {chunk_idx + 1} failed after {RETRY_MAX} attempts: {last_exc}")


def generate_notes_audio_with_sync(
    content: str,
    note_id: int,
    language: str = "en",
) -> tuple[str, str]:
    """
    Generate MP3 audio + word-level sync JSON for a comp note.

    Args:
        content: Full Extended Markdown content of the note.
        note_id: Used for DigitalOcean storage path.
        language: Note language (passed for logging; ElevenLabs multilingual handles it).

    Returns:
        (audio_url, audio_sync_json_str)
        audio_sync_json_str is a JSON string: {"words": [...], "duration_sec": N}
    """
    plain_text = _strip_markdown_to_text(content)
    if not plain_text:
        raise ValueError("Note content is empty after stripping markdown")

    chunks = _split_text_chunks(plain_text)
    logger.info(f"Note {note_id}: generating audio for {len(chunks)} chunks ({len(plain_text)} chars, lang={language})")

    # Run chunks in parallel, preserve order via index
    results: dict[int, tuple[bytes, list[dict], float]] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = {
            executor.submit(_generate_chunk, chunk, i, len(chunks)): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Merge in order: apply cumulative time offset to each chunk's words
    all_mp3_parts: list[bytes] = []
    all_words: list[dict] = []
    cumulative_offset = 0.0

    for i in range(len(chunks)):
        mp3_bytes, words, duration = results[i]
        all_mp3_parts.append(mp3_bytes)
        for word in words:
            all_words.append({
                "w": word["w"],
                "s": round(word["s"] + cumulative_offset, 3),
                "e": round(word["e"] + cumulative_offset, 3),
            })
        cumulative_offset += duration

    combined_mp3 = b"".join(all_mp3_parts)
    sync_dict = {"words": all_words, "duration_sec": round(cumulative_offset, 2)}

    # Upload MP3 to DigitalOcean
    do_path = f"comp/notes/audio/{note_id}"
    audio_url = upload_bytes_to_do(combined_mp3, "audio.mp3", do_path, content_type="audio/mpeg")
    logger.info(f"Note {note_id}: audio uploaded → {audio_url} ({len(all_words)} words, {cumulative_offset:.1f}s)")

    return audio_url, json.dumps(sync_dict)
