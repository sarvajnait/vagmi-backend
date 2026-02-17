"""
Audio generation service using Gemini 2.5 Flash TTS via google-genai SDK.

Uses the same GOOGLE_API_KEY already configured for the rest of the app.

Flow:
  1. Load PDF via PyPDFLoader
  2. Convert Kannada legacy text
  3. Split into TTS-safe chunks (~3000 chars, sentence-boundary aware)
  4. Call gemini-2.5-pro-preview-tts for each chunk
  5. Concatenate raw PCM and wrap in a single WAV file
  6. Upload final audio to DigitalOcean Spaces
  7. Return the public audio URL
"""

import io
import os
import re
import time
import uuid
import wave

import boto3
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from app.utils.kannada_converter import convert_kannada_text

DO_REGION = "blr1"
DO_BUCKET = "vagmi"
DO_ENDPOINT = f"https://{DO_REGION}.digitaloceanspaces.com"

TTS_CHUNK_SIZE = 2000  # chars — keep under 4000 byte API limit (style prefix adds ~200 chars)
TTS_MODEL = "gemini-2.5-pro-preview-tts"
TTS_VOICE = "Kore"
WAV_SAMPLE_RATE = 24000
WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2  # 16-bit PCM
TTS_RETRY_MAX_ATTEMPTS = 4
TTS_RETRY_BASE_DELAY_SEC = 1.5

TTS_STYLE_PREFIX = (
    "You are an experienced teacher narrating educational content for students. "
    "Output AUDIO ONLY (no text). Do not display or return any written content. "
    "First, silently remove non-content artifacts such as headers, footers, page numbers, "
    "watermarks, repeated titles, or other noise from the text. "
    "Then rewrite the content so it sounds natural and engaging: keep the core meaning, "
    "but do not stay verbatim. Add short transitions, clarifying phrases, and gentle, "
    "positive tone uplift where appropriate. Keep it concise, accurate, and suitable for "
    "spoken audio without adding new factual claims beyond the source.\n\n"
)

_HTTP_5XX_PATTERN = re.compile(r"\b5\d{2}\b")


def _is_retryable_tts_error(exc: Exception) -> bool:
    """Retry only transient provider-side TTS failures (HTTP 5xx)."""
    message = str(exc)
    return bool(_HTTP_5XX_PATTERN.search(message))


def _split_text_for_tts(text: str, chunk_size: int = TTS_CHUNK_SIZE) -> list[str]:
    """Split long text into chunks that fit within TTS input limits."""
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind(". ", 0, chunk_size)
        if split_at == -1:
            split_at = text.rfind(" ", 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[: split_at + 1].strip())
        text = text[split_at + 1 :].strip()
    if text:
        chunks.append(text)
    return chunks


def _pcm_chunks_to_wav(pcm_chunks: list[bytes]) -> bytes:
    """Combine multiple raw PCM chunks into a single WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(WAV_CHANNELS)
        wf.setsampwidth(WAV_SAMPLE_WIDTH)
        wf.setframerate(WAV_SAMPLE_RATE)
        for pcm in pcm_chunks:
            wf.writeframes(pcm)
    return buf.getvalue()


def _upload_audio_to_do(audio_bytes: bytes, do_path: str, filename: str, content_type: str = "audio/wav") -> str:
    """Upload audio bytes to DigitalOcean Spaces and return public URL."""
    access_key = os.getenv("DO_SPACES_ACCESS_KEY")
    secret_key = os.getenv("DO_SPACES_SECRET_KEY")

    s3_client = boto3.client(
        "s3",
        region_name=DO_REGION,
        endpoint_url=DO_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    object_key = f"{do_path}/{uuid.uuid4().hex}_{filename}"
    s3_client.upload_fileobj(
        io.BytesIO(audio_bytes),
        DO_BUCKET,
        object_key,
        ExtraArgs={"ACL": "public-read", "ContentType": content_type},
    )
    return f"{DO_ENDPOINT}/{DO_BUCKET}/{object_key}"


def generate_audio_from_pdf(
    file_url: str,
    resource_type: str,
    resource_id: int,
    chapter_id: int,
) -> str:
    """
    Generate an audiobook from a PDF file using Gemini TTS.

    Args:
        file_url: Public URL of the PDF in DigitalOcean Spaces.
        resource_type: "textbook" or "notes" (used for storage path).
        resource_id: ID of the StudentTextbook or StudentNotes record.
        chapter_id: Parent chapter ID (used for storage path).

    Returns:
        Public URL of the generated audio file.
    """
    # 1. Load and extract text from PDF
    loader = PyPDFLoader(file_url)
    pages = loader.load()

    # 2. Convert Kannada legacy encoding
    for page in pages:
        page.page_content = convert_kannada_text(page.page_content)

    full_text = "\n\n".join(
        page.page_content.strip() for page in pages if page.page_content.strip()
    )

    if not full_text:
        raise ValueError("PDF contains no extractable text for audio generation.")

    # 3. Split into TTS-safe chunks
    chunks = _split_text_for_tts(full_text)
    logger.info(
        f"Generating audio for {resource_type} id={resource_id}: "
        f"{len(chunks)} chunks from {len(pages)} pages"
    )

    # 4. TTS each chunk via google-genai SDK (uses GOOGLE_API_KEY)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    pcm_parts: list[bytes] = []

    for i, chunk in enumerate(chunks):
        logger.debug(f"TTS chunk {i + 1}/{len(chunks)} ({len(chunk)} chars): {repr(chunk[:200])}")
        prompt = TTS_STYLE_PREFIX + chunk
        response = None
        for attempt in range(1, TTS_RETRY_MAX_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=TTS_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=TTS_VOICE,
                                )
                            )
                        )
                    ),
                )
                break
            except Exception as exc:
                should_retry = _is_retryable_tts_error(exc) and attempt < TTS_RETRY_MAX_ATTEMPTS
                if not should_retry:
                    raise
                delay = TTS_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"TTS chunk {i + 1}/{len(chunks)} failed with retryable error "
                    f"(attempt {attempt}/{TTS_RETRY_MAX_ATTEMPTS}): {exc}. "
                    f"Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        if response is None:
            raise ValueError(f"TTS failed with no response for chunk {i + 1}")
        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
            finish = candidate.finish_reason if candidate else "no candidate"
            raise ValueError(f"TTS returned no content for chunk {i + 1} (finish_reason={finish})")
        pcm_data = candidate.content.parts[0].inline_data.data
        if not pcm_data:
            raise ValueError(f"TTS returned empty audio for chunk {i + 1}")
        pcm_parts.append(pcm_data)

    # 5. Combine PCM chunks into a single WAV file
    wav_bytes = _pcm_chunks_to_wav(pcm_parts)

    # 6. Upload to DigitalOcean
    do_path = f"chapters/{chapter_id}/student-content/audio/{resource_type}"
    filename = f"{resource_type}_{resource_id}.wav"
    audio_url = _upload_audio_to_do(wav_bytes, do_path, filename, content_type="audio/wav")

    logger.info(f"Audio generated and uploaded: {audio_url}")
    return audio_url
