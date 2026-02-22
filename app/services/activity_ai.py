import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, conlist

from app.core.agents.graph import merge_chunks_remove_overlap, vector_store_textbooks


MAX_TOPIC_CHARS = 12000
MAX_ACTIVITY_CHUNK_CHARS = 12000


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty AI response")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in AI response")
    return json.loads(match.group(0))


# --------------------
# LLM helpers
# --------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        streaming=False,
    )


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# --------------------
# Context fetchers
# --------------------

def get_full_chapter_text(chapter_id: int) -> str:
    from app.core.config import settings
    import psycopg

    logger.debug(f"[chapter={chapter_id}] Fetching full chapter text from DB")
    try:
        postgres_url = settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")

        with psycopg.connect(postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document, cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection WHERE name = 'llm_textbooks'
                    )
                    AND cmetadata->>'chapter_id' = %s
                    ORDER BY (cmetadata->>'chunk_index')::int NULLS LAST
                    """,
                    (str(chapter_id),),
                )
                rows = cur.fetchall()

                if not rows:
                    logger.warning(f"[chapter={chapter_id}] No embeddings found in llm_textbooks — chapter has no uploaded content")
                    return ""

                raw_chunks = [row[0] for row in rows]
                text = merge_chunks_remove_overlap(raw_chunks, overlap_chars=200)
                logger.debug(f"[chapter={chapter_id}] Fetched {len(rows)} chunks, merged text length={len(text)}")
                return text
    except Exception as e:
        logger.error(f"[chapter={chapter_id}] Error fetching chapter text from DB: {e} — falling back to similarity search")
        docs = vector_store_textbooks.similarity_search(
            query="",
            k=1000,
            filter={"chapter_id": str(chapter_id)},
        )
        if not docs:
            logger.warning(f"[chapter={chapter_id}] Fallback similarity search also returned no docs")
            return ""
        docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        raw_chunks = [d.page_content for d in docs]
        return merge_chunks_remove_overlap(raw_chunks, overlap_chars=200)



def get_qa_context(chapter_id: int) -> str:
    """Fetch all Q&A pattern content for a chapter directly from the DB."""
    from app.core.config import settings
    import psycopg

    try:
        postgres_url = settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document
                    FROM langchain_pg_embedding
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection WHERE name = 'qa_patterns'
                    )
                    AND cmetadata->>'chapter_id' = %s
                    ORDER BY (cmetadata->>'chunk_index')::int NULLS LAST
                    """,
                    (str(chapter_id),),
                )
                rows = cur.fetchall()
                if not rows:
                    logger.debug(f"[chapter={chapter_id}] No QA patterns found")
                    return ""
                logger.debug(f"[chapter={chapter_id}] Fetched {len(rows)} QA pattern chunks")
                return "\n\n".join([row[0] for row in rows])
    except Exception as e:
        logger.warning(f"[chapter={chapter_id}] Failed to fetch QA patterns: {e}")
        return ""


# --------------------
# Topic generation
# --------------------

def _language_instruction(medium_name: str, context: str = "topic titles and summaries") -> str:
    medium_lower = medium_name.lower()
    if any(
        lang in medium_lower
        for lang in [
            "english", "hindi", "kannada", "malayalam", "tamil",
            "telugu", "sanskrit", "urdu", "bengali", "marathi", "gujarati",
        ]
    ):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        return (
            f"\nIMPORTANT: This is {lang_name} medium. Generate all {context} "
            f"in {lang_name} language only."
        )
    return ""


def _generate_topics_from_text(
    text: str, medium_name: str = ""
) -> List[Dict[str, str]]:
    human_prompt = (
        "Analyze the provided chapter text and extract a comprehensive yet concise list of key topics "
        "that represent the entire scope of the content.\n\n"
        "Requirements:\n"
        "- Optimized Topic Count: Extract between 6 to 15 topics depending on the chapter's length and complexity. "
        "Avoid over-segmentation — if two concepts are closely related, group them under a single broader heading.\n"
        "- Complete Coverage: Ensure that every concept, definition, and principle mentioned "
        "in the chapter is mapped to one of the topics. No part of the chapter should be left out.\n"
        "- Conceptual Pillars: Each topic name must be a significant 'Conceptual Pillar' rather than a minor detail.\n"
        "- Exhaustive Logic: The final list must be structured so that 'Important Questions' generated for these "
        "topics will collectively cover the entire chapter without any gaps.\n\n"
        "Keep titles short and summaries 1 sentence.\n"
        'Return JSON in this schema only: { "topics": [ { "title": "...", "summary": "..." } ] }\n'
        f"{_language_instruction(medium_name)}\n\n"
        f"MEDIUM: {medium_name}\n"
        f"CHAPTER TEXT:\n{text}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content="Act as an expert Curriculum Designer. Your output must be strict JSON only — no markdown, no prose, no extra text."),
            HumanMessage(content=human_prompt),
        ]
    )
    payload = _extract_json(response.content)
    topics = payload.get("topics", [])
    return topics if isinstance(topics, list) else []


def _consolidate_topics(
    topics: List[Dict[str, str]],
    medium_name: str = "",
) -> List[Dict[str, str]]:
    human_prompt = (
        "Given the topic candidates below extracted from different parts of a chapter, "
        "deduplicate and consolidate them into the final comprehensive topic list.\n\n"
        "Requirements:\n"
        "- Optimized Topic Count: Return between 6 to 15 topics depending on the chapter's breadth. "
        "Avoid over-segmentation — merge closely related subtopics under a single broader heading.\n"
        "- Complete Coverage: Preserve all distinct concepts — do not drop topics that represent unique content.\n"
        "- Conceptual Pillars: Each topic name must be a significant 'Conceptual Pillar' rather than a minor detail.\n"
        "- Deduplication: Merge topics that refer to the same concept into one well-named topic.\n"
        "- Exhaustive Logic: The final list must be thorough enough that generating 'Important Questions' "
        "for each topic would cover the entire chapter without gaps.\n\n"
        'Return JSON in this schema only: { "topics": [ { "title": "...", "summary": "..." } ] }\n'
        f"{_language_instruction(medium_name)}\n\n"
        f"MEDIUM: {medium_name}\n\n"
        f"TOPICS:\n{json.dumps(topics)}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content="Act as an expert Curriculum Designer. Your output must be strict JSON only — no markdown, no prose, no extra text."),
            HumanMessage(content=human_prompt),
        ]
    )
    payload = _extract_json(response.content)
    topics = payload.get("topics", [])
    return topics if isinstance(topics, list) else []


def generate_topics(chapter_id: int, medium_name: str = "") -> List[Dict[str, str]]:
    logger.info(f"[chapter={chapter_id}] Starting topic generation (medium={medium_name!r})")
    chapter_text = get_full_chapter_text(chapter_id)
    if not chapter_text:
        logger.warning(f"[chapter={chapter_id}] No chapter text — skipping topic generation")
        return []

    logger.info(f"[chapter={chapter_id}] Chapter text length={len(chapter_text)}")
    all_topics: List[Dict[str, str]] = []
    if len(chapter_text) <= MAX_TOPIC_CHARS:
        logger.debug(f"[chapter={chapter_id}] Single chunk topic extraction")
        all_topics.extend(_generate_topics_from_text(chapter_text, medium_name))
        logger.info(f"[chapter={chapter_id}] Topics before consolidation: {len(all_topics)}")
        final = _consolidate_topics(all_topics, medium_name=medium_name)
        logger.info(f"[chapter={chapter_id}] Final topics after consolidation: {len(final)}")
        return final

    chunks = _split_text(chapter_text, chunk_size=MAX_TOPIC_CHARS, chunk_overlap=400)
    logger.info(f"[chapter={chapter_id}] Chapter too large, split into {len(chunks)} chunks for parallel extraction")
    with ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as executor:
        futures = {
            executor.submit(_generate_topics_from_text, chunk, medium_name): i
            for i, chunk in enumerate(chunks)
        }
        chunk_results: dict[int, list] = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                chunk_results[idx] = future.result()
                logger.debug(f"[chapter={chapter_id}] Chunk {idx} returned {len(chunk_results[idx])} topics")
            except Exception as exc:
                logger.warning(f"[chapter={chapter_id}] Topic extraction failed for chunk {idx}: {exc}")
                chunk_results[idx] = []
        for i in range(len(chunks)):
            all_topics.extend(chunk_results.get(i, []))

    logger.info(f"[chapter={chapter_id}] Total topics before consolidation: {len(all_topics)}")
    if not all_topics:
        return []
    final = _consolidate_topics(all_topics, medium_name=medium_name)
    logger.info(f"[chapter={chapter_id}] Final topics after consolidation: {len(final)}")
    return final


# --------------------
# Activity generation — Pydantic structured output
# --------------------

class _MCQActivity(BaseModel):
    question_text: str
    options: conlist(str, min_length=4, max_length=4)
    correct_answer: str
    answer_description: str

class _DescriptiveActivity(BaseModel):
    question_text: str
    answer_text: str

class _TopicActivities(BaseModel):
    topic: str
    mcqs: List[_MCQActivity]
    descriptives: List[_DescriptiveActivity]

class _ActivityOutput(BaseModel):
    topics: List[_TopicActivities]


def generate_activities(
    chapter_id: int,
    topic_titles: List[str],
    mcq_count: int,
    descriptive_count: int,
    medium_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Generate activities for all topic titles in a single structured LLM call.
    Returns a flat list of activities, each with a 'topic' field.
    """
    logger.info(
        f"[chapter={chapter_id}] generate_activities: topics={topic_titles}, "
        f"mcq={mcq_count}, descriptive={descriptive_count}, medium={medium_name!r}"
    )
    chapter_text = get_full_chapter_text(chapter_id)
    if not chapter_text:
        logger.warning(f"[chapter={chapter_id}] Empty chapter text — cannot generate activities")
        return []
    logger.info(f"[chapter={chapter_id}] Chapter text length={len(chapter_text)}")

    qa_context = get_qa_context(chapter_id)
    topics_str = "\n".join(f"- {t}" for t in topic_titles)

    language_instruction = ""
    medium_lower = medium_name.lower()
    if any(
        lang in medium_lower
        for lang in [
            "kannada", "malayalam", "tamil", "telugu", "hindi",
            "sanskrit", "urdu", "bengali", "marathi", "gujarati",
        ]
    ):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = (
            f"\n\nCRITICAL: This is {lang_name} medium. "
            f"ALL questions, options, and answers MUST be in {lang_name} language."
        )

    human_prompt = (
        f"Generate exactly {mcq_count} MCQ questions and exactly {descriptive_count} descriptive questions "
        f"for EACH of the following topics, based on the chapter text below.\n\n"
        "Core Constraints:\n"
        "- Generate questions for EVERY topic listed — do not skip any.\n"
        "- For each topic: exactly {mcq_count} MCQs and exactly {descriptive_count} descriptive questions.\n"
        "- MCQ: provide 4 plausible options, correct_answer must be the EXACT text of one option.\n"
        "- MCQ: include answer_description (1-2 sentence explanation of why the correct answer is right).\n"
        "- Descriptive: include a full model answer_text.\n"
        "- Cognitive Depth: distribute across Bloom's Taxonomy (Recall, Understanding, Application).\n"
        f"{language_instruction}\n\n"
        f"MEDIUM: {medium_name}\n\n"
        f"TOPICS:\n{topics_str}\n\n"
        + (
            f"IMPORTANT Q&A (Previous Year / Must-Study):\n"
            f"Incorporate these high-priority questions first where relevant.\n{qa_context}\n\n"
            if qa_context else ""
        )
        + f"CHAPTER TEXT:\n{chapter_text}"
    )

    logger.info(f"[chapter={chapter_id}] Calling LLM with structured output (prompt_len={len(human_prompt)})")
    llm = _get_llm().with_structured_output(_ActivityOutput)
    try:
        result: _ActivityOutput = llm.invoke(
            [
                SystemMessage(content="You are an expert Pedagogy & Assessment Design AI."),
                HumanMessage(content=human_prompt),
            ]
        )
    except Exception as exc:
        logger.error(f"[chapter={chapter_id}] Structured output LLM call failed: {exc}")
        return []

    all_activities: List[Dict[str, Any]] = []
    for topic_block in result.topics:
        for mcq in topic_block.mcqs:
            all_activities.append({
                "type": "mcq",
                "topic": topic_block.topic,
                "question_text": mcq.question_text,
                "options": list(mcq.options),
                "correct_answer": mcq.correct_answer,
                "answer_description": mcq.answer_description,
            })
        for desc in topic_block.descriptives:
            all_activities.append({
                "type": "descriptive",
                "topic": topic_block.topic,
                "question_text": desc.question_text,
                "answer_text": desc.answer_text,
            })

    logger.info(f"[chapter={chapter_id}] Structured output returned {len(all_activities)} total activities across {len(result.topics)} topics")
    return all_activities


# --------------------
# Chapter summary (free text — no structured output)
# --------------------

MAX_SUMMARY_CHARS = 20000


def generate_chapter_summary(chapter_id: int, medium_name: str = "") -> str:
    """
    Generate a structured chapter summary from the full chapter text.
    Returns a markdown-formatted summary string.
    Called once during textbook processing and stored as a ChapterArtifact.
    """
    chapter_text = get_full_chapter_text(chapter_id)
    if not chapter_text:
        return ""

    text = chapter_text[:MAX_SUMMARY_CHARS]

    language_instruction = ""
    medium_lower = medium_name.lower()
    if any(
        lang in medium_lower
        for lang in [
            "hindi", "kannada", "malayalam", "tamil", "telugu",
            "sanskrit", "urdu", "bengali", "marathi", "gujarati",
        ]
    ):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = f"\nIMPORTANT: This is {lang_name} medium. Write the entire summary in {lang_name} language."

    human_prompt = (
        "Based on the chapter text below, write a comprehensive chapter summary.\n\n"
        "Structure:\n"
        "1. **What this chapter is about** — 2-3 sentences overview\n"
        "2. **Key Topics** — bullet list of main topics covered\n"
        "3. **Important Concepts & Definitions** — key terms defined briefly\n"
        "4. **Key Takeaways** — 3-5 most important points a student must remember\n\n"
        "Write in clear, student-friendly language. Use markdown formatting."
        f"{language_instruction}\n\n"
        f"MEDIUM: {medium_name}\n\n"
        f"CHAPTER TEXT:\n{text}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert Curriculum Designer. Generate a clear, structured chapter summary for students."),
            HumanMessage(content=human_prompt),
        ]
    )
    return response.content.strip()


# --------------------
# Activity normalization
# --------------------

def normalize_activity(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    activity_type = str(item.get("type", "")).strip().lower()
    question_text = str(item.get("question_text", "")).strip()

    if not activity_type or not question_text:
        return None

    if activity_type == "mcq":
        options = item.get("options", [])
        if not isinstance(options, list) or len(options) != 4:
            return None
        cleaned_options = [str(opt).strip() for opt in options]
        if any(not opt for opt in cleaned_options):
            return None
        correct_answer = str(item.get("correct_answer", "")).strip()
        try:
            correct_option_index = cleaned_options.index(correct_answer) + 1
        except ValueError:
            lower_options = [o.lower() for o in cleaned_options]
            try:
                correct_option_index = lower_options.index(correct_answer.lower()) + 1
            except ValueError:
                return None
        answer_description = str(item.get("answer_description", "")).strip() or None
        return {
            "type": "mcq",
            "question_text": question_text,
            "options": cleaned_options,
            "correct_option_index": int(correct_option_index),
            "answer_text": None,
            "answer_description": answer_description,
        }

    if activity_type == "descriptive":
        answer_text = str(item.get("answer_text", "")).strip()
        if not answer_text:
            return None
        return {
            "type": "descriptive",
            "question_text": question_text,
            "options": None,
            "correct_option_index": None,
            "answer_text": answer_text,
        }

    return None


# --------------------
# Descriptive answer evaluation
# --------------------

def evaluate_descriptive_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
    medium_name: str = "",
) -> Dict[str, Any]:
    """
    Evaluate a descriptive answer using AI.

    Returns:
        {
            "score": int (0-100),
            "feedback": List[str] (3-4 bullet points)
        }
    """
    language_instruction = ""
    medium_lower = medium_name.lower()
    if any(
        lang in medium_lower
        for lang in [
            "kannada", "malayalam", "tamil", "telugu", "hindi",
            "sanskrit", "urdu", "bengali", "marathi", "gujarati",
        ]
    ):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = f"\n\nIMPORTANT: Generate feedback in {lang_name} language since this is {lang_name} medium."

    human_prompt = (
        "Evaluate the student's answer compared to the correct answer. "
        'Return JSON in this schema: { "score": 75, "feedback": ["Good: ...", "Improve: ..."] }\n\n'
        "Requirements:\n"
        "- score: 0-100 based on correctness, completeness, and clarity\n"
        "- feedback: Exactly 3-4 bullet points\n"
        "- Start each feedback point with 'Good:' or 'Improve:'\n"
        "- Be specific and constructive\n"
        f"{language_instruction}\n\n"
        f"MEDIUM: {medium_name}\n"
        f"QUESTION: {question}\n\n"
        f"CORRECT ANSWER:\n{correct_answer}\n\n"
        f"STUDENT'S ANSWER:\n{user_answer}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content="You are an assistant that outputs strict JSON only. Do not include markdown or extra text."),
            HumanMessage(content=human_prompt),
        ]
    )
    payload = _extract_json(response.content)
    score = max(0, min(100, int(payload.get("score", 0))))
    feedback = payload.get("feedback", [])
    if not isinstance(feedback, list):
        feedback = []
    return {
        "score": score,
        "feedback": feedback,
    }
