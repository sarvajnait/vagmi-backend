from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Union
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel

from app.core.agents.graph import merge_chunks_remove_overlap, vector_store_textbooks


MAX_TOPIC_CHARS = 12000
MAX_CONTEXT_CHARS = 15000


# --------------------
# Pydantic output schemas
# --------------------

class TopicItem(BaseModel):
    title: str
    summary: str

class TopicList(BaseModel):
    topics: List[TopicItem]


class MCQActivity(BaseModel):
    type: Literal["mcq"]
    question_text: str
    options: List[str]
    correct_answer: str
    answer_description: Optional[str] = None

class DescriptiveActivity(BaseModel):
    type: Literal["descriptive"]
    question_text: str
    answer_text: str

class ActivityList(BaseModel):
    activities: List[Union[MCQActivity, DescriptiveActivity]]


class EvaluationResult(BaseModel):
    score: int
    feedback: List[str]


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
                    return ""

                raw_chunks = [row[0] for row in rows]
                return merge_chunks_remove_overlap(raw_chunks, overlap_chars=200)
    except Exception as e:
        logger.error(f"Error fetching chapter text directly: {e}")
        docs = vector_store_textbooks.similarity_search(
            query="",
            k=1000,
            filter={"chapter_id": str(chapter_id)},
        )
        if not docs:
            return ""
        docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        raw_chunks = [d.page_content for d in docs]
        return merge_chunks_remove_overlap(raw_chunks, overlap_chars=200)


def get_topic_context(chapter_id: int, topic: str) -> str:
    docs = vector_store_textbooks.similarity_search(
        query=topic,
        k=6,
        filter={"chapter_id": str(chapter_id)},
    )
    if not docs:
        return ""
    return "\n\n".join([doc.page_content for doc in docs])


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
                    return ""
                return "\n\n".join([row[0] for row in rows])
    except Exception as e:
        logger.warning(f"Failed to fetch QA patterns for chapter {chapter_id}: {e}")
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
        "Keep titles short and summaries 1 sentence."
        f"{_language_instruction(medium_name)}\n\n"
        f"MEDIUM: {medium_name}\n"
        f"CHAPTER TEXT:\n{text}"
    )

    llm = _get_llm().with_structured_output(TopicList)
    result: TopicList = llm.invoke(
        [
            SystemMessage(content="Act as an expert Curriculum Designer."),
            HumanMessage(content=human_prompt),
        ]
    )
    return [t.model_dump() for t in result.topics]


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
        f"{_language_instruction(medium_name)}\n\n"
        f"MEDIUM: {medium_name}\n\n"
        f"TOPICS:\n{json.dumps(topics)}"
    )

    llm = _get_llm().with_structured_output(TopicList)
    result: TopicList = llm.invoke(
        [
            SystemMessage(content="Act as an expert Curriculum Designer."),
            HumanMessage(content=human_prompt),
        ]
    )
    return [t.model_dump() for t in result.topics]


def generate_topics(chapter_id: int, medium_name: str = "") -> List[Dict[str, str]]:
    chapter_text = get_full_chapter_text(chapter_id)
    if not chapter_text:
        return []

    all_topics: List[Dict[str, str]] = []
    if len(chapter_text) <= MAX_TOPIC_CHARS:
        all_topics.extend(_generate_topics_from_text(chapter_text, medium_name))
        return _consolidate_topics(all_topics, medium_name=medium_name)

    chunks = _split_text(chapter_text, chunk_size=MAX_TOPIC_CHARS, chunk_overlap=400)
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
            except Exception as exc:
                logger.warning(f"Topic extraction failed for chunk {idx}: {exc}")
                chunk_results[idx] = []
        for i in range(len(chunks)):
            all_topics.extend(chunk_results.get(i, []))

    if not all_topics:
        return []
    return _consolidate_topics(all_topics, medium_name=medium_name)


# --------------------
# Activity generation
# --------------------

def generate_activities(
    chapter_id: int,
    topic_titles: List[str],
    mcq_count: int,
    descriptive_count: int,
    medium_name: str = "",
) -> List[Dict[str, Any]]:
    all_context_parts = []
    for title in topic_titles:
        ctx = get_topic_context(chapter_id, title)
        if ctx:
            all_context_parts.append(ctx)

    if not all_context_parts:
        chapter_text = get_full_chapter_text(chapter_id)
        topic_context = chapter_text[:MAX_CONTEXT_CHARS]
    else:
        topic_context = "\n\n".join(all_context_parts)[:MAX_CONTEXT_CHARS]

    qa_context = get_qa_context(chapter_id)
    topics_str = ", ".join(topic_titles)

    language_instruction = ""
    examples = ""
    medium_lower = medium_name.lower()
    if any(
        lang in medium_lower
        for lang in [
            "kannada", "malayalam", "tamil", "telugu", "hindi",
            "sanskrit", "urdu", "bengali", "marathi", "gujarati",
        ]
    ):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = f"\n\nCRITICAL: This is {lang_name} medium. ALL questions, options, and answers MUST be in {lang_name} language."
        examples = f"""

Example for {lang_name} medium:
{{
  "type": "mcq",
  "question_text": "[Question in {lang_name}]",
  "options": ["[Option 1 in {lang_name}]", "[Option 2 in {lang_name}]", "[Option 3 in {lang_name}]", "[Option 4 in {lang_name}]"],
  "correct_answer": "[Correct option text in {lang_name}]",
  "answer_description": "[1-2 sentence explanation of why the correct answer is right, in {lang_name}]"
}}
"""

    human_prompt = (
        f"Generate {mcq_count} MCQs and {descriptive_count} Descriptive questions "
        "based on the provided TOPICS and CONTEXT.\n\n"
        "Core Constraints:\n"
        "- Zero-Gap Coverage: Every topic listed in the TOPICS section must be covered by at least one activity.\n"
        "- Cognitive Depth: Distribute questions across Bloom's Taxonomy (Recall, Understanding, and Application).\n"
        "- Distractor Quality: For MCQs, provide plausible distractors (wrong options). "
        "Avoid 'None of the above' unless absolutely necessary.\n"
        "- IMPORTANT: correct_answer must be the EXACT text of one of the options.\n"
        "- For MCQs, include answer_description: a 1-2 sentence explanation of why the correct answer is right.\n"
        f"{examples}"
        f"{language_instruction}\n\n"
        f"MEDIUM: {medium_name}\n"
        f"TOPICS: {topics_str}\n\n"
        + (
            f"IMPORTANT Q&A (Previous Year / Must-Study):\n"
            f"These are high-priority questions from previous exams. Anchor-First: adapt and incorporate "
            f"these into your output before filling gaps from the textbook.\n{qa_context}\n\n"
            if qa_context else ""
        )
        + f"TEXTBOOK CONTEXT:\n{topic_context}"
    )

    llm = _get_llm().with_structured_output(ActivityList)
    result: ActivityList = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert Pedagogy & Assessment Design AI. "
                    "Generate high-quality, exam-standard activities."
                )
            ),
            HumanMessage(content=human_prompt),
        ]
    )
    return [a.model_dump() for a in result.activities]


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
        "Evaluate the student's answer compared to the correct answer.\n\n"
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

    llm = _get_llm().with_structured_output(EvaluationResult)
    result: EvaluationResult = llm.invoke(
        [
            SystemMessage(content="You are an expert educational evaluator."),
            HumanMessage(content=human_prompt),
        ]
    )

    score = max(0, min(100, result.score))
    return {
        "score": score,
        "feedback": result.feedback,
    }
