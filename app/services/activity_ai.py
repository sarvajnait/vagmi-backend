import json
import re
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.agents.graph import merge_chunks_remove_overlap, vector_store_textbooks


TOPIC_COUNT = 6
MAX_TOPIC_CHARS = 12000
MAX_CONTEXT_CHARS = 15000


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty AI response")

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in AI response")

    return json.loads(match.group(0))


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


def get_full_chapter_text(chapter_id: int) -> str:
    # Use a generic query instead of empty string to avoid embedding errors
    docs = vector_store_textbooks.similarity_search(
        query="chapter content",
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


def _generate_topics_from_text(text: str, medium_name: str = "") -> List[Dict[str, str]]:
    system_prompt = (
        "You are an assistant that outputs strict JSON only. "
        "Do not include markdown or extra text."
    )

    language_instruction = ""
    medium_lower = medium_name.lower()
    if any(lang in medium_lower for lang in ["english", "hindi", "kannada", "malayalam", "tamil", "telugu", "sanskrit", "urdu", "bengali", "marathi", "gujarati"]):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = f"\nIMPORTANT: This is {lang_name} medium. Generate topic titles and summaries in {lang_name} language."

    human_prompt = (
        "From the chapter text below, extract exactly 6 key topics. "
        "Return JSON in this schema:\n"
        '{ "topics": [ { "title": "...", "summary": "..." } ] }\n'
        "Keep titles short and summaries 1 sentence."
        f"{language_instruction}\n\n"
        f"MEDIUM: {medium_name}\n"
        f"CHAPTER TEXT:\n{text}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    payload = _extract_json(response.content)
    topics = payload.get("topics", [])
    return topics if isinstance(topics, list) else []


def _consolidate_topics(topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an assistant that outputs strict JSON only. "
        "Do not include markdown or extra text."
    )
    human_prompt = (
        "Given the topic candidates below, deduplicate and return exactly 6 "
        "most important topics. Return JSON in this schema:\n"
        '{ "topics": [ { "title": "...", "summary": "..." } ] }\n'
        f"TOPICS:\n{json.dumps(topics)}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    payload = _extract_json(response.content)
    topics_out = payload.get("topics", [])
    return topics_out if isinstance(topics_out, list) else []


def generate_topics(chapter_id: int, medium_name: str = "") -> List[Dict[str, str]]:
    chapter_text = get_full_chapter_text(chapter_id)
    if not chapter_text:
        return []

    if len(chapter_text) <= MAX_TOPIC_CHARS:
        topics = _generate_topics_from_text(chapter_text, medium_name)
        return topics[:TOPIC_COUNT]

    chunks = _split_text(chapter_text, chunk_size=MAX_TOPIC_CHARS, chunk_overlap=400)
    all_topics: List[Dict[str, str]] = []
    for chunk in chunks[:5]:
        all_topics.extend(_generate_topics_from_text(chunk, medium_name))

    consolidated = _consolidate_topics(all_topics)
    return consolidated[:TOPIC_COUNT]


def generate_activities(
    chapter_id: int,
    topic_titles: List[str],
    mcq_count: int,
    descriptive_count: int,
    medium_name: str = "",
) -> List[Dict[str, Any]]:
    # Gather context from all topics
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

    topics_str = ", ".join(topic_titles)

    system_prompt = (
        "You are an assistant that outputs strict JSON only. "
        "Do not include markdown or extra text."
    )

    # Language-specific instruction and examples
    language_instruction = ""
    examples = ""
    medium_lower = medium_name.lower()
    if any(lang in medium_lower for lang in ["kannada", "malayalam", "tamil", "telugu", "hindi", "sanskrit", "urdu", "bengali", "marathi", "gujarati"]):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = f"\n\nCRITICAL: This is {lang_name} medium. ALL questions, options, and answers MUST be in {lang_name} language."
        examples = f'''

Example for {lang_name} medium:
{{
  "type": "mcq",
  "question_text": "[Question in {lang_name}]",
  "options": ["[Option 1 in {lang_name}]", "[Option 2 in {lang_name}]", "[Option 3 in {lang_name}]", "[Option 4 in {lang_name}]"],
  "correct_answer": "[Correct option text in {lang_name}]"
}}
'''

    human_prompt = (
        "Generate activities based on the topics and context below. "
        "Cover ALL the given topics in the generated questions. "
        "Return JSON in this schema:\n"
        '{ "activities": [ { "type": "mcq", "question_text": "...", '
        '"options": ["a","b","c","d"], "correct_answer": "b" }, '
        '{ "type": "descriptive", "question_text": "...", "answer_text": "..." } ] }\n'
        "IMPORTANT: correct_answer must be the EXACT text of one of the options.\n"
        f"{examples}"
        f"Requirements:\n- mcq_count: {mcq_count}\n"
        f"- descriptive_count: {descriptive_count}\n"
        "- Keep questions clear and concise.\n"
        "- Distribute questions evenly across all topics."
        f"{language_instruction}\n\n"
        f"MEDIUM: {medium_name}\n"
        f"TOPICS: {topics_str}\n\n"
        f"CONTEXT:\n{topic_context}"
    )

    llm = _get_llm()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    payload = _extract_json(response.content)
    activities = payload.get("activities", [])
    return activities if isinstance(activities, list) else []


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
        # Match correct_answer text to option index
        try:
            correct_option_index = cleaned_options.index(correct_answer) + 1
        except ValueError:
            # Fallback: try case-insensitive match
            lower_options = [o.lower() for o in cleaned_options]
            try:
                correct_option_index = lower_options.index(correct_answer.lower()) + 1
            except ValueError:
                return None
        return {
            "type": "mcq",
            "question_text": question_text,
            "options": cleaned_options,
            "correct_option_index": int(correct_option_index),
            "answer_text": None,
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
    system_prompt = (
        "You are an assistant that outputs strict JSON only. "
        "Do not include markdown or extra text."
    )

    language_instruction = ""
    medium_lower = medium_name.lower()
    if any(lang in medium_lower for lang in ["kannada", "malayalam", "tamil", "telugu", "hindi", "sanskrit", "urdu", "bengali", "marathi", "gujarati"]):
        lang_name = medium_name.replace(" medium", "").replace("Medium", "").strip()
        language_instruction = f"\n\nIMPORTANT: Generate feedback in {lang_name} language since this is {lang_name} medium."

    human_prompt = (
        "Evaluate the student's answer compared to the correct answer. "
        "Return JSON in this schema:\n"
        '{ "score": 75, "feedback": ["Good: ...", "Good: ...", "Improve: ...", "Improve: ..."] }\n\n'
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
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    payload = _extract_json(response.content)

    score = int(payload.get("score", 0))
    feedback = payload.get("feedback", [])

    if not isinstance(feedback, list):
        feedback = []

    # Ensure score is in valid range
    score = max(0, min(100, score))

    return {
        "score": score,
        "feedback": feedback,
    }
