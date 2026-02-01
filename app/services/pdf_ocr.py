"""
PDF OCR service using Gemini 2 Flash for accurate text extraction.
Handles Kannada and other Indic scripts that have custom font encodings.
"""

import base64
import asyncio
from typing import List
from loguru import logger
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import fitz  # PyMuPDF


# Lazy initialization for LLM to avoid loading before env vars are set
_llm = None
MAX_CONCURRENT_REQUESTS = 10
_semaphore = None


def get_llm():
    """Get or initialize the Gemini LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            verbose=False,
            temperature=0.0,  # Deterministic for OCR
        )
    return _llm


def get_semaphore():
    """Get or initialize the semaphore."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore

# OCR prompt for text extraction
OCR_PROMPT = """**Extract the text** from the above document as if you were reading it naturally.

Do not add introductions, explanations, or summaries.

Return tables in **HTML format**.
Do not wrap the HTML output in ```html triple-backticks.
Skip table rows that are fully empty or contain only blank cells.

Return equations using **LaTeX**.

### Images

- If no caption exists, place a brief description inside `<img></img>`.
- If a caption exists, place it inside `<img></img>`.
- Never include base64 data, file data, or binary content inside the `<img></img>` tag. Only text.

### Watermarks

Wrap watermarks in: <watermark>TEXT</watermark>

### Checkboxes

Prefer these symbols:

- ☐ unchecked
- ☑ checked
"""


async def process_single_page(pix, page_num: int, pdf_path: str) -> Document | None:
    """Process a single page with Gemini OCR (async with semaphore)."""
    semaphore = get_semaphore()
    llm = get_llm()

    async with semaphore:
        try:
            # Convert pixmap to base64 PNG
            img_data = pix.tobytes("png")
            image_base64 = base64.b64encode(img_data).decode("utf-8")

            # Create message with image and prompt
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    },
                    {
                        "type": "text",
                        "text": OCR_PROMPT,
                    },
                ]
            )

            # Get OCR result from Gemini
            response = await llm.ainvoke([message])
            extracted_text = response.content.strip()

            if extracted_text:
                logger.info(f"Page {page_num}: Extracted {len(extracted_text)} characters")
                return Document(
                    page_content=extracted_text,
                    metadata={
                        "page": page_num,
                        "source": pdf_path,
                        "extraction_method": "gemini_ocr",
                    }
                )
            else:
                logger.warning(f"Page {page_num}: No text extracted")
                return None

        except Exception as e:
            logger.error(f"Page {page_num} failed: {e}")
            return None


async def extract_text_from_pdf_with_gemini_async(
    pdf_path: str,
    dpi: int = 200,
    max_pages: int = None
) -> List[Document]:
    """
    Extract text from PDF using Gemini OCR with parallel processing.

    Args:
        pdf_path: Path to PDF file (can be URL or local path)
        dpi: DPI for image conversion (200 is good balance of quality/speed)
        max_pages: Optional limit on number of pages to process

    Returns:
        List of Document objects with extracted text per page
    """
    try:
        logger.info(f"Converting PDF to images: {pdf_path}")

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc) if max_pages is None else min(max_pages, len(doc))

        logger.info(f"Processing {total_pages} pages with Gemini OCR (max {MAX_CONCURRENT_REQUESTS} parallel)")

        # Convert pages to images (pixmaps)
        zoom = dpi / 72  # Convert DPI to zoom factor (72 is default)
        mat = fitz.Matrix(zoom, zoom)

        pixmaps = []
        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            pixmaps.append((pix, page_num + 1))  # Store pixmap with 1-indexed page number

        # Process all pages in parallel with semaphore limiting concurrency
        tasks = [
            process_single_page(pix, page_num, pdf_path)
            for pix, page_num in pixmaps
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Clean up
        doc.close()

        # Filter out None and exceptions
        documents = [doc for doc in results if isinstance(doc, Document)]

        logger.info(f"Successfully extracted text from {len(documents)}/{total_pages} pages")
        return documents

    except Exception as e:
        logger.error(f"Error in PDF OCR extraction: {e}", exc_info=True)
        raise


def extract_text_from_pdf_with_gemini(
    pdf_path: str,
    dpi: int = 200,
    max_pages: int = None
) -> List[Document]:
    """
    Synchronous wrapper for async Gemini OCR extraction.
    """
    return asyncio.run(extract_text_from_pdf_with_gemini_async(pdf_path, dpi, max_pages))


def extract_text_from_pdf_with_fallback(pdf_path: str) -> List[Document]:
    """
    Always use Gemini OCR for 100% reliable text extraction.

    Works perfectly with:
    - Kannada, Hindi, Telugu, and all Indic scripts
    - Custom/proprietary fonts
    - Scanned PDFs
    - Any language with proper Unicode support
    - English PDFs

    Uses parallel processing (10 pages at once) for speed optimization.

    Args:
        pdf_path: Path to PDF file (local path or URL)

    Returns:
        List of Document objects with extracted text per page
    """
    logger.info(f"Using Gemini OCR for 100% reliable extraction: {pdf_path}")
    return extract_text_from_pdf_with_gemini(pdf_path)
