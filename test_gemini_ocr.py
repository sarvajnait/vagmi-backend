"""
Quick test script to verify Gemini OCR integration for Kannada PDFs.
Usage: python test_gemini_ocr.py <path_to_pdf>
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in environment variables!")
    print("Please create a .env file with: GOOGLE_API_KEY=your_api_key_here")
    sys.exit(1)

from app.services.pdf_ocr import extract_text_from_pdf_with_fallback, extract_text_from_pdf_with_gemini


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gemini_ocr.py <path_to_kannada_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"Testing Gemini OCR for: {pdf_path}")
    print(f"{'='*60}\n")

    # Test with fallback (tries standard first, then OCR)
    print("Method: Fallback (Standard PDF → Gemini OCR)")
    print("-" * 60)

    try:
        documents = extract_text_from_pdf_with_fallback(pdf_path)

        if documents:
            print(f"✓ Successfully extracted {len(documents)} pages\n")

            # Show first page preview
            first_page = documents[0]
            print("First page preview:")
            print("-" * 60)
            print(first_page.page_content[:500])
            print("-" * 60)

            print(f"\nTotal characters extracted: {sum(len(doc.page_content) for doc in documents)}")
            print(f"Extraction method: {first_page.metadata.get('extraction_method', 'unknown')}")

            # Check if Kannada Unicode is present
            has_kannada = any(
                '\u0C80' <= char <= '\u0CFF'
                for doc in documents
                for char in doc.page_content
            )
            print(f"Contains Kannada Unicode: {'✓ Yes' if has_kannada else '✗ No'}")

        else:
            print("✗ No documents extracted")

    except Exception as e:
        print(f"✗ Error: {e}")

    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")
