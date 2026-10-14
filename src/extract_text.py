
import json
import re
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def split_text(text):
    """Split page text into overlapping character chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a word boundary.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {RAW_DIR}")
        return

    total_pages = 0
    total_chunks = 0
    empty_pdfs = []
    errors = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            print(f"\nReading: {pdf_path.name}")

            try:
                with pymupdf.open(pdf_path) as document:
                    pdf_chunk_count = 0
                    pages_with_text = 0

                    for page_index, page in enumerate(document):
                        page_text = page.get_text("text")

                        if page_text.strip():
                            pages_with_text += 1

                        page_chunks = split_text(page_text)

                        for chunk_index, chunk in enumerate(
                            page_chunks, start=1
                        ):
                            record = {
                                "source": pdf_path.name,
                                "year": int(
                                    re.match(
                                        r"(\d{4})_", pdf_path.name
                                    ).group(1)
                                ),
                                "document_type": (
                                    "annual_report"
                                    if "_annual_report.pdf"
                                    in pdf_path.name
                                    else "proxy_statement"
                                ),
                                "page": page_index + 1,
                                "chunk_index": chunk_index,
                                "text": chunk,
                            }

                            output.write(
                                json.dumps(
                                    record, ensure_ascii=False
                                ) + "\n"
                            )

                            pdf_chunk_count += 1

                    total_pages += len(document)
                    total_chunks += pdf_chunk_count

                    print(f"Pages: {len(document)}")
                    print(f"Pages with text: {pages_with_text}")
                    print(f"Chunks created: {pdf_chunk_count}")

                    if pages_with_text == 0:
                        empty_pdfs.append(pdf_path.name)

            except Exception as exc:
                print(f"ERROR: {exc}")
                errors.append({
                    "file": pdf_path.name,
                    "error": str(exc),
                })

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Total pages processed: {total_pages}")
    print(f"Total text chunks: {total_chunks}")
    print(f"PDFs with no extractable text: {len(empty_pdfs)}")
    print(f"PDF processing errors: {len(errors)}")
    print(f"Output file: {OUTPUT_FILE}")

    if empty_pdfs:
        print("\nPDFs needing OCR review:")
        for name in empty_pdfs:
            print(f"- {name}")

    if errors:
        print("\nPDFs with errors:")
        for item in errors:
            print(f'- {item["file"]}: {item["error"]}')


if __name__ == "__main__":
    main()
