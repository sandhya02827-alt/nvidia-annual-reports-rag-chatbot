
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv(
    "NVIDIA_MODEL",
    "openai/gpt-oss-20b",
)

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError(
        "NVIDIA_API_KEY missing. Check the .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=180.0,
    max_retries=1,
)


def generate_answer(question, passages):
    """Generate a concise answer grounded in NVIDIA documents."""

    if not question or not question.strip():
        return "Please enter a question."

    if not passages:
        return (
            "I could not find sufficient evidence in the "
            "currently indexed NVIDIA documents to answer "
            "this question."
        )

    evidence_sections = []

    for number, passage in enumerate(passages, start=1):
        source = Path(
            passage.get("source", "Unknown source")
        ).name

        page = passage.get("page", "Unknown")
        year = passage.get("year", "Unknown")
        text = passage.get("text", "").strip()

        if not text:
            continue

        evidence_sections.append(
            f"[{number}]\n"
            f"Source: {source}\n"
            f"Report year: {year}\n"
            f"PDF page: {page}\n"
            f"Evidence:\n{text[:1800]}"
        )

    if not evidence_sections:
        return "No usable text was found in the retrieved passages."

    evidence = "\n\n".join(evidence_sections)

    system_prompt = """
You are a careful research assistant for NVIDIA annual reports
and proxy statements.

Answer using ONLY the supplied document evidence.

Rules:
1. Never invent facts, figures, dates, or citations.
2. Cite evidence using square brackets, such as [1] or [2].
3. Preserve units and fiscal-year labels.
4. If evidence is insufficient, clearly say so.
5. Distinguish fiscal years from calendar years.
6. Keep the answer concise, usually 2-5 sentences.
7. Cite only passages that support the relevant claim.
8. PDF page numbers are the supplied PDF page metadata.
9. Do not output malformed Markdown or unusual citation brackets.
"""

    user_prompt = f"""
Question:
{question.strip()}

Retrieved document evidence:
{evidence}

Give a concise, accurate answer with citations such as [1].
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=350,
            extra_body={"reasoning_effort": "low"},
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            return (
                "The model returned an empty response. "
                "Please try again."
            )

        answer = answer.strip()

        # Normalize common citation bracket variations.
        answer = re.sub(r"【\s*(\d+)\s*】", r"[\1]", answer)

        return answer

    except Exception as exc:
        print(
            f"Answer generation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return (
            "Answer generation failed because the model "
            "request timed out or returned an API error. "
            "Please retry the question."
        )
