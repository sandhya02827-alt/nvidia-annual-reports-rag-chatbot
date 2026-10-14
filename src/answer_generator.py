
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError
from openai import RateLimitError, AuthenticationError

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv(
    "NVIDIA_MODEL",
    "openai/gpt-oss-20b",
)

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY is missing. Configure it in "
        "Streamlit Cloud Secrets or your local .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
    timeout=180.0,
    max_retries=1,
)


# --------------------------------------------------
# ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, passages):
    """Generate a concise answer grounded in NVIDIA documents."""

    if not isinstance(question, str) or not question.strip():
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
            str(passage.get("source", "Unknown source"))
        ).name

        page = passage.get("page", "Unknown")
        year = passage.get("year", "Unknown")
        text = str(passage.get("text", "")).strip()

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
6. Keep answers concise, usually 2-5 sentences.
7. Cite only passages that support each claim.
8. Use the supplied PDF page metadata.
9. Do not invent source documents or page numbers.
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

        if not response.choices:
            return (
                "The model returned no answer. "
                "Please try again later."
            )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            return (
                "The model returned an empty response. "
                "Please try again."
            )

        answer = answer.strip()
        answer = re.sub(r"【\s*(\d+)\s*】", r"[\1]", answer)

        return answer

    except APITimeoutError:
        print("Answer generation failed: API request timed out.")
        return (
            "The NVIDIA API request timed out. "
            "Please retry with a shorter question."
        )

    except RateLimitError as exc:
        print(
            f"Answer generation failed: rate limit or usage limit. "
            f"HTTP status: {getattr(exc, 'status_code', None)}"
        )
        return (
            "The NVIDIA API rate or usage limit may have been "
            "reached. Please wait and check your API usage."
        )

    except AuthenticationError:
        print("Answer generation failed: API authentication error.")
        return (
            "NVIDIA API authentication failed. "
            "Please check the API key in app Secrets."
        )

    except APIError as exc:
        status = getattr(exc, "status_code", None)
        print(
            f"Answer generation failed: APIError, "
            f"HTTP status={status}, type={type(exc).__name__}"
        )
        return (
            f"The NVIDIA API returned an error "
            f"(HTTP status: {status}). Please try again later "
            "or check the app logs."
        )

    except Exception as exc:
        # Do not print arbitrary exception text, which might
        # contain sensitive request details.
        print(
            f"Answer generation failed: "
            f"{type(exc).__name__}"
        )
        return (
            "An unexpected error occurred during answer "
            "generation. Please check the app logs."
        )
