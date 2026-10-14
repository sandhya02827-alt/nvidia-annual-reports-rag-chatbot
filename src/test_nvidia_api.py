
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("NVIDIA_API_KEY")
model = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

if not api_key or api_key == "your_existing_nvidia_api_key_here":
    raise ValueError(
        "NVIDIA_API_KEY missing. Please check your .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=0,
)

print(f"Testing NVIDIA API with model: {model}")

try:
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: TEST OK"
        }
    ],
    max_tokens=20,
    temperature=0,
    timeout=60.0,
    extra_body={"reasoning_effort": "low"},
)


    print("\nAPI test successful!")
    print(response.choices[0].message.content)

except Exception as exc:
    print("\nAPI test failed.")
    print(f"Error type: {type(exc).__name__}")
    print(f"Error details: {exc}")
    print("Please do not share your API key.")
