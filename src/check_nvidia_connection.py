
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("NVIDIA_API_KEY")
url = "https://integrate.api.nvidia.com/v1/models"

try:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    response.raise_for_status()

    models = response.json().get("data", [])
    model_ids = [item.get("id", "") for item in models]

    target = "openai/gpt-oss-20b"

    print("Models returned:", len(model_ids))
    print("GPT-OSS-20B available:", target in model_ids)

    print("\nSome available models:")
    for model_id in model_ids[:10]:
        print("-", model_id)

except requests.RequestException as error:
    print("Request failed:", type(error).__name__)
    print(str(error))
