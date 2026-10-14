import os
from types import SimpleNamespace

os.environ.setdefault("NVIDIA_API_KEY", "unit-test-key")

from src import answer_generator


class FakeClient:
    def __init__(self, answer="Revenue was $130.5 billion. [1]"):
        self.answer = answer
        self.messages = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer)
                )
            ]
        )


def test_empty_question_returns_prompt_message():
    assert answer_generator.generate_answer("   ", []) == (
        "Please enter a question."
    )


def test_no_passages_returns_insufficient_evidence_message():
    result = answer_generator.generate_answer(
        "What was NVIDIA's revenue?", []
    )
    assert "sufficient evidence" in result.lower()


def test_empty_passage_text_does_not_call_api(monkeypatch):
    fake_client = FakeClient()
    monkeypatch.setattr(answer_generator, "client", fake_client)

    result = answer_generator.generate_answer(
        "What was NVIDIA's revenue?",
        [{"source": "report.pdf", "page": 10, "text": "   "}],
    )

    assert result == "No usable text was found in the retrieved passages."
    assert fake_client.messages is None


def test_answer_generation_uses_document_evidence(monkeypatch):
    fake_client = FakeClient()
    monkeypatch.setattr(answer_generator, "client", fake_client)

    result = answer_generator.generate_answer(
        "What was NVIDIA's revenue in fiscal year 2025?",
        [{
            "source": "nvidia_2025_annual_report.pdf",
            "year": 2025,
            "page": 27,
            "text": "Total revenue for fiscal year 2025 was $130.5 billion.",
        }],
    )

    assert "$130.5 billion" in result
    assert fake_client.messages is not None

    prompt_text = "\n".join(
        message["content"] for message in fake_client.messages
    )
    assert "ONLY the supplied document evidence" in prompt_text
    assert "nvidia_2025_annual_report.pdf" in prompt_text
    assert "PDF page: 27" in prompt_text
    assert "$130.5 billion" in prompt_text

