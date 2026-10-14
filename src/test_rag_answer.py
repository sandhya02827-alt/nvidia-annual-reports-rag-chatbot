
from retrieve import load_retriever, retrieve_passages
from answer_generator import generate_answer


def main():
    print("Loading NVIDIA report search index...")

    embedding_model, index, chunks = load_retriever()

    question = (
        "According to NVIDIA's fiscal year 2025 annual "
        "report, what was total revenue for fiscal year 2025?"
    )

    print("\nQuestion:", question)
    print("\nSearching documents...")

    passages = retrieve_passages(
        question,
        embedding_model,
        index,
        chunks,
        top_k=5,
    )

    if not passages:
        print("No relevant passages found.")
        return

    print("\nGenerating answer using NVIDIA API...")

    answer = generate_answer(question, passages)

    print("\n========== CHATBOT ANSWER ==========")
    print(answer)

    print("\n========== SOURCE REFERENCES ==========")

    for number, passage in enumerate(passages, start=1):
        print(
            f"[{number}] "
            f"{passage.get('source', 'Unknown')} | "
            f"PDF page: {passage.get('page', 'Unknown')}"
        )


if __name__ == "__main__":
    main()
