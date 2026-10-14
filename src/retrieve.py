

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "faiss.index"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_retriever():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {CHUNKS_FILE}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Loading chunk metadata...")
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index has {index.ntotal} vectors, "
            f"but metadata has {len(chunks)} chunks."
        )

    print(f"Retriever ready: {len(chunks)} chunks.")
    return embedding_model, index, chunks


def retrieve_passages(question, embedding_model, index, chunks, top_k=5):
    if not question or not question.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    clean_question = question.strip()

    year_match = re.search(
        r"\b(?:fiscal\s+year|annual\s+report|FY)\s*[-:]?\s*(20\d{2})\b",
        clean_question,
        flags=re.IGNORECASE,
    )
    target_year = int(year_match.group(1)) if year_match else None

    risk_query = bool(
        re.search(r"\brisk\s+factors?\b|\brisks\b", clean_question, re.I)
    )

    search_question = clean_question
    if risk_query:
        search_question += (
            " Item 1A business risks uncertainties competition "
            "cybersecurity supply chain export controls regulations "
            "legal risks customer demand manufacturing geopolitical "
            "intellectual property financial risks"
        )

    query_vector = embedding_model.encode(
        [search_question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, indices = index.search(query_vector, index.ntotal)

    stop_words = {
        "what", "were", "was", "the", "and", "for", "from",
        "with", "that", "this", "does", "discuss", "discusses",
        "major", "about", "according", "nvidia", "report",
        "annual", "fiscal", "year", "their", "its", "are",
    }

    query_terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]+", clean_question)
        if term.lower() not in stop_words and len(term) > 1
    }

    risk_detail_terms = {
        "competition", "cybersecurity", "supply", "chain",
        "export", "regulatory", "regulation", "security",
        "customer", "demand", "products", "market",
        "intellectual", "property", "laws", "uncertainties",
        "geopolitical", "manufacturing", "technology",
        "disruption", "litigation", "financial", "operations",
        "semiconductor",
    }

    results = []

    for semantic_score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        record = chunks[int(chunk_index)]
        source = record.get("source", "")
        passage_text = record.get("text", "").strip()
        lower_text = passage_text.lower()

        if target_year is not None:
            correct_year = record.get("year") == target_year
            is_annual_report = "annual_report" in source.lower()

            if not (correct_year and is_annual_report):
                continue

        matched_terms = sum(
            1 for term in query_terms if term in lower_text
        )
        lexical_score = (
            matched_terms / len(query_terms) if query_terms else 0.0
        )

        rank_score = float(semantic_score) + 0.15 * lexical_score

        if risk_query:
            detail_matches = sum(
                1 for term in risk_detail_terms if term in lower_text
            )
            rank_score += 0.04 * min(detail_matches, 8)

            if "item 1a" in lower_text and "risk factors" in lower_text:
                rank_score += 0.08

            if "table of contents" in lower_text:
                rank_score -= 0.45

            reference_only = (
                "refer to" in lower_text
                and "item 1a" in lower_text
                and detail_matches < 3
            )
            if reference_only:
                rank_score -= 0.15

        results.append({
            "score": float(semantic_score),
            "rank_score": rank_score,
            "source": source or "Unknown",
            "year": record.get("year"),
            "document_type": record.get("document_type"),
            "page": record.get("page"),
            "text": passage_text,
            "chunk_index": record.get("chunk_index"),
        })

    results.sort(key=lambda item: item["rank_score"], reverse=True)
    return results[:top_k]


def main():
    embedding_model, index, chunks = load_retriever()

    print("\nAsk a question about NVIDIA reports.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Search session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        results = retrieve_passages(
            question, embedding_model, index, chunks, top_k=5
        )

        print("\n========== RETRIEVED PASSAGES ==========")

        if not results:
            print(
                "No passages found for the requested report year. "
                "Check whether that year's report is indexed."
            )
            continue

        for number, result in enumerate(results, start=1):
            print(f"\n[{number}] Similarity score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"PDF page: {result['page']}")
            print(f"Year: {result['year']}")
            print("-" * 60)
            print(result["text"][:1200])


if __name__ == "__main__":
    main()
