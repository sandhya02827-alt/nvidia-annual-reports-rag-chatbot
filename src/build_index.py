
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "faiss.index"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.jsonl"

# A lightweight model suitable for semantic search.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_FILE}"
        )

    records = []

    print("Loading extracted text chunks...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if record.get("text", "").strip():
                    records.append(record)

    if not records:
        raise ValueError("No valid text chunks found.")

    texts = [record["text"] for record in records]

    print(f"Chunks loaded: {len(records)}")
    print(f"Loading embedding model: {MODEL_NAME}")
    print("The first run may take some time to download the model.")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    print("Building FAISS index...")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    with CHUNKS_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Reload the index to verify that it was saved correctly.
    saved_index = faiss.read_index(str(INDEX_FILE))

    print("\n========== INDEX SUMMARY ==========")
    print(f"Chunks indexed: {saved_index.ntotal}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Indexed chunk metadata: {CHUNKS_FILE}")

    if saved_index.ntotal != len(records):
        raise RuntimeError("Index verification failed.")

    print("Index verification: PASSED")


if __name__ == "__main__":
    main()
