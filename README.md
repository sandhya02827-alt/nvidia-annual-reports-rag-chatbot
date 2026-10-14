# NVIDIA Annual Reports & Proxy Statements RAG Chatbot

An AI-powered chatbot that helps users find and understand information from NVIDIA annual reports and proxy statements. It retrieves relevant document passages and uses an AI model to generate answers with source references.

## Features

- Ask questions about NVIDIA financial reports and proxy statements.
- Semantic search using Sentence Transformers and FAISS.
- AI-generated answers using the NVIDIA API.
- Source references to help users verify answers.
- Interactive web interface built with Streamlit.

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- PyMuPDF
- NVIDIA API
- OpenAI-compatible Python client

## Project Structure

```text
nvidia-rage-chatbot-fresh/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   └── processed/
│       ├── faiss.index
│       └── indexed_chunks.jsonl
└── src/
    ├── answer_generator.py
    ├── build_index.py
    ├── discover_documents.py
    ├── download_documents.py
    ├── extract_text.py
    ├── retrieve.py
    └── test_rag_answer.py
```

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sandhya02827-alt/nvidia-annual-reports-rag-chatbot.git
   cd nvidia-annual-reports-rag-chatbot
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:

   ```text
   NVIDIA_API_KEY=your_nvidia_api_key
   NVIDIA_MODEL=openai/gpt-oss-20b
   ```

   Replace the placeholder with your own NVIDIA API key. Never commit the `.env` file or expose your API key.

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Document Coverage

The current processed dataset contains 35 PDF documents:
- 16 annual reports covering fiscal years 2011–2026.
- 19 proxy statements covering years 1998–2016.

This is a collected subset of NVIDIA's published documents, not a complete archive of every year and document type.

## How It Works

1. PDF documents are processed and divided into text chunks.
2. Sentence Transformers converts text into embeddings.
3. FAISS retrieves relevant passages for a user's question.
4. The NVIDIA API generates an answer based on the retrieved context.
5. The application displays the answer and source references.

## Limitations

- Answers depend on the documents available in the indexed dataset.
- Some questions may not have sufficient supporting evidence.
- AI-generated answers should be verified against the cited source documents.
- API access and a valid API key are required to generate answers.

## Future Improvements

- Expand document coverage.
- Add automated retrieval and answer-quality evaluations.
- Improve citation verification and document filtering.
- Deploy the application for public access.
