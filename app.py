
import streamlit as st

from src.retrieve import load_retriever, retrieve_passages
from src.answer_generator import generate_answer


st.set_page_config(
    page_title="NVIDIA Reports RAG Chatbot",
    page_icon="📊",
    layout="wide",
)

st.title("NVIDIA Annual Reports & Proxy Statements")

st.caption(
    "Ask questions about NVIDIA's indexed annual reports "
    "and proxy statements. Answers use retrieved document evidence."
)

st.info(
    "Current document coverage: Annual reports 2011–2026 "
    "and proxy statements 1998–2016. Coverage varies by document type."
)


@st.cache_resource(show_spinner="Loading document search index...")
def initialize_retriever():
    return load_retriever()


try:
    embedding_model, index, chunks = initialize_retriever()
except Exception as error:
    st.error(f"Could not load the document index: {error}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("About this chatbot")
    st.write("**Documents indexed:** 35 PDFs")
    st.write("**Pages processed:** 4,240")
    st.write("**Searchable passages:** 16,126")
    st.write("**Embedding model:** all-MiniLM-L6-v2")
    st.write("**Answer model:** NVIDIA API — gpt-oss-20b")

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Verify important financial figures against the cited "
        "document and PDF page."
    )


def display_sources(sources):
    if not sources:
        return

    with st.expander("View retrieved source passages"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{number}] {source.get('source', 'Unknown source')}**"
                f" — PDF page {source.get('page', 'Unknown')}"
            )

            st.caption(
                f"Report year: {source.get('year', 'Unknown')} "
                f"| Similarity: {source.get('score', 0):.3f}"
            )

            st.write(source.get("text", ""))
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


question = st.chat_input(
    "Ask about NVIDIA revenue, risks, or financial results..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching reports and preparing your answer..."):
            try:
                passages = retrieve_passages(
                    question,
                    embedding_model,
                    index,
                    chunks,
                    top_k=5,
                )

                if passages:
                    answer = generate_answer(question, passages)
                else:
                    answer = (
                        "I could not find relevant passages in the "
                        "currently indexed documents. Try specifying "
                        "a report year or rephrasing your question."
                    )

            except Exception as error:
                print(
                    f"Chat error: {type(error).__name__}: {error}"
                )
                answer = (
                    "I could not process your question right now. "
                    "Please check the terminal for the error and try again."
                )
                passages = []

        st.markdown(answer)
        display_sources(passages)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": passages,
        }
    )
