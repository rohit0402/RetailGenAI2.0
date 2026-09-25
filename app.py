import tempfile
import time

import streamlit as st

from utils.file_processor import process_file
from utils.prompts import SUMMARY_PROMPT, RECOMMENDATION_PROMPT
from utils.rag_pipeline import create_rag_chain
from utils.session_state import initialize_session_state


st.set_page_config(
    page_title="RetailGenAI 2.0",
    layout="wide",
)

initialize_session_state()


st.title("🤖 RetailGenAI 2.0")

st.markdown(
    "Upload a **CSV, JSON, PDF, or TXT** file "
    "to generate summaries, insights, and answers."
)


def process_uploaded_file(uploaded_file):
    """Save, index, and initialize the uploaded dataset."""

    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}",
    ) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    with st.spinner(f"Indexing {uploaded_file.name}..."):
        vectordb = process_file(
            uploaded_file,
            temp_path,
        )

        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5}
        )

        rag_chain = create_rag_chain(
            retriever,
            df=st.session_state.df,
        )

    st.session_state.processed_file = uploaded_file.name
    st.session_state.vectordb = vectordb
    st.session_state.rag_chain = rag_chain
    st.session_state.file_processed = True

    st.success("Indexing complete!")


def generate_report(prompt, session_key, success_message):
    """Generate and store a report using the RAG chain."""

    start_time = time.time()

    answer = st.session_state.rag_chain.invoke(prompt)

    elapsed_time = time.time() - start_time

    st.session_state[session_key] = answer

    st.success(
        f"{success_message} in {elapsed_time:.2f} sec"
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("1. Upload Data")

    uploaded_file = st.file_uploader(
        "Upload File",
        type=["csv", "pdf", "json", "txt"],
    )

    if st.session_state.processed_file:
        st.success(
            f"Current Uploaded File: "
            f"{st.session_state.processed_file}"
        )

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button(
            "Process & Index File",
            type="primary",
        ):
            st.session_state.summary = None
            st.session_state.recommendations = None
            st.session_state.messages = []

            process_uploaded_file(uploaded_file)


# --------------------------------------------------
# Main Section
# --------------------------------------------------

if st.session_state.rag_chain is None:
    st.info("👈 Upload and process a file to begin.")
    st.stop()


# --------------------------------------------------
# Executive Summary
# --------------------------------------------------

if st.button("📝 Generate Executive Summary"):
    with st.spinner("Generating summary..."):
        generate_report(
            SUMMARY_PROMPT,
            "summary",
            "Summary generated",
        )


if st.session_state.summary:
    st.subheader("📄 Executive Summary")
    st.info(st.session_state.summary)


# --------------------------------------------------
# AI Recommendations
# --------------------------------------------------

if st.session_state.summary:

    if st.button("📌 Generate AI Recommendations"):
        with st.spinner("Generating recommendations..."):
            generate_report(
                RECOMMENDATION_PROMPT,
                "recommendations",
                "Recommendations generated",
            )

    if st.session_state.recommendations:
        st.subheader("📌 AI Recommendations")
        st.info(st.session_state.recommendations)


# --------------------------------------------------
# Chat
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions about the Data")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            st.session_state.rag_chain.stream(prompt)
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )