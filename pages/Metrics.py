import streamlit as st


st.set_page_config(
    page_title="System Metrics",
    layout="wide"
)

st.title("⚙️ RAG System Metrics")

st.info(
    "Automated benchmarking will be added after "
    "the Gemini baseline is verified."
)


if (
    "rag_chain" in st.session_state
    and st.session_state.rag_chain is not None
):

    st.success(
        "RAG System Status: Active"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "LLM",
            "Gemini"
        )

        st.caption(
            "Temperature: 0.1"
        )

    with col2:

        st.metric(
            "Embeddings",
            "all-MiniLM-L6-v2"
        )

        st.caption(
            "Local Sentence Transformers"
        )

    with col3:

        st.metric(
            "Retrieval K",
            "5"
        )

        st.caption(
            "Semantic Vector Search"
        )

else:

    st.warning(
        "System offline. "
        "Please upload and index a file first."
    )