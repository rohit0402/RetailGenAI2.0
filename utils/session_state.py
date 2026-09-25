import streamlit as st


DEFAULT_SESSION_STATE = {
    "messages": [],
    "rag_chain": None,
    "df": None,
    "processed_file": None,
    "vectordb": None,
    "file_processed": False,
    "summary": None,
    "recommendations": None,
}


def initialize_session_state():
    """Initialize Streamlit session state with default values."""

    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value