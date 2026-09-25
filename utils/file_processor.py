import json

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text

from utils.llm_config import embedding_model


VECTOR_STORE_PATH = "vector_store/chroma_db"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


def create_csv_documents(df):
    documents = []
    metadatas = []

    for _, row in df.iterrows():
        document = "\n".join(
            f"{column}: {row[column]}"
            for column in df.columns
        )

        documents.append(document)

        metadatas.append({
            "record_id": str(row["record_id"]),
            "campaign_name": str(row["campaign_name"]),
            "region": str(row["region"]),
            "start_date": str(row["start_date"]),
        })

    return documents, metadatas


def process_file(file, temp_path):
    extension = file.name.rsplit(".", 1)[-1].lower()
    splitter = create_text_splitter()

    chunks = []
    metadatas = None

    if extension == "csv":
        df = pd.read_csv(temp_path)
        st.session_state.df = df

        chunks, metadatas = create_csv_documents(df)

    elif extension == "pdf":
        text = extract_text(temp_path)
        chunks = splitter.split_text(text)

    elif extension == "json":
        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = json.dumps(data, indent=2)
        chunks = splitter.split_text(text)

        try:
            st.session_state.df = pd.read_json(temp_path)
        except ValueError:
            st.session_state.df = None

    elif extension == "txt":
        with open(temp_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    if not chunks:
        raise ValueError(
            "No text could be extracted from the uploaded file."
        )

    kwargs = {
        "texts": chunks,
        "embedding": embedding_model,
        "persist_directory": VECTOR_STORE_PATH,
    }

    if metadatas is not None:
        kwargs["metadatas"] = metadatas

    return Chroma.from_texts(**kwargs)