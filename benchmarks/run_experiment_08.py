import os
import statistics
import sys
import time
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pandas as pd
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from utils.prompts import RAG_TEMPLATE


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluations"
    / "benchmark_retail.csv"
)

TOP_K = 5


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,
)


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# --------------------------------------------------
# QUESTIONS
# --------------------------------------------------
# These are deliberately semantic questions.
# We do NOT use deterministic analytics here because
# Experiment 8 is testing streaming behaviour.
# --------------------------------------------------

QUESTIONS = [
    "Describe the performance of the Black Friday campaign in the North region, considering its revenue and ROI.",

    "What factors make the Festive Flash Sale campaign notable based on the retrieved retail data?",

    "Summarize the performance characteristics of the campaigns retrieved for this question.",
]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p / 100

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower]
        + weight * (values[upper] - values[lower])
    )


def content_to_text(content):

    """
    Gemini/LangChain versions can return streamed content
    either as a string or as structured list content.

    Convert both safely to text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text_value = item.get("text")

                if text_value:
                    parts.append(str(text_value))

            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    print("\nLoading benchmark dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset records: {len(df)}")

    print("\nDataset columns:")
    for column in df.columns:
        print(f"  - {repr(column)}")

    return df


# --------------------------------------------------
# CREATE ROW-AWARE DOCUMENTS
# --------------------------------------------------

def create_documents(df):

    documents = []

    for _, row in df.iterrows():

        text = "\n".join(
            [
                f"Record ID: {row['record_id']}",
                f"Campaign Name: {row['campaign_name']}",
                f"Start Date: {row['start_date']}",
                f"Region: {row['region']}",
                f"Units Sold: {row['units_sold']}",
                f"ROI: {row['ROI']}",
                f"Marketing Spend: {row['marketing_spend']}",
                f"Revenue: {row['Revenue']}",
                f"Profit: {row['Profit']}",
                f"Cost: {row['Cost']}",
                f"Customer Traffic: {row['customer_traffic']}",
            ]
        )

        documents.append(text)

    return documents


# --------------------------------------------------
# BUILD VECTOR STORE
# --------------------------------------------------

def build_vector_store(documents):

    print("\nCreating vector store...")

    # Ephemeral Chroma collection.
    # This avoids modifying the application's existing
    # vector database.

    vectordb = Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        collection_name="experiment_08_streaming",
    )

    return vectordb


# --------------------------------------------------
# BUILD RAG CHAIN
# --------------------------------------------------

def build_chain(retriever):

    prompt = ChatPromptTemplate.from_template(
        RAG_TEMPLATE
    )

    chain = (
        {
            "context": retriever,
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# --------------------------------------------------
# STREAMING TEST
# --------------------------------------------------

def run_streaming_test(chain, question):

    start = time.perf_counter()

    first_chunk_time = None

    chunks = []

    chunk_count = 0

    for chunk in chain.stream(question):

        now = time.perf_counter()

        if first_chunk_time is None:

            first_chunk_time = now

        chunk_count += 1

        text = content_to_text(chunk)

        chunks.append(text)

    end = time.perf_counter()

    if first_chunk_time is None:

        return None

    ttft = first_chunk_time - start

    total_time = end - start

    answer = "".join(chunks)

    return {
        "question": question,
        "ttft": ttft,
        "total_time": total_time,
        "chunk_count": chunk_count,
        "answer_length": len(answer),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 70)
    print("RetailGenAI 2.0 - EXPERIMENT 8")
    print("Streaming / Time-to-First-Chunk Benchmark")
    print("=" * 70)

    print(f"Model: {GEMINI_MODEL}")
    print(f"Top K: {TOP_K}")
    print(f"Questions: {len(QUESTIONS)}")

    # ----------------------------------------------
    # Dataset
    # ----------------------------------------------

    df = load_dataset()

    # ----------------------------------------------
    # Row-aware documents
    # ----------------------------------------------

    documents = create_documents(df)

    print(f"Row-aware documents: {len(documents)}")

    # ----------------------------------------------
    # Vector store
    # ----------------------------------------------

    vectordb = build_vector_store(documents)

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    # ----------------------------------------------
    # RAG chain
    # ----------------------------------------------

    chain = build_chain(retriever)

    # ----------------------------------------------
    # Benchmark
    # ----------------------------------------------

    results = []

    for i, question in enumerate(QUESTIONS, 1):

        print("\n" + "-" * 70)

        print(
            f"[{i}/{len(QUESTIONS)}] {question}"
        )

        try:

            result = run_streaming_test(
                chain,
                question
            )

            if result is None:

                print("No streamed output.")

                continue

            results.append(result)

            print(
                f"TTFT:        {result['ttft']:.4f}s"
            )

            print(
                f"Total time:  {result['total_time']:.4f}s"
            )

            print(
                f"Chunks:      {result['chunk_count']}"
            )

            print(
                f"Answer len:  {result['answer_length']}"
            )

        except Exception as e:

            print(
                f"ERROR: {type(e).__name__}: {e}"
            )

    # ----------------------------------------------
    # Aggregate metrics
    # ----------------------------------------------

    if not results:

        print("\nNo successful results.")
        return

    ttfts = [
        r["ttft"]
        for r in results
    ]

    totals = [
        r["total_time"]
        for r in results
    ]

    chunk_counts = [
        r["chunk_count"]
        for r in results
    ]

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 8 RESULTS")
    print("=" * 70)

    print(
        f"Successful requests: {len(results)}"
    )

    print("\nTime To First Chunk")
    print("-" * 40)

    print(
        f"Average: {statistics.mean(ttfts):.4f}s"
    )

    print(
        f"P50:     {percentile(ttfts, 50):.4f}s"
    )

    print(
        f"P95:     {percentile(ttfts, 95):.4f}s"
    )

    print("\nTotal Generation Time")
    print("-" * 40)

    print(
        f"Average: {statistics.mean(totals):.4f}s"
    )

    print(
        f"P50:     {percentile(totals, 50):.4f}s"
    )

    print(
        f"P95:     {percentile(totals, 95):.4f}s"
    )

    print("\nStreaming")
    print("-" * 40)

    print(
        f"Average chunks: "
        f"{statistics.mean(chunk_counts):.2f}"
    )

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        """
Experiment 8 measures streaming responsiveness.

Time To First Chunk measures how quickly the
user begins receiving output.

Total Generation Time measures the complete
LLM generation duration.

Streaming should NOT automatically be considered
a latency optimization. Its primary benefit is
improved perceived responsiveness.
"""
    )


if __name__ == "__main__":
    main()