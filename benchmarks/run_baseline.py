import sys
import json
import re
import statistics
import time
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluations" / "benchmark_retail.csv"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluations" / "evaluation_questions.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_PATH = RESULTS_DIR / "baseline_results.json"

# Baseline configuration - DO NOT CHANGE
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

from utils.llm_config import llm, embedding_model
from utils.prompts import RAG_TEMPLATE


def normalize_answer(answer):
    """Convert Gemini/LangChain response into plain text."""
    if hasattr(answer, "content"):
        answer = answer.content

    if isinstance(answer, list):
        parts = []

        for item in answer:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(answer)


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in [
            "questions",
            "evaluation_questions",
            "items",
            "data"
        ]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Could not determine question list structure.")


def question_text(item):
    if isinstance(item, str):
        return item

    for key in [
        "question",
        "query",
        "prompt",
        "text"
    ]:
        if key in item:
            return str(item[key])

    raise ValueError(f"Could not find question text in: {item}")


def extract_record_id(question):
    match = re.search(
        r"record\s+(\d+)",
        question,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Could not extract record_id from: {question}"
        )

    return int(match.group(1))


def build_chunks(df):
    """
    Reproduce the baseline application's exact text representation
    and chunking while tracking which record IDs overlap each chunk.

    The record/chunk mapping is evaluation-only metadata.
    It does NOT change the baseline retrieval system.
    """

    # Exact same representation used by the application
    raw_text = df.to_string(index=False)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(raw_text)

    # --------------------------------------------------------------
    # Build character-position information for each record.
    # --------------------------------------------------------------

    record_positions = []

    search_start = 0

    for _, row in df.iterrows():

        record_id = int(row["record_id"])

        row_df = pd.DataFrame([row])

        row_text = row_df.to_string(
            index=False,
            header=False
        ).strip()

        position = raw_text.find(
            row_text,
            search_start
        )

        if position == -1:
            # Fallback: search the record_id text.
            record_text = str(record_id)

            position = raw_text.find(
                record_text,
                search_start
            )

            if position == -1:
                continue

            row_start = position
            row_end = position + len(record_text)

        else:
            row_start = position
            row_end = position + len(row_text)

        record_positions.append(
            {
                "record_id": record_id,
                "start": row_start,
                "end": row_end
            }
        )

        search_start = row_end

    # --------------------------------------------------------------
    # Find which records overlap each chunk.
    #
    # NOTE:
    # RecursiveCharacterTextSplitter returns text chunks but does
    # not expose their original character offsets. Therefore we
    # locate each chunk sequentially in the original text.
    # --------------------------------------------------------------

    chunk_record_ids = []

    search_start = 0

    for chunk in chunks:

        chunk_start = raw_text.find(
            chunk,
            search_start
        )

        if chunk_start == -1:
            chunk_record_ids.append([])
            continue

        chunk_end = chunk_start + len(chunk)

        overlapping_ids = []

        for record in record_positions:

            record_start = record["start"]
            record_end = record["end"]

            # Two ranges overlap if:
            #
            # record_start < chunk_end
            # AND
            # record_end > chunk_start

            if (
                record_start < chunk_end
                and record_end > chunk_start
            ):
                overlapping_ids.append(
                    record["record_id"]
                )

        chunk_record_ids.append(
            overlapping_ids
        )

        # Because chunks overlap, move forward but retain enough
        # search context to find the next occurrence correctly.
        search_start = max(
            chunk_start + 1,
            chunk_end - CHUNK_OVERLAP
        )

    return raw_text, chunks, chunk_record_ids


def create_vector_store(chunks, chunk_record_ids):
    """
    Fresh in-memory Chroma store for every benchmark run.
    No persistent application index is reused.
    """

    metadatas = []

    for i, ids in enumerate(chunk_record_ids):
        metadatas.append(
            {
                "chunk_id": i,
                "record_ids": json.dumps(ids)
            }
        )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=f"retailgenai_baseline_{int(time.time() * 1000000)}"
    )

    return vectordb


def invoke_with_retry(prompt, max_retries=3):
    """
    Retry transient LLM/network failures.
    """

    for attempt in range(max_retries):

        try:
            return llm.invoke(prompt)

        except Exception as e:

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"  LLM request failed "
                f"(attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)


def evaluate_answer(answer, expected_revenue):
    """
    Deterministic correctness check.

    The benchmark question asks for Revenue, so we check whether
    the expected revenue value appears in the generated answer.
    """

    answer_text = normalize_answer(answer)
    answer_lower = answer_text.lower()

    revenue = float(expected_revenue)

    # Handle values such as:
    # 45600
    # 45,600
    # 45600.0
    # 45,600.0
    candidates = {
        str(int(revenue)),
        f"{revenue:.0f}",
        f"{revenue:,.0f}",
        f"{revenue:.1f}",
        f"{revenue:,.1f}"
    }

    return any(
        candidate.lower() in answer_lower
        for candidate in candidates
    )


def evaluate_retrieval(docs, target_record_id):
    """
    Proper chunk-level retrieval evaluation.

    A retrieval hit occurs if a retrieved chunk contains the
    complete ground-truth record row.
    """

    for rank, doc in enumerate(docs, start=1):

        metadata = doc.metadata or {}

        raw_ids = metadata.get("record_ids", "[]")

        try:
            record_ids = json.loads(raw_ids)
        except Exception:
            record_ids = []

        if target_record_id in record_ids:

            return {
                "hit": 1,
                "rank": rank,
                "reciprocal_rank": 1.0 / rank
            }

    return {
        "hit": 0,
        "rank": None,
        "reciprocal_rank": 0.0
    }


def main():

    print("=" * 70)
    print("RetailGenAI 2.0 - BASELINE BENCHMARK")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"Chunk size      : {CHUNK_SIZE}")
    print(f"Chunk overlap   : {CHUNK_OVERLAP}")
    print("Embedding       : all-MiniLM-L6-v2")
    print("Vector DB       : ChromaDB")
    print(f"Retrieval K     : {TOP_K}")
    print("LLM             : Gemini")

    # ------------------------------------------------------------------
    # Load benchmark data
    # ------------------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Questions not found: {QUESTIONS_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    questions = load_questions()

    print(f"\nDataset records : {len(df)}")
    print(f"Evaluation qs   : {len(questions)}")

    # Validate all record IDs before running Gemini.
    for item in questions:

        q = question_text(item)
        record_id = extract_record_id(q)

        matches = df[df["record_id"] == record_id]

        if matches.empty:
            raise ValueError(
                f"Benchmark question references missing "
                f"record_id={record_id}"
            )

    # ------------------------------------------------------------------
    # Create exact baseline chunks
    # ------------------------------------------------------------------

    print("\nCreating baseline chunks...")

    index_start = time.perf_counter()

    raw_text, chunks, chunk_record_ids = build_chunks(df)

    print(f"Chunks created  : {len(chunks)}")

    # ------------------------------------------------------------------
    # Create fresh Chroma index
    # ------------------------------------------------------------------

    print("\nCreating Chroma vector store...")

    vectordb = create_vector_store(
        chunks,
        chunk_record_ids
    )

    indexing_time = time.perf_counter() - index_start

    print(
        f"Indexing time   : {indexing_time:.4f} seconds"
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    prompt = ChatPromptTemplate.from_template(
        RAG_TEMPLATE
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)

    results = []
    latencies = []

    retrieval_hits = []
    reciprocal_ranks = []
    answer_correctness = []

    for i, item in enumerate(questions, start=1):

        question = question_text(item)
        target_record_id = extract_record_id(question)

        row = df[
            df["record_id"] == target_record_id
        ].iloc[0]

        expected_revenue = row["Revenue"]

        print(
            f"\n[{i}/{len(questions)}] {question}"
        )

        # --------------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------------

        retrieval_start = time.perf_counter()

        docs = retriever.invoke(question)

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        retrieval_result = evaluate_retrieval(
            docs,
            target_record_id
        )

        # --------------------------------------------------------------
        # Build RAG prompt
        # --------------------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        formatted_prompt = prompt.format(
            context=context,
            question=question
        )

        # --------------------------------------------------------------
        # Gemini generation
        # --------------------------------------------------------------

        generation_start = time.perf_counter()

        response = invoke_with_retry(
            formatted_prompt
        )

        latency = (
            time.perf_counter() - generation_start
        )

        answer = normalize_answer(response)

        # --------------------------------------------------------------
        # Answer evaluation
        # --------------------------------------------------------------

        correct = evaluate_answer(
            answer,
            expected_revenue
        )

        latencies.append(latency)

        retrieval_hits.append(
            retrieval_result["hit"]
        )

        reciprocal_ranks.append(
            retrieval_result["reciprocal_rank"]
        )

        answer_correctness.append(
            int(correct)
        )

        # --------------------------------------------------------------
        # Print result
        # --------------------------------------------------------------

        print(
            f"  Retrieval Hit@5 : "
            f"{retrieval_result['hit']}"
        )

        print(
            f"  Relevant Rank   : "
            f"{retrieval_result['rank']}"
        )

        print(
            f"  Reciprocal Rank : "
            f"{retrieval_result['reciprocal_rank']:.4f}"
        )

        print(
            f"  Expected Revenue: "
            f"{expected_revenue}"
        )

        print(
            f"  Answer Correct  : "
            f"{int(correct)}"
        )

        print(
            f"  Latency         : "
            f"{latency:.4f}s"
        )

        results.append(
            {
                "id": item.get(
                    "id",
                    f"q{i:02d}"
                ) if isinstance(item, dict)
                else f"q{i:02d}",

                "question": question,

                "target_record_id": target_record_id,

                "expected_revenue": float(
                    expected_revenue
                ),

                "answer": answer,

                "retrieval_hit_at_5":
                    retrieval_result["hit"],

                "relevant_rank":
                    retrieval_result["rank"],

                "reciprocal_rank":
                    retrieval_result["reciprocal_rank"],

                "answer_correct":
                    int(correct),

                "retrieval_time_seconds":
                    retrieval_time,

                "latency_seconds":
                    latency,

                "retrieved_chunk_ids": [
                    doc.metadata.get(
                        "chunk_id"
                    )
                    for doc in docs
                ]
            }
        )

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    hit_rate = statistics.mean(
        retrieval_hits
    )

    mrr = statistics.mean(
        reciprocal_ranks
    )

    answer_accuracy = statistics.mean(
        answer_correctness
    )

    average_latency = statistics.mean(
        latencies
    )

    sorted_latencies = sorted(latencies)

    p50 = statistics.median(
        sorted_latencies
    )

    def percentile(values, percentile):

        if len(values) == 1:
            return values[0]

        index = (
            (len(values) - 1)
            * percentile
        )

        lower = int(index)
        upper = min(
            lower + 1,
            len(values) - 1
        )

        weight = index - lower

        return (
            values[lower]
            * (1 - weight)
            + values[upper]
            * weight
        )

    p95 = percentile(
        sorted_latencies,
        0.95
    )

    p99 = percentile(
        sorted_latencies,
        0.99
    )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(
        f"\nHit Rate@5       : {hit_rate:.2%}"
    )

    print(
        f"Recall@5         : {hit_rate:.2%}"
    )

    print(
        f"MRR              : {mrr:.4f}"
    )

    print(
        f"Answer Accuracy  : {answer_accuracy:.2%}"
    )

    print(
        f"\nIndexing Time    : "
        f"{indexing_time:.4f}s"
    )

    print(
        f"Average Latency  : "
        f"{average_latency:.4f}s"
    )

    print(
        f"P50 Latency      : "
        f"{p50:.4f}s"
    )

    print(
        f"P95 Latency      : "
        f"{p95:.4f}s"
    )

    print(
        f"P99 Latency      : "
        f"{p99:.4f}s"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "benchmark": {
            "name":
                "RetailGenAI 2.0 Baseline",

            "dataset_records":
                len(df),

            "evaluation_questions":
                len(questions)
        },

        "configuration": {
            "chunk_size":
                CHUNK_SIZE,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "embedding_model":
                "all-MiniLM-L6-v2",

            "vector_database":
                "ChromaDB",

            "retrieval_strategy":
                "semantic_search",

            "top_k":
                TOP_K,

            "llm":
                "Gemini"
        },

        "metrics": {
            "retrieval": {
                "hit_rate_at_5":
                    hit_rate,

                "recall_at_5":
                    hit_rate,

                "mrr":
                    mrr
            },

            "generation": {
                "deterministic_answer_accuracy":
                    answer_accuracy
            },

            "performance": {
                "indexing_time_seconds":
                    indexing_time,

                "average_generation_latency_seconds":
                    average_latency,

                "p50_latency_seconds":
                    p50,

                "p95_latency_seconds":
                    p95,

                "p99_latency_seconds":
                    p99
            }
        },

        "individual_results":
            results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to:"
    )

    print(RESULTS_PATH)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()