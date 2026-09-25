import sys
import json
import statistics
import time
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from langchain_chroma import Chroma
from utils.llm_config import llm, embedding_model
from utils.prompts import RAG_TEMPLATE


DATASET_PATH = PROJECT_ROOT / "data" / "evaluations" / "benchmark_retail.csv"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluations" / "evaluation_questions.json"

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_PATH = RESULTS_DIR / "experiment_02_results.json"

TOP_K = 5


def normalize_answer(answer):
    if hasattr(answer, "content"):
        answer = answer.content

    if isinstance(answer, list):
        return "\n".join(
            item if isinstance(item, str) else str(item)
            for item in answer
        )

    return str(answer)


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    for key in ["questions", "evaluation_questions", "items", "data"]:
        if key in data:
            return data[key]

    raise ValueError("Could not find evaluation questions.")


def question_text(item):
    if isinstance(item, str):
        return item

    for key in ["question", "query", "prompt", "text"]:
        if key in item:
            return str(item[key])

    raise ValueError("Could not find question text.")


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


def build_documents(df):
    """
    Experiment 2:
    Each row remains one document, but we now attach
    richer structured metadata.
    """

    documents = []
    metadatas = []

    for _, row in df.iterrows():

        record_id = int(row["record_id"])

        text = (
            f"record_id: {record_id}\n"
            f"campaign_name: {row['campaign_name']}\n"
            f"start_date: {row['start_date']}\n"
            f"region: {row['region']}\n"
            f"units_sold: {row['units_sold']}\n"
            f"ROI: {row['ROI']}\n"
            f"marketing_spend: {row['marketing_spend']}\n"
            f"Revenue: {row['Revenue']}\n"
            f"Profit: {row['Profit']}\n"
            f"Cost: {row['Cost']}\n"
            f"customer_traffic: {row['customer_traffic']}"
        )

        metadata = {
            "record_id": record_id,
            "campaign_name": str(row["campaign_name"]),
            "region": str(row["region"]),
            "start_date": str(row["start_date"])
        }

        documents.append(text)
        metadatas.append(metadata)

    return documents, metadatas


def create_vector_store(documents, metadatas):

    return Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=f"retailgenai_exp02_{int(time.time() * 1000000)}"
    )


def invoke_with_retry(prompt, max_retries=3):

    for attempt in range(max_retries):

        try:
            return llm.invoke(prompt)

        except Exception:

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"  LLM retry {attempt + 1}/{max_retries} "
                f"after {wait_time}s..."
            )

            time.sleep(wait_time)


def evaluate_retrieval(docs, target_record_id):

    for rank, doc in enumerate(docs, start=1):

        if doc.metadata.get("record_id") == target_record_id:
            return 1, rank, 1.0 / rank

    return 0, None, 0.0


def evaluate_answer(answer, expected_revenue):

    answer = normalize_answer(answer).lower()

    revenue = float(expected_revenue)

    candidates = {
        str(int(revenue)),
        f"{revenue:.0f}",
        f"{revenue:,.0f}",
        f"{revenue:.1f}",
        f"{revenue:,.1f}"
    }

    return any(
        value.lower() in answer
        for value in candidates
    )


def percentile(values, p):

    if len(values) == 1:
        return values[0]

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def main():

    print("=" * 70)
    print("RetailGenAI - EXPERIMENT 02")
    print("Metadata-Aware Retrieval")
    print("=" * 70)

    print("\nConfiguration:")
    print("Chunking        : One row per chunk")
    print("Metadata        : record_id, campaign, region, date")
    print("Embedding       : all-MiniLM-L6-v2")
    print("Vector DB       : ChromaDB")
    print("Retrieval       : Metadata-filtered semantic search")
    print(f"Retrieval K     : {TOP_K}")
    print("LLM             : Gemini")

    df = pd.read_csv(DATASET_PATH)
    questions = load_questions()

    print(f"\nDataset records : {len(df)}")
    print(f"Evaluation qs   : {len(questions)}")

    # --------------------------------------------------------------
    # Index
    # --------------------------------------------------------------

    print("\nCreating row-aware documents + metadata...")

    start = time.perf_counter()

    documents, metadatas = build_documents(df)

    print(f"Documents       : {len(documents)}")

    vectordb = create_vector_store(
        documents,
        metadatas
    )

    indexing_time = time.perf_counter() - start

    print(
        f"Indexing time   : {indexing_time:.4f}s"
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)

    hits = []
    reciprocal_ranks = []
    correct_answers = []
    latencies = []

    individual_results = []

    for i, item in enumerate(questions, start=1):

        question = question_text(item)

        record_id = extract_record_id(question)

        row = df[
            df["record_id"] == record_id
        ].iloc[0]

        expected_revenue = row["Revenue"]

        print(f"\n[{i}/{len(questions)}] {question}")

        # ----------------------------------------------------------
        # Metadata filter
        # ----------------------------------------------------------

        docs = vectordb.similarity_search(
            question,
            k=TOP_K,
            filter={
                "record_id": record_id
            }
        )

        hit, rank, rr = evaluate_retrieval(
            docs,
            record_id
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = RAG_TEMPLATE.format(
            context=context,
            question=question
        )

        generation_start = time.perf_counter()

        response = invoke_with_retry(prompt)

        latency = (
            time.perf_counter()
            - generation_start
        )

        answer = normalize_answer(response)

        correct = evaluate_answer(
            answer,
            expected_revenue
        )

        hits.append(hit)
        reciprocal_ranks.append(rr)
        correct_answers.append(int(correct))
        latencies.append(latency)

        print(f"  Metadata Filter : record_id={record_id}")
        print(f"  Retrieval Hit@5 : {hit}")
        print(f"  Relevant Rank   : {rank}")
        print(f"  Reciprocal Rank : {rr:.4f}")
        print(f"  Expected Revenue: {expected_revenue}")
        print(f"  Answer Correct  : {int(correct)}")
        print(f"  Latency         : {latency:.4f}s")

        individual_results.append(
            {
                "id": (
                    item.get("id", f"q{i:02d}")
                    if isinstance(item, dict)
                    else f"q{i:02d}"
                ),
                "question": question,
                "target_record_id": record_id,
                "expected_revenue": float(expected_revenue),
                "answer": answer,
                "metadata_filter": {
                    "record_id": record_id
                },
                "retrieval_hit_at_5": hit,
                "relevant_rank": rank,
                "reciprocal_rank": rr,
                "answer_correct": int(correct),
                "latency_seconds": latency
            }
        )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    hit_rate = statistics.mean(hits)
    mrr = statistics.mean(reciprocal_ranks)
    answer_accuracy = statistics.mean(correct_answers)

    average_latency = statistics.mean(latencies)

    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 02 RESULTS")
    print("=" * 70)

    print(f"\nHit Rate@5       : {hit_rate:.2%}")
    print(f"Recall@5         : {hit_rate:.2%}")
    print(f"MRR              : {mrr:.4f}")
    print(f"Answer Accuracy  : {answer_accuracy:.2%}")

    print(f"\nIndexing Time    : {indexing_time:.4f}s")
    print(f"Average Latency  : {average_latency:.4f}s")
    print(f"P50 Latency      : {p50:.4f}s")
    print(f"P95 Latency      : {p95:.4f}s")
    print(f"P99 Latency      : {p99:.4f}s")

    output = {
        "experiment": {
            "name": "Experiment 02 - Metadata-Aware Retrieval",
            "dataset_records": len(df),
            "evaluation_questions": len(questions)
        },
        "configuration": {
            "chunking": "one_row_per_chunk",
            "metadata": [
                "record_id",
                "campaign_name",
                "region",
                "start_date"
            ],
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "ChromaDB",
            "retrieval_strategy": "metadata_filtered_semantic_search",
            "top_k": TOP_K,
            "llm": "Gemini"
        },
        "metrics": {
            "retrieval": {
                "hit_rate_at_5": hit_rate,
                "recall_at_5": hit_rate,
                "mrr": mrr
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
        "individual_results": individual_results
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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

    print("\nResults saved to:")
    print(RESULTS_PATH)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()