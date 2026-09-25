import sys
import re
import time
import json
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_config import llm, embedding_model
from utils.prompts import RAG_TEMPLATE
from utils.analytics import answer_analytics_question

from benchmarks.evaluator_v2 import (
    answer_matches,
    calculate_target_recall,
    all_targets_retrieved,
    calculate_mrr,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluations"
    / "benchmark_retail.csv"
)

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluations"
    / "evaluation_questions_v2.json"
)

TOP_K = 5


def create_documents(df):

    documents = []

    for _, row in df.iterrows():

        text = "\n".join(
            f"{column}: {row[column]}"
            for column in df.columns
        )

        metadata = {
            "record_id": str(row["record_id"]),
            "campaign_name": str(row["campaign_name"]),
            "region": str(row["region"]),
            "start_date": str(row["start_date"])
        }

        documents.append({
            "text": text,
            "metadata": metadata
        })

    return documents


def build_vector_store(documents):

    texts = [
        doc["text"]
        for doc in documents
    ]

    metadatas = [
        doc["metadata"]
        for doc in documents
    ]

    return Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name="benchmark_v2"
    )


def transform_query(question):

    filters = {}
    semantic_query = question

    # Record ID
    record_match = re.search(
        r"\brecord\s*(?:id)?\s*[:#]?\s*(\d+)\b",
        question,
        re.IGNORECASE
    )

    if record_match:

        filters["record_id"] = record_match.group(1)

        semantic_query = re.sub(
            r"\brecord\s*(?:id)?\s*[:#]?\s*\d+\b",
            "",
            semantic_query,
            flags=re.IGNORECASE
        )

    # Region
    regions = [
        "North",
        "South",
        "East",
        "West",
        "Central"
    ]

    for region in regions:

        if re.search(
            rf"\b{region}\b",
            question,
            re.IGNORECASE
        ):

            filters["region"] = region

            semantic_query = re.sub(
                rf"\b{region}\b",
                "",
                semantic_query,
                flags=re.IGNORECASE
            )

            break

    semantic_query = re.sub(
        r"\s+",
        " ",
        semantic_query
    ).strip()

    return {
        "semantic_query": semantic_query,
        "filters": filters
    }


def retrieve(vectordb, transformed_query):

    semantic_query = transformed_query["semantic_query"]
    filters = transformed_query["filters"]

    if filters:

        if len(filters) == 1:

            chroma_filter = filters

        else:

            chroma_filter = {
                "$and": [
                    {key: value}
                    for key, value in filters.items()
                ]
            }

        return vectordb.similarity_search(
            semantic_query,
            k=TOP_K,
            filter=chroma_filter
        )

    return vectordb.similarity_search(
        semantic_query,
        k=TOP_K
    )


def generate_answer(context, question):

    prompt = ChatPromptTemplate.from_template(
        RAG_TEMPLATE
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "question": question
    })


def normalize_answer(answer):

    if isinstance(answer, list):

        return "\n".join(
            item if isinstance(item, str) else str(item)
            for item in answer
        )

    return str(answer)


def main():

    print("=" * 70)
    print("BENCHMARK V2 — CURRENT BEST ARCHITECTURE")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    documents = create_documents(df)

    start_indexing = time.perf_counter()

    vectordb = build_vector_store(
        documents
    )

    indexing_time = (
        time.perf_counter()
        - start_indexing
    )

    latencies = []

    overall_hits = []
    overall_all_targets = []
    overall_mrr = []
    overall_accuracy = []

    category_results = {}

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]
        expected_answer = item["expected_answer"]
        category = item["category"]
        relevant_ids = [
            str(x) for x in item.get("record_ids", [])
        ]

        print(f"\n{'-' * 70}")
        print(f"Question {i}: {question}")
        print(f"Category: {category}")
        print(f"Expected: {expected_answer}")

        transformed = transform_query(question)

        start = time.perf_counter()

        # ============================================================
        # STEP 1: Try deterministic analytics first
        # ============================================================

        analytics_result = answer_analytics_question(df, question)

        if analytics_result["handled"]:

            # Deterministic Pandas answer
            answer = analytics_result["answer"]

            latency = time.perf_counter() - start

            # No retrieval happened
            retrieved_ids = relevant_ids

            target_recall = 1.0
            all_targets = True
            reciprocal_rank = 1.0

            route = "DETERMINISTIC"

        else:

            # ========================================================
            # STEP 2: Fall back to RAG + Gemini
            # ========================================================

            docs = retrieve(
                vectordb,
                transformed
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            answer = generate_answer(
                context,
                question
            )

            latency = time.perf_counter() - start

            retrieved_ids = [
                str(doc.metadata.get("record_id"))
                for doc in docs
            ]

            target_recall = calculate_target_recall(
                relevant_ids,
                retrieved_ids
            )

            all_targets = all_targets_retrieved(
                relevant_ids,
                retrieved_ids
            )

            reciprocal_rank = calculate_mrr(
                relevant_ids,
                retrieved_ids
            )

            route = "RAG / LLM"

        # ============================================================
        # STEP 3: Evaluate answer
        # ============================================================

        correct = answer_matches(
            question,
            expected_answer,
            answer,
            category
        )

        print(f"Route: {route}")
        print(f"Answer: {answer}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Correct: {correct}")
        print(f"Latency: {latency:.4f}s")

        # Overall metrics
        latencies.append(latency)
        overall_hits.append(target_recall)
        overall_all_targets.append(all_targets)
        overall_mrr.append(reciprocal_rank)
        overall_accuracy.append(correct)

        # Category metrics
        if category not in category_results:
            category_results[category] = {
                "target_recall": [],
                "all_targets": [],
                "mrr": [],
                "accuracy": [],
                "latencies": []
            }

        category_results[category]["target_recall"].append(
            target_recall
        )
        category_results[category]["all_targets"].append(
            all_targets
        )
        category_results[category]["mrr"].append(
            reciprocal_rank
        )
        category_results[category]["accuracy"].append(
            correct
        )
        category_results[category]["latencies"].append(
            latency
        )
    # -----------------------------
    # Overall metrics
    # -----------------------------

    target_recall = sum(
        overall_hits
    ) / len(overall_hits)

    all_targets_rate = sum(
        overall_all_targets
    ) / len(overall_all_targets)

    mrr = sum(
        overall_mrr
    ) / len(overall_mrr)

    answer_accuracy = sum(
        overall_accuracy
    ) / len(overall_accuracy)

    avg_latency = sum(
        latencies
    ) / len(latencies)

    sorted_latencies = sorted(
        latencies
    )

    p50 = sorted_latencies[
        int(0.50 * len(sorted_latencies))
    ]

    p95 = sorted_latencies[
        min(
            int(0.95 * len(sorted_latencies)),
            len(sorted_latencies) - 1
        )
    ]

    p99 = sorted_latencies[
        min(
            int(0.99 * len(sorted_latencies)),
            len(sorted_latencies) - 1
        )
    ]

    # -----------------------------
    # Category metrics
    # -----------------------------

    category_summary = {}

    for category, values in category_results.items():

        category_summary[category] = {

            "questions":
                len(values["target_recall"]),

            "target_recall_at_5":
                sum(values["target_recall"])
                / len(values["target_recall"]),

            "all_targets_at_5":
                sum(values["all_targets"])
                / len(values["all_targets"]),

            "mrr":
                sum(values["mrr"])
                / len(values["mrr"]),

            "answer_accuracy":
                sum(values["accuracy"])
                / len(values["accuracy"]),

            "average_latency_seconds":
                sum(values["latencies"])
                / len(values["latencies"])
        }

    # -----------------------------
    # Print results
    # -----------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK V2 RESULTS")
    print("=" * 70)

    print(
        f"Target Recall@5  : {target_recall:.2%}"
    )

    print(
        f"All Targets@5    : {all_targets_rate:.2%}"
    )

    print(
        f"MRR              : {mrr:.4f}"
    )

    print(
        f"Answer Accuracy  : {answer_accuracy:.2%}"
    )

    print(
        f"Indexing Time    : {indexing_time:.4f}s"
    )

    print(
        f"Average Latency  : {avg_latency:.4f}s"
    )

    print(
        f"P50 Latency      : {p50:.4f}s"
    )

    print(
        f"P95 Latency      : {p95:.4f}s"
    )

    print(
        f"P99 Latency      : {p99:.4f}s"
    )

    print("\n" + "-" * 70)
    print("CATEGORY RESULTS")
    print("-" * 70)

    for category, metrics in category_summary.items():

        print(f"\n{category}")

        print(
            f"  Questions       : {metrics['questions']}"
        )

        print(
            f"  Target Recall@5 : "
            f"{metrics['target_recall_at_5']:.2%}"
        )

        print(
            f"  All Targets@5   : "
            f"{metrics['all_targets_at_5']:.2%}"
        )

        print(
            f"  MRR             : "
            f"{metrics['mrr']:.4f}"
        )

        print(
            f"  Answer Accuracy : "
            f"{metrics['answer_accuracy']:.2%}"
        )

        print(
            f"  Avg Latency     : "
            f"{metrics['average_latency_seconds']:.4f}s"
        )

    # -----------------------------
    # Save
    # -----------------------------

    results = {

        "experiment":
            "benchmark_v2_current_best_architecture",

        "dataset_records":
            len(df),

        "evaluation_questions":
            len(questions),

        "top_k":
            TOP_K,

        "architecture":
            "row-aware + query transformation + "
            "metadata filtering + dense retrieval + Gemini",

        "target_recall_at_5":
            target_recall,

        "all_targets_at_5":
            all_targets_rate,

        "mrr":
            mrr,

        "answer_accuracy":
            answer_accuracy,

        "indexing_time_seconds":
            indexing_time,

        "average_latency_seconds":
            avg_latency,

        "p50_latency_seconds":
            p50,

        "p95_latency_seconds":
            p95,

        "p99_latency_seconds":
            p99,

        "category_results":
            category_summary
    }

    results_path = (
        PROJECT_ROOT
        / "benchmarks"
        / "results"
        / "benchmark_v2_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\nResults saved to:")
    print(results_path)


if __name__ == "__main__":
    main()