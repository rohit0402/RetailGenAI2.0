import sys
import re
import time
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_config import llm, embedding_model
from utils.prompts import RAG_TEMPLATE


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
    / "evaluation_questions.json"
)

TOP_K = 5

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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

        documents.append(
            {
                "text": text,
                "metadata": metadata
            }
        )

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
        collection_name="experiment_05_reranking"
    )


def transform_query(question):

    filters = {}
    semantic_query = question

    # Extract record ID
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

    # Extract region
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


def retrieve_candidates(
    vectordb,
    transformed_query
):

    semantic_query = transformed_query[
        "semantic_query"
    ]

    filters = transformed_query[
        "filters"
    ]

    if filters:

        if len(filters) == 1:

            chroma_filter = filters

        else:

            chroma_filter = {
                "$and": [
                    {
                        key: value
                    }
                    for key, value
                    in filters.items()
                ]
            }

        docs = vectordb.similarity_search(
            semantic_query,
            k=TOP_K,
            filter=chroma_filter
        )

    else:

        docs = vectordb.similarity_search(
            semantic_query,
            k=TOP_K
        )

    return docs


def rerank(
    question,
    docs,
    reranker
):

    pairs = [
        [
            question,
            doc.page_content
        ]
        for doc in docs
    ]

    scores = reranker.predict(
        pairs
    )

    scored_docs = list(
        zip(
            docs,
            scores
        )
    )

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score
        in scored_docs
    ]


def generate_answer(
    context,
    question
):

    prompt = ChatPromptTemplate.from_template(
        RAG_TEMPLATE
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


def normalize_answer(answer):

    if isinstance(answer, list):

        return "\n".join(
            item
            if isinstance(item, str)
            else str(item)
            for item in answer
        )

    return str(answer)


def extract_record_id(question):

    match = re.search(
        r"\brecord\s*(?:id)?\s*[:#]?\s*(\d+)\b",
        question,
        re.IGNORECASE
    )

    if match:

        return match.group(1)

    return None


def main():

    print("=" * 70)
    print("EXPERIMENT 5 — CROSS-ENCODER RERANKING")
    print("=" * 70)

    df = pd.read_csv(
        DATASET_PATH
    )

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    documents = create_documents(
        df
    )

    print(
        "\nLoading reranker:",
        RERANK_MODEL
    )

    reranker = CrossEncoder(
        RERANK_MODEL
    )

    start_indexing = time.perf_counter()

    vectordb = build_vector_store(
        documents
    )

    indexing_time = (
        time.perf_counter()
        - start_indexing
    )

    latencies = []

    retrieval_hits = []

    reciprocal_ranks = []

    answer_correct = []

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_answer = str(
            item["expected_answer"]
        )

        record_id = extract_record_id(
            question
        )

        print(
            f"\nQ{i}: {question}"
        )

        transformed = transform_query(
            question
        )

        print(
            "Transformed query:"
        )

        print(
            transformed
        )

        start = time.perf_counter()

        # Stage 1: retrieve candidates
        candidate_docs = retrieve_candidates(
            vectordb,
            transformed
        )

        candidate_ids = [
            str(
                doc.metadata["record_id"]
            )
            for doc in candidate_docs
        ]

        # Stage 2: rerank
        docs = rerank(
            question,
            candidate_docs,
            reranker
        )

        retrieved_ids = [
            str(
                doc.metadata["record_id"]
            )
            for doc in docs
        ]

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        answer = generate_answer(
            context,
            question
        )

        latency = (
            time.perf_counter()
            - start
        )

        latencies.append(
            latency
        )

        print(
            "Candidate IDs:",
            candidate_ids
        )

        print(
            "Reranked IDs:",
            retrieved_ids
        )

        # -------------------------
        # Retrieval evaluation
        # -------------------------

        if (
            record_id is not None
            and record_id in retrieved_ids
        ):

            retrieval_hits.append(1)

            rank = (
                retrieved_ids.index(
                    record_id
                )
                + 1
            )

            reciprocal_ranks.append(
                1 / rank
            )

            print(
                "Correct rank:",
                rank
            )

        else:

            retrieval_hits.append(0)

            reciprocal_ranks.append(0)

            print(
                "Correct record not retrieved"
            )

        # -------------------------
        # Answer evaluation
        # -------------------------

        answer = normalize_answer(
            answer
        )

        expected_match = re.search(
            r"[\d,]+(?:\.\d+)?",
            expected_answer
        )

        if expected_match:

            expected_value = (
                expected_match.group(0)
            )

            is_correct = (
                expected_value in answer
            )

        else:

            is_correct = False

        answer_correct.append(
            1 if is_correct else 0
        )

        print(
            "Answer:",
            answer
        )

        print(
            "Expected:",
            expected_answer
        )

        print(
            "Correct:",
            is_correct
        )

        print(
            f"Latency: {latency:.4f}s"
        )

    # -------------------------
    # Metrics
    # -------------------------

    hit_rate = (
        sum(retrieval_hits)
        / len(retrieval_hits)
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    answer_accuracy = (
        sum(answer_correct)
        / len(answer_correct)
    )

    avg_latency = (
        sum(latencies)
        / len(latencies)
    )

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

    results = {

        "experiment":
            "experiment_05_cross_encoder_reranking",

        "dataset_records":
            len(df),

        "evaluation_questions":
            len(questions),

        "top_k":
            TOP_K,

        "reranker":
            RERANK_MODEL,

        "hit_rate_at_5":
            hit_rate,

        "recall_at_5":
            hit_rate,

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
            p99
    }

    results_path = (
        PROJECT_ROOT
        / "benchmarks"
        / "results"
        / "experiment_05_results.json"
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

    print("\n" + "=" * 70)
    print("EXPERIMENT 5 RESULTS")
    print("=" * 70)

    print(
        f"Hit Rate@5       : {hit_rate:.2%}"
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

    print("\nResults saved to:")
    print(results_path)


if __name__ == "__main__":
    main()