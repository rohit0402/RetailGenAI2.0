import sys
import json
import time
import statistics
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_config import llm, embedding_model
from utils.file_processor import format_docs
from utils.prompts import RAG_TEMPLATE
from utils.query_router import classify_query
from utils.analytics import answer_analytics_question
from benchmarks.evaluator_v2 import answer_matches
from utils.cache import LLMCache

# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "data/evaluations/benchmark_retail.csv"
QUESTIONS_PATH = "data/evaluations/evaluation_questions_v2.json"

TOP_K = 5


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATASET_PATH)

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)


# ============================================================
# Build row-aware documents
# ============================================================

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

    documents.append(
        {
            "text": text,
            "metadata": {
                "record_id": str(row["record_id"]),
                "campaign_name": str(row["campaign_name"]),
                "region": str(row["region"]),
                "start_date": str(row["start_date"]),
            },
        }
    )


# ============================================================
# Create Chroma index
# ============================================================

texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    collection_name="experiment_06",
)


# ============================================================
# RAG prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# Retrieval
# ============================================================

def retrieve_documents(question, route):

    filters = []

    record_ids = route.get("record_ids", [])
    region = route.get("region")

    if record_ids:
        filters.append(
            {"record_id": str(record_ids[0])}
        )

    if region:
        filters.append(
            {"region": region}
        )

    if len(filters) == 2:

        search_filter = {
            "$and": filters
        }

        docs = vectordb.similarity_search(
            question,
            k=TOP_K,
            filter=search_filter
        )

    elif len(filters) == 1:

        docs = vectordb.similarity_search(
            question,
            k=TOP_K,
            filter=filters[0]
        )

    else:

        docs = vectordb.similarity_search(
            question,
            k=TOP_K
        )

    return docs


# ============================================================
# RAG answer
# ============================================================

def answer_with_rag(question, docs):

    context = format_docs(docs)

    response = rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response


# ============================================================
# Evaluation helpers
# ============================================================

def calculate_target_recall(retrieved_ids, relevant_ids):

    if not relevant_ids:
        return 0.0

    retrieved = set(str(x) for x in retrieved_ids)
    relevant = set(str(x) for x in relevant_ids)

    return len(retrieved & relevant) / len(relevant)


def calculate_all_targets(retrieved_ids, relevant_ids):

    if not relevant_ids:
        return True

    retrieved = set(str(x) for x in retrieved_ids)
    relevant = set(str(x) for x in relevant_ids)

    return relevant.issubset(retrieved)


def calculate_mrr(retrieved_ids, relevant_ids):

    relevant = set(str(x) for x in relevant_ids)

    for rank, record_id in enumerate(retrieved_ids, start=1):

        if str(record_id) in relevant:
            return 1.0 / rank

    return 0.0




# ============================================================
# Run Experiment 6
# ============================================================

results = []

analytics_count = 0
gemini_count = 0

latencies = []
cache = LLMCache()
cache_hits = 0

print("=" * 70)
print("EXPERIMENT 6 — QUERY ROUTER + DETERMINISTIC ANALYTICS")
print("=" * 70)


for item in questions:

    question = item["question"]
    category = item["category"]
    expected = item["expected_answer"]
    relevant_ids = item["record_ids"]

    start_time = time.perf_counter()

    route = classify_query(question)

    # --------------------------------------------------------
    # Deterministic analytics path
    # --------------------------------------------------------

    analytics_result = answer_analytics_question(
        df,
        question
    )

    if analytics_result["handled"]:

        answer = analytics_result["answer"]

        retrieved_ids = analytics_result["record_ids"]

        route_used = "deterministic"

        analytics_count += 1

    # --------------------------------------------------------
    # RAG path
    # --------------------------------------------------------

    else:

        cached_answer = cache.get(question)

        if cached_answer is not None:

            answer = cached_answer
            retrieved_ids = []
            route_used = "cache"
            cache_hits += 1

        else:

            docs = retrieve_documents(
                question,
                route
            )

            retrieved_ids = [
                str(doc.metadata.get("record_id"))
                for doc in docs
            ]

            answer = answer_with_rag(
                question,
                docs
            )

            cache.set(
                question,
                answer
            )

            route_used = "rag"
            gemini_count += 1

    latency = time.perf_counter() - start_time

    latencies.append(latency)

    target_recall = calculate_target_recall(
        retrieved_ids,
        relevant_ids
    )

    all_targets = calculate_all_targets(
        retrieved_ids,
        relevant_ids
    )

    mrr = calculate_mrr(
        retrieved_ids,
        relevant_ids
    )

    accuracy = answer_matches(
    question,
    expected,
    answer,
    category
)

    results.append(
        {
            "id": item["id"],
            "category": category,
            "question": question,
            "route": route_used,
            "router_type": route["type"],
            "retrieved_ids": retrieved_ids,
            "target_ids": relevant_ids,
            "target_recall": target_recall,
            "all_targets_retrieved": all_targets,
            "mrr": mrr,
            "answer_accuracy": accuracy,
            "latency": latency,
            "answer": answer,
        }
    )

    print()
    print("-" * 70)
    print("QUESTION:", question)
    print("ROUTE:", route_used)
    print("ROUTER TYPE:", route["type"])
    print("RETRIEVED:", retrieved_ids)
    print("EXPECTED:", expected)
    print("ANSWER:", answer)
    print("TARGET RECALL:", target_recall)
    print("ANSWER ACCURACY:", accuracy)
    print("LATENCY:", round(latency, 4), "s")


# ============================================================
# Metrics
# ============================================================

target_recall = statistics.mean(
    r["target_recall"]
    for r in results
)

all_targets = statistics.mean(
    1 if r["all_targets_retrieved"] else 0
    for r in results
)

mrr = statistics.mean(
    r["mrr"]
    for r in results
)

answer_accuracy = statistics.mean(
    1 if r["answer_accuracy"] else 0
    for r in results
)

average_latency = statistics.mean(latencies)

sorted_latencies = sorted(latencies)

p50 = statistics.median(sorted_latencies)

p95_index = min(
    len(sorted_latencies) - 1,
    int(0.95 * len(sorted_latencies))
)

p99_index = min(
    len(sorted_latencies) - 1,
    int(0.99 * len(sorted_latencies))
)

p95 = sorted_latencies[p95_index]
p99 = sorted_latencies[p99_index]


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 6 RESULTS")
print("=" * 70)

print(f"Questions              : {len(results)}")
print(f"Deterministic handled  : {analytics_count}")
print(f"RAG / Gemini handled   : {gemini_count}")

print()
print(f"Target Recall@5        : {target_recall:.2%}")
print(f"All Targets@5          : {all_targets:.2%}")
print(f"MRR                    : {mrr:.4f}")
print(f"Answer Accuracy        : {answer_accuracy:.2%}")

print()
print(f"Average Latency        : {average_latency:.4f}s")
print(f"P50 Latency            : {p50:.4f}s")
print(f"P95 Latency            : {p95:.4f}s")
print(f"P99 Latency            : {p99:.4f}s")

print()
print(
    f"Deterministic Route %  : "
    f"{analytics_count / len(results):.2%}"
)

print(
    f"Gemini Calls Avoided   : "
    f"{analytics_count}"
)

# ============================================================
# Save results
# ============================================================

output_path = (
    "benchmarks/results/experiment_06_results.json"
)

output = {
    "experiment": "Experiment 6 - Query Router + Deterministic Analytics",
    "questions": len(results),
    "deterministic_handled": analytics_count,
    "gemini_calls": gemini_count,
    "deterministic_route_percentage": (
        analytics_count / len(results)
    ),
    "target_recall_at_5": target_recall,
    "all_targets_at_5": all_targets,
    "mrr": mrr,
    "answer_accuracy": answer_accuracy,
    "average_latency": average_latency,
    "p50_latency": p50,
    "p95_latency": p95,
    "p99_latency": p99,
    "results": results,
}

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )

print()
print("Results saved to:")
print(output_path)
print("=" * 70)