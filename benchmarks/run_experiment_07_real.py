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
from utils.cache import LLMCache


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "data/evaluations/benchmark_retail.csv"
TOP_K = 5


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)


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
# Create vector store
# ============================================================

texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    collection_name="experiment_07_real",
)


# ============================================================
# RAG chain
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

        docs = vectordb.similarity_search(
            question,
            k=TOP_K,
            filter={"$and": filters}
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
# Gemini RAG call
# ============================================================

def answer_with_rag(question, docs):

    context = format_docs(docs)

    return rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


# ============================================================
# Real repeated-query workload
# ============================================================

questions = [
    "What was the Revenue of the Black Friday campaign in the North region for record 36?",
    "What was the Revenue of the Black Friday campaign in the North region for record 36?",

    "Which campaign record generated the highest Revenue in the dataset?",
    "Which campaign record generated the highest Revenue in the dataset?",

    "Which campaign had the strongest ROI among the records?",
    "Which campaign had the strongest ROI among the records?",

    "What was the Revenue of the Mumbai campaign?",
    "What was the Revenue of the Mumbai campaign?",
]


# ============================================================
# Cache
# ============================================================

cache = LLMCache()

results = []

cache_hits = 0
gemini_calls = 0


print("=" * 70)
print("EXPERIMENT 7 — REAL RAG + LLM CACHE")
print("=" * 70)


# ============================================================
# Run workload
# ============================================================

for index, question in enumerate(questions, start=1):

    start = time.perf_counter()

    route = classify_query(question)

    # --------------------------------------------------------
    # Deterministic path
    # --------------------------------------------------------

    analytics_result = answer_analytics_question(
        df,
        question
    )

    if analytics_result["handled"]:

        answer = analytics_result["answer"]

        route_used = "deterministic"

        retrieved_ids = analytics_result["record_ids"]

    # --------------------------------------------------------
    # RAG + Cache path
    # --------------------------------------------------------

    else:

        cached_answer = cache.get(question)

        if cached_answer is not None:

            answer = cached_answer

            route_used = "cache"

            retrieved_ids = []

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

            route_used = "gemini"

            gemini_calls += 1

    latency = time.perf_counter() - start

    results.append(
        {
            "request": index,
            "question": question,
            "route": route_used,
            "retrieved_ids": retrieved_ids,
            "latency": latency,
            "answer": answer,
        }
    )

    print()
    print("-" * 70)
    print("REQUEST:", index)
    print("QUESTION:", question)
    print("ROUTE:", route_used)
    print("RETRIEVED:", retrieved_ids)
    print("LATENCY:", round(latency, 4), "s")


# ============================================================
# Metrics
# ============================================================

latencies = [
    result["latency"]
    for result in results
]

average_latency = statistics.mean(latencies)

p50 = statistics.median(latencies)

sorted_latencies = sorted(latencies)

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
# Results
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 7 RESULTS")
print("=" * 70)

print(
    f"Total requests        : {len(results)}"
)

print(
    f"Gemini calls          : {gemini_calls}"
)

print(
    f"Cache hits            : {cache_hits}"
)

print(
    f"Cache hit rate        : {cache.hit_rate:.2%}"
)

print(
    f"Gemini calls avoided  : {cache_hits}"
)

print()
print(
    f"Average latency       : {average_latency:.4f}s"
)

print(
    f"P50 latency           : {p50:.4f}s"
)

print(
    f"P95 latency           : {p95:.4f}s"
)

print(
    f"P99 latency           : {p99:.4f}s"
)

print("=" * 70)


# ============================================================
# Save results
# ============================================================

output_path = (
    "benchmarks/results/experiment_07_real_results.json"
)

output = {
    "experiment": "Experiment 7 - Real RAG + LLM Cache",
    "total_requests": len(results),
    "gemini_calls": gemini_calls,
    "cache_hits": cache_hits,
    "cache_hit_rate": cache.hit_rate,
    "gemini_calls_avoided": cache_hits,
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