import sys
import time
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from utils.cache import LLMCache


def simulate_llm_call(question):
    """
    Placeholder for an actual LLM call.

    In the real pipeline this will be replaced
    by the Gemini/RAG invocation.
    """
    time.sleep(0.5)

    return f"Generated answer for: {question}"


cache = LLMCache()


questions = [
    "What is the Revenue of record 36?",
    "What is the Revenue of record 47?",
    "What is the Revenue of record 36?",
    "What is the Revenue of record 47?",
    "What is the Revenue of record 36?",
    "What is the Revenue of record 74?",
]


latencies = []

llm_calls = 0
cache_hits = 0


for question in questions:

    start = time.perf_counter()

    answer = cache.get(question)

    if answer is not None:

        cache_hits += 1
        route = "cache"

    else:

        answer = simulate_llm_call(question)

        cache.set(
            question,
            answer
        )

        llm_calls += 1
        route = "llm"

    latency = time.perf_counter() - start

    latencies.append(latency)

    print()
    print("-" * 70)
    print("QUESTION:", question)
    print("ROUTE:", route)
    print("ANSWER:", answer)
    print("LATENCY:", round(latency, 4), "s")


print()
print("=" * 70)
print("EXPERIMENT 7 — LLM CACHE")
print("=" * 70)

print("Total requests       :", len(questions))
print("Cache hits            :", cache_hits)
print("LLM calls             :", llm_calls)
print("Cache hit rate        :", f"{cache.hit_rate:.2%}")
print(
    "LLM calls avoided      :",
    cache_hits
)

print(
    "Average latency        :",
    f"{sum(latencies) / len(latencies):.4f}s"
)

print("=" * 70)