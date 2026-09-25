import sys
import json
import statistics
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "evaluations" / "benchmark_retail.csv"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluations" / "evaluation_questions.json"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ["questions", "evaluation_questions", "items", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    if isinstance(data, list):
        return data

    raise ValueError("Could not determine question list structure.")


def extract_record_id(question):
    match = re.search(r"record\s+(\d+)", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_question_text(item):
    if isinstance(item, str):
        return item

    for key in ["question", "query", "prompt", "text"]:
        if key in item:
            return str(item[key])

    return str(item)


def main():
    print("=" * 70)
    print("RetailGenAI - BENCHMARK VALIDATION")
    print("=" * 70)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions not found: {QUESTIONS_PATH}")

    df = pd.read_csv(DATASET_PATH)
    questions = load_questions()

    print(f"\nDataset rows     : {len(df)}")
    print(f"Question count   : {len(questions)}")
    print(f"Dataset columns  : {list(df.columns)}")

    print("\n" + "-" * 70)
    print("QUESTION → DATASET VALIDATION")
    print("-" * 70)

    all_valid = True

    for i, item in enumerate(questions, start=1):
        question = get_question_text(item)
        record_id = extract_record_id(question)

        print(f"\n[{i}] {question}")
        print(f"    Extracted record_id: {record_id}")

        if record_id is None:
            print("    ❌ Could not extract record_id")
            all_valid = False
            continue

        if "record_id" not in df.columns:
            print("    ❌ Dataset has no record_id column")
            all_valid = False
            continue

        matches = df[df["record_id"] == record_id]

        if matches.empty:
            print("    ❌ RECORD DOES NOT EXIST")
            all_valid = False
            continue

        if len(matches) > 1:
            print(f"    ⚠️ Multiple rows found: {len(matches)}")

        row = matches.iloc[0]

        print("    ✅ Record exists")

        for column in ["campaign_name", "region", "revenue", "marketing_spend", "roi"]:
            if column in df.columns:
                print(f"    {column:18}: {row[column]}")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if all_valid:
        print("✅ All benchmark questions reference valid records.")
    else:
        print("❌ Benchmark contains invalid or unresolved questions.")

    print("\nActual dataset records:")
    print(df[["record_id"] + [
        c for c in ["campaign_name", "region", "revenue"]
        if c in df.columns
    ]].to_string(index=False))


if __name__ == "__main__":
    main()