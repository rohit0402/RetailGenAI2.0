import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def main():

    df = pd.read_csv(DATASET_PATH)

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)

    dataset_ids = set(
        df["record_id"]
        .astype(str)
    )

    print("=" * 70)
    print("BENCHMARK V2 VALIDATION")
    print("=" * 70)

    print(
        "Dataset records:",
        len(df)
    )

    print(
        "Questions:",
        len(questions)
    )

    categories = {}

    errors = []

    for question in questions:

        category = question["category"]

        categories[category] = (
            categories.get(category, 0) + 1
        )

        for record_id in question["record_ids"]:

            if str(record_id) not in dataset_ids:

                errors.append(
                    f'{question["id"]}: '
                    f'record {record_id} not found'
                )

    print("\nCategories:")

    for category, count in categories.items():

        print(
            f"  {category}: {count}"
        )

    if errors:

        print("\nERRORS:")

        for error in errors:
            print(error)

        raise ValueError(
            "Benchmark validation failed."
        )

    print("\nAll referenced record IDs exist.")

    print("Benchmark v2 validation PASSED.")


if __name__ == "__main__":
    main()