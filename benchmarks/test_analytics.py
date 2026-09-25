import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils.query_router import classify_query
from utils.analytics import answer_analytics_question


df = pd.read_csv("data/evaluations/benchmark_retail.csv")


questions = [
    "Which campaign record generated the highest Revenue in the dataset?",
    "Which campaign had the strongest ROI among the records?",
    "Which record had Revenue of 100500 and belonged to the North region?",
    "Which South region campaign generated 110000 Revenue?",
    "Compare the Revenue of records 36 and 96.",
    "Which generated more Revenue: record 60 or record 100?",
    "What is the average Revenue of records 36 and 96?",
    "What is the total Revenue of records 47 and 74?",
    "What was the Revenue of record 999?",
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    route = classify_query(question)

    print("ROUTE:", route)

    result = answer_analytics_question(df, question)

    print("HANDLED:", result["handled"])
    print("ANSWER:", result["answer"])
    print("RECORDS:", result["record_ids"])