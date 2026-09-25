import re


UNAVAILABLE = "Information not available in uploaded data."

REGIONS = {"north", "south", "east", "west", "central"}


def extract_record_ids(text):
    text = str(text)

    plural_match = re.search(
        r"\brecords?\b\s*(?:id)?\s*[:#]?\s*"
        r"(\d+(?:\s*(?:and|or|,)\s*\d+)*)",
        text,
        re.IGNORECASE,
    )

    if plural_match:
        return re.findall(r"\d+", plural_match.group(1))

    return re.findall(
        r"\brecord\b\s*(?:id)?\s*[:#]?\s*(\d+)",
        text,
        re.IGNORECASE,
    )


def extract_revenue_value(text):
    patterns = (
        r"\brevenue\b\s*(?:of|is|was|:)?\s*(\d+(?:,\d+)*)",
        r"(\d+(?:,\d+)*)\s*\brevenue\b",
    )

    for pattern in patterns:
        match = re.search(pattern, str(text), re.IGNORECASE)

        if match:
            return float(match.group(1).replace(",", ""))

    return None


def extract_region(question):
    for region in REGIONS:
        if re.search(
            rf"\b{region}\b",
            question,
            re.IGNORECASE,
        ):
            return region.capitalize()

    return None


def find_records(df, record_ids):
    if not record_ids:
        return df.iloc[0:0]

    record_ids = [str(record_id) for record_id in record_ids]

    return df[
        df["record_id"].astype(str).isin(record_ids)
    ]


def unavailable_response():
    return {
        "handled": True,
        "answer": UNAVAILABLE,
        "record_ids": [],
    }


def answer_analytics_question(df, question):
    question_lower = question.lower()

    record_ids = extract_record_ids(question)
    region = extract_region(question)

# Dataset-level revenue
    if any(
        phrase in question_lower
        for phrase in (
            "highest revenue",
            "maximum revenue",
            "max revenue",
            "best revenue",
            "top revenue",
        )
    ):
        row = df.loc[df["Revenue"].idxmax()]

        return {
            "handled": True,
            "answer": (
                f"Record {row['record_id']} "
                f"({row['campaign_name']}) generated the "
                f"highest Revenue of {row['Revenue']}."
            ),
            "record_ids": [str(row["record_id"])],
        }

    # Dataset-level ROI
    if any(
        phrase in question_lower
        for phrase in (
            "strongest roi",
            "highest roi",
            "maximum roi",
            "best roi",
        )
    ):
        row = df.loc[df["ROI"].idxmax()]

        return {
            "handled": True,
            "answer": (
                f"Record {row['record_id']} "
                f"({row['campaign_name']}) had the strongest "
                f"ROI of {row['ROI']}."
            ),
            "record_ids": [str(row["record_id"])],
        }

    # Region + revenue
    if region and "revenue" in question_lower:
        revenue = extract_revenue_value(question)

        if revenue is not None:
            rows = df[
                (df["region"].astype(str).str.lower() == region.lower())
                & (df["Revenue"] == revenue)
            ]

            if rows.empty:
                return unavailable_response()

            record_ids = [
                str(record_id)
                for record_id in rows["record_id"]
            ]

            return {
                "handled": True,
                "answer": (
                    f"Record {', '.join(record_ids)} "
                    f"in the {region} region generated Revenue "
                    f"of {revenue:.0f}."
                ),
                "record_ids": record_ids,
            }

    # Compare records
    comparison_terms = (
        "compare",
        "which generated more",
        "which had more",
        "which generated less",
        "which had less",
        "higher revenue",
        "lower revenue",
        "more revenue",
        "less revenue",
    )

    if (
        len(record_ids) >= 2
        and any(term in question_lower for term in comparison_terms)
    ):
        rows = find_records(df, record_ids)

        if len(rows) != len(record_ids):
            return unavailable_response()

        revenue_values = {
            str(row["record_id"]): row["Revenue"]
            for _, row in rows.iterrows()
        }

        higher_record = max(
            revenue_values,
            key=revenue_values.get,
        )

        details = ", ".join(
            f"record {record_id}: {revenue}"
            for record_id, revenue in revenue_values.items()
        )

        return {
            "handled": True,
            "answer": (
                f"Revenue comparison — {details}. "
                f"Record {higher_record} generated more Revenue."
            ),
            "record_ids": list(revenue_values),
        }

    # Average revenue
    if any(
        phrase in question_lower
        for phrase in ("average revenue", "mean revenue")
    ):
        rows = find_records(df, record_ids)

        if len(rows) != len(record_ids):
            return unavailable_response()

        average_revenue = rows["Revenue"].mean()

        return {
            "handled": True,
            "answer": (
                f"The average Revenue of records "
                f"{', '.join(record_ids)} is "
                f"{average_revenue:.0f}."
            ),
            "record_ids": record_ids,
        }

    # Total revenue
    if any(
        phrase in question_lower
        for phrase in ("total revenue", "sum of revenue")
    ):
        rows = find_records(df, record_ids)

        if len(rows) != len(record_ids):
            return unavailable_response()

        total_revenue = rows["Revenue"].sum()

        return {
            "handled": True,
            "answer": (
                f"The total Revenue of records "
                f"{', '.join(record_ids)} is "
                f"{total_revenue:.0f}."
            ),
            "record_ids": record_ids,
        }

    # Exact record lookup
    if record_ids:
        rows = find_records(df, record_ids)

        if rows.empty:
            return unavailable_response()

        if len(rows) == 1:
            row = rows.iloc[0]

            return {
                "handled": True,
                "answer": (
                    f"Record {row['record_id']} "
                    f"({row['campaign_name']}) belongs to the "
                    f"{row['region']} region and generated "
                    f"Revenue of {row['Revenue']}."
                ),
                "record_ids": [str(row["record_id"])],
            }

    return {
        "handled": False,
        "answer": None,
        "record_ids": [],
    }