import re


REGIONS = ("North", "South", "East", "West", "Central")

AGGREGATION_KEYWORDS = (
    "average",
    "mean",
    "total",
    "sum",
)

COMPARISON_KEYWORDS = (
    "compare",
    "which generated more",
    "which had more",
    "higher revenue",
    "lower revenue",
)

ANALYTICS_KEYWORDS = (
    "highest revenue",
    "maximum revenue",
    "max revenue",
    "strongest roi",
    "highest roi",
    "maximum roi",
    "best roi",
)


def extract_record_ids(question):
    """Extract one or more record IDs from a question."""

    question = str(question)

    match = re.search(
        r"\brecords?\b(?:\s+id)?\s*[:#]?\s*([^?.!]+)",
        question,
        re.IGNORECASE,
    )

    if not match:
        return []

    record_text = match.group(1)

    record_text = re.split(
        r"\b(?:revenue|roi|region|units|profit|cost|spend)\b",
        record_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    return re.findall(r"\d+", record_text)


def extract_region(question):
    """Extract a supported region from the question."""

    for region in REGIONS:
        if re.search(
            rf"\b{region}\b",
            question,
            re.IGNORECASE,
        ):
            return region

    return None


def classify_query(question):
    """
    Classify a question into a routing category.

    Returns:
        {
            "type": str,
            "record_ids": list,
            "region": str | None,
        }
    """

    question_lower = str(question).lower()

    record_ids = extract_record_ids(question)
    region = extract_region(question)

    if any(
        keyword in question_lower
        for keyword in AGGREGATION_KEYWORDS
    ):
        return {
            "type": "aggregation",
            "record_ids": record_ids,
            "region": region,
        }

    if any(
        keyword in question_lower
        for keyword in COMPARISON_KEYWORDS
    ):
        return {
            "type": "comparison",
            "record_ids": record_ids,
            "region": region,
        }

    has_revenue_filter = "revenue" in question_lower and region
    has_multi_condition = (
        "and" in question_lower
        and (
            "revenue" in question_lower
            or "roi" in question_lower
        )
    )

    if has_revenue_filter or has_multi_condition:
        if record_ids or region:
            return {
                "type": "multi_condition",
                "record_ids": record_ids,
                "region": region,
            }

    if any(
        keyword in question_lower
        for keyword in ANALYTICS_KEYWORDS
    ):
        return {
            "type": "analytics",
            "record_ids": record_ids,
            "region": region,
        }

    if record_ids:
        return {
            "type": "exact_lookup",
            "record_ids": record_ids,
            "region": region,
        }

    if region:
        return {
            "type": "metadata_filter",
            "record_ids": [],
            "region": region,
        }

    return {
        "type": "semantic",
        "record_ids": [],
        "region": None,
    }