import re


def normalize_text(text):
    """Normalize text for reliable comparison."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def extract_numbers(text):
    """Extract numbers while ignoring commas."""
    numbers = re.findall(
        r"\d+(?:,\d+)*(?:\.\d+)?",
        str(text)
    )

    return [n.replace(",", "") for n in numbers]


def contains_unavailable(answer):
    """Detect answers that explicitly say required information is unavailable."""
    fallback_phrases = [
        "information not available",
        "not available",
        "no information",
        "cannot find",
        "not found",
        "does not contain",
    ]

    return any(
        phrase in answer
        for phrase in fallback_phrases
    )


def answer_matches(question, expected, answer, category):
    """
    Category-aware deterministic answer evaluation.
    This is NOT an LLM-as-judge.
    """

    expected = normalize_text(expected)
    answer = normalize_text(answer)

    # --------------------------------------------------
    # Unanswerable
    # --------------------------------------------------

    if category == "unanswerable":
        return contains_unavailable(answer)

    expected_numbers = extract_numbers(expected)
    answer_numbers = extract_numbers(answer)

    # --------------------------------------------------
    # Exact lookup
    # --------------------------------------------------

    if category == "exact_lookup":
        return all(
            number in answer_numbers
            for number in expected_numbers
        )

    # --------------------------------------------------
    # Metadata filter
    # --------------------------------------------------

    if category == "metadata_filter":

        numbers_correct = all(
            number in answer_numbers
            for number in expected_numbers
        )

        expected_words = re.findall(
            r"[a-zA-Z]+",
            expected
        )

        text_correct = all(
            word in answer
            for word in expected_words
        )

        return numbers_correct and text_correct

    # --------------------------------------------------
    # Multi-condition
    # --------------------------------------------------

    if category == "multi_condition":
        return all(
            number in answer_numbers
            for number in expected_numbers
        )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    if category == "comparison":

        # A comparison cannot be correct if the answer
        # explicitly says one of the required records
        # is unavailable.
        if contains_unavailable(answer):
            return False

        # Q9 contains both expected revenue values.
        if expected_numbers:
            if not all(
                number in answer_numbers
                for number in expected_numbers
            ):
                return False

        # Q10 expected answer is "record 100".
        if "record 100" in expected:
            return "record 100" in answer

        return True

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------

    if category == "aggregation":
        return all(
            number in answer_numbers
            for number in expected_numbers
        )

    # --------------------------------------------------
    # Semantic
    # --------------------------------------------------

    if category == "semantic":
        return all(
            number in answer_numbers
            for number in expected_numbers
        )

    return expected in answer


def calculate_target_recall(retrieved_ids, relevant_ids):
    """Calculate fraction of required records retrieved."""

    if not relevant_ids:
        return 0.0

    retrieved = set(str(x) for x in retrieved_ids)
    relevant = set(str(x) for x in relevant_ids)

    return len(retrieved & relevant) / len(relevant)


def all_targets_retrieved(retrieved_ids, relevant_ids):
    """Return True only when every required record was retrieved."""

    if not relevant_ids:
        return True

    retrieved = set(str(x) for x in retrieved_ids)
    relevant = set(str(x) for x in relevant_ids)

    return relevant.issubset(retrieved)


def calculate_mrr(retrieved_ids, relevant_ids):
    """
    Reciprocal rank of the first relevant document.

    For multi-record questions this measures how quickly
    we encounter at least one relevant document.
    """

    relevant = set(str(x) for x in relevant_ids)

    for rank, record_id in enumerate(retrieved_ids, start=1):

        if str(record_id) in relevant:
            return 1.0 / rank

    return 0.0