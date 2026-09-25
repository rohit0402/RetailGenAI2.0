import hashlib


class LLMCache:
    """In-memory exact-match cache for LLM responses."""

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, question):
        normalized_question = " ".join(
            str(question).lower().split()
        )

        return hashlib.sha256(
            normalized_question.encode("utf-8")
        ).hexdigest()

    def get(self, question):
        key = self._make_key(question)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, question, answer):
        key = self._make_key(question)
        self.cache[key] = answer

    @property
    def hit_rate(self):
        total_requests = self.hits + self.misses

        if total_requests == 0:
            return 0.0

        return self.hits / total_requests