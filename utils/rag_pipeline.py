from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from utils.analytics import answer_analytics_question
from utils.cache import LLMCache
from utils.file_processor import format_docs
from utils.llm_config import llm
from utils.prompts import RAG_TEMPLATE


class CachedRAGChain:

    def __init__(self, chain, df=None):
        self.chain = chain
        self.df = df
        self.cache = LLMCache()

        self.cache_hits = 0
        self.llm_calls = 0
        self.deterministic_calls = 0

    def _get_deterministic_answer(self, question):
        if self.df is None:
            return None

        result = answer_analytics_question(
            self.df,
            question,
        )

        if not result["handled"]:
            return None

        self.deterministic_calls += 1

        return result["answer"]

    def _get_cached_answer(self, question):
        answer = self.cache.get(question)

        if answer is not None:
            self.cache_hits += 1

        return answer

    def invoke(self, question):
        answer = self._get_deterministic_answer(question)

        if answer is not None:
            return answer

        answer = self._get_cached_answer(question)

        if answer is not None:
            return answer

        answer = self.chain.invoke(question)

        self.cache.set(question, answer)
        self.llm_calls += 1

        return answer

    def stream(self, question):
        answer = self._get_deterministic_answer(question)

        if answer is not None:
            yield answer
            return

        answer = self._get_cached_answer(question)

        if answer is not None:
            yield answer
            return

        full_answer = ""

        try:
            for chunk in self.chain.stream(question):
                full_answer += chunk
                yield chunk

        except Exception:
            yield (
                "\n\n"
                "⚠️ Unable to generate a response right now. "
                "Please try again."
            )
            return

        self.cache.set(question, full_answer)
        self.llm_calls += 1

    @property
    def cache_hit_rate(self):
        total_requests = self.cache_hits + self.llm_calls

        if total_requests == 0:
            return 0.0

        return self.cache_hits / total_requests


def create_rag_chain(retriever, df=None):
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return CachedRAGChain(
        chain,
        df=df,
    )