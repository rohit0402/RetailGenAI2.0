RetailGenAI 2.0
RetailGenAI 2.0 is a Streamlit-based retail analytics and Retrieval-Augmented Generation (RAG) application for answering questions over uploaded retail datasets.
The system combines deterministic Pandas analytics for structured numerical questions with ChromaDB + embeddings + Gemini for questions requiring semantic retrieval and natural-language generation.
Architecture
User Question
      |
      v
Deterministic Analytics
   |            |
handled       not handled
   |            |
   v            v
Exact Answer   Cache
                  |
             hit / miss
                  |
                 miss
                  |
                  v
             Chroma + Gemini
                  |
                  v
               Response
Features
- CSV, JSON, PDF and TXT upload
- Row-aware CSV ingestion
- ChromaDB vector retrieval
- all-MiniLM-L6-v2 embeddings
- Gemini RAG fallback
- Deterministic Pandas analytics
- Exact record lookup, comparisons, totals, averages, highest Revenue and ROI
- Metadata-aware retrieval
- Exact-match LLM caching using normalized SHA-256 keys
- Streaming responses with Streamlit
- Query classification utility
- Benchmarking of retrieval quality and latency
Tech Stack
Component	Technology
UI	Streamlit
Language	Python
LLM	Google Gemini
Framework	LangChain
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector DB	ChromaDB
Analytics	Pandas
PDF extraction	pdfminer.six
Cache	In-memory dictionary + SHA-256
Configuration	python-dotenv


Project Structure
RetailGenAI/
├── app.py
├── requirements.txt
├── README.md
├── TECHNICAL_DEEP_DIVE_QA.md
└── utils/
    ├── analytics.py
    ├── cache.py
    ├── file_processor.py
    ├── llm_config.py
    ├── prompts.py
    ├── query_router.py
    ├── rag_pipeline.py
    └── session_state.py
RAG Pipeline
General text is split using RecursiveCharacterTextSplitter with chunk size 1000 and overlap 200. CSV rows are treated as individual documents and enriched with metadata such as record_id, campaign_name, region, and start_date.
Question → Retriever → Chroma → Context → Prompt → Gemini → Answer
Deterministic Analytics
Structured questions are answered directly with Pandas instead of the LLM. Examples:
What was the Revenue of record 96?
Which campaign generated the highest Revenue?
Compare the Revenue of records 36 and 96.
What is the total Revenue of records 47 and 74?
Validated results include:
Record 96 → Revenue 108000
Highest Revenue → Record 87, Festive Flash Sale - South, 115000
Records 36 vs 96 → 100500 vs 108000
Records 47 + 74 → 200500
Experiments
Baseline
Records: 100
Evaluation questions: 12
Chunk size: 1000
Overlap: 200
Embedding: all-MiniLM-L6-v2
Vector DB: ChromaDB
Top K: 5
LLM: Gemini
Recall@5: 83.33%
MRR: 0.5972
Answer accuracy: 83.33%
Average latency: 2.7858 s
P50: 2.4875 s
P95: 4.5102 s
P99: 4.6585 s
Experiment 1 — Row-aware chunking — KEPT
100% retrieval/answer accuracy on the evaluated set. Average latency: 1.2523 s; P50: 1.2119 s; P95: 1.4135 s; P99: 1.4166 s. Average latency was roughly 55% lower than baseline.
Experiment 2 — Metadata-aware retrieval — KEPT
Recall 100%, MRR 1.0, answer accuracy 100%, indexing 2.0932 s, average latency 1.6428 s, P95 2.0716 s.
Experiment 3 — Query transformation/routing — KEPT
Rule-based routing classified exact lookup, metadata filter, semantic, multi-condition, comparison, aggregation and analytics queries. Tested retrieval/answer accuracy was 100% with MRR 1.0.
Experiment 4 — Dense + BM25 + RRF — REJECTED
Recall and accuracy reached 100%, but MRR was 0.8194 and average latency increased to 3.9706 s, with P95 5.8483 s. No sufficient quality gain justified the latency.
Experiment 5 — Cross-encoder reranking — REJECTED
Used cross-encoder/ms-marco-MiniLM-L-6-v2. Recall and accuracy were 100%, MRR 1.0, but average latency was 2.8743 s and P95 6.2437 s. The relevant documents were already in the candidate set, so reranking added latency without quality gain.
Experiment 6 — Deterministic analytics — KEPT
Pandas handles exact lookups, maximum Revenue/ROI, comparisons, averages, totals and multi-condition filtering. This avoids unnecessary LLM calls and improves numerical reliability.
Experiment 7 — Exact-match cache — KEPT
Controlled test: 6 requests, 3 hits, 3 LLM calls, 50% constructed hit rate. A separate real Gemini test had 8 requests, 1 Gemini call and 1 cache hit. The 50% figure is not a production hit-rate claim.
Experiment 8 — Streaming — KEPT
Three-question test: average TTFT 16.8035 s, P50 TTFT 10.7482 s, P95 TTFT 29.5543 s; average total 19.9163 s, P50 13.5210 s, P95 33.0769 s. Streaming improves perceived responsiveness rather than guaranteeing lower total generation time.
Experiment 9 — Dataset-level analytics routing — KEPT
14-question corrected benchmark: 13/14 deterministic, 1/14 RAG fallback, 92.86% deterministic route, 100% answer accuracy. Retrieval metrics from this unified benchmark should not be treated as pure RAG retrieval metrics because deterministic questions were assigned perfect retrieval credit.
Engineering Lessons
- RAG is not the right tool for every question.
- Exact numerical operations should be handled deterministically.
- A reranker cannot recover a relevant document missing from the candidate set.
- Retrieval improvements must be evaluated on both quality and latency.
- Streaming improves perceived responsiveness, not necessarily total latency.
- Dataset-wide ranking should not be inferred from a small top-k retrieved subset.
Current Limitation
A broad question such as “tell me the best revenue campaign, why it performed well, and how to improve the remaining campaigns” should first calculate verified dataset-wide metrics deterministically, then provide those metrics to the LLM for explanation. Allowing top-k RAG to determine the dataset-wide winner can produce a grounding error.
Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Create .env:
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
Never commit .env or API keys.
Example Questions
What was the Revenue of record 96?
Which campaign record generated the highest Revenue in the dataset?
Compare the Revenue of records 36 and 96.
What is the total Revenue of records 47 and 74?
What was the Revenue of the Mumbai campaign?
Unavailable information is returned as:
Information not available in uploaded data.
Future Improvements
- Redis-backed distributed cache
- Stronger intent classification
- Dataset-level metric extraction before LLM generation
- Automated faithfulness evaluation
- Retry/backoff for transient LLM failures
- Production API layer
- Authentication and user-level data isolation
- Observability and tracing
- Persistent vector-store lifecycle management