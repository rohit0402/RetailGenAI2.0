RetailGenAI 2.0
A hybrid Retail Analytics + RAG application that combines deterministic Pandas analytics with ChromaDB and Gemini.

RetailGenAI lets users upload retail data and ask questions in natural language. Structured questions are answered directly from the dataset, while semantic questions fall back to a RAG pipeline.
🚀 Highlights
- 📂 Upload CSV, JSON, PDF, and TXT
- 🔎 ChromaDB-based semantic retrieval
- 🧠 Gemini-powered RAG fallback
- 📊 Deterministic Pandas analytics for exact numerical questions
- 🧩 Row-aware CSV chunking
- 🏷️ Metadata-aware retrieval
- ⚡ Exact-match LLM response caching
- 🌊 Streaming LLM responses
- 🧪 Controlled retrieval and latency experiments
- 📈 Retrieval, MRR, accuracy, P50/P95/P99 benchmarking
🏗️ Architecture
                         User Question
                              |
                              v
                    +---------------------+
                    | Deterministic       |
                    | Analytics (Pandas)  |
                    +----------+----------+
                               |
                    +----------+----------+
                    |                     |
                 Handled              Not handled
                    |                     |
                    v                     v
              Exact Answer             Cache
                                          |
                               +----------+----------+
                               |                     |
                             Hit                   Miss
                               |                     |
                               v                     v
                         Cached Answer       ChromaDB Retriever
                                                   |
                                                   v
                                               Gemini LLM
                                                   |
                                                   v
                                               Response
Core design principle
Use deterministic computation where exactness matters, and use RAG + an LLM where semantic understanding is useful.
🛠️ Tech Stack
Layer	Technology
UI	Streamlit
Language	Python
LLM	Google Gemini
LLM Framework	LangChain
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Database	ChromaDB
Structured Analytics	Pandas
PDF Extraction	pdfminer.six
Cache	In-memory dictionary + SHA-256
Configuration	python-dotenv


📁 Project Structure
RetailGenAI/
│
├── app.py
├── requirements.txt
├── README.md
├── TECHNICAL_DEEP_DIVE_QA.md
│
├── utils/
│   ├── analytics.py
│   ├── cache.py
│   ├── file_processor.py
│   ├── llm_config.py
│   ├── prompts.py
│   ├── query_router.py
│   ├── rag_pipeline.py
│   └── session_state.py
│
├── benchmarks/
│   ├── run_baseline.py
│   ├── run_benchmark_v2.py
│   ├── run_experiment_01.py
│   ├── run_experiment_02.py
│   ├── run_experiment_03.py
│   ├── run_experiment_04.py
│   ├── run_experiment_05.py
│   ├── run_experiment_06.py
│   ├── run_experiment_07.py
│   ├── run_experiment_08.py
│   ├── run_experiment_09.py
│   └── results/
│
├── data/
│   └── evaluations/
│
└── pages/
    ├── Analytics.py
    └── Metrics.py
📥 Data Processing
CSV
CSV rows are converted into individual documents.
Example:
record_id: 96
campaign_name: Black Friday - North
region: North
Revenue: 108000
...
Metadata is attached to each row:
record_id
campaign_name
region
start_date
This makes record-level retrieval more targeted.
PDF / TXT / JSON
General text is split using:
RecursiveCharacterTextSplitter
chunk_size = 1000
chunk_overlap = 200
🔎 RAG Pipeline
Question
   ↓
Chroma Retriever
   ↓
Retrieved Documents
   ↓
format_docs()
   ↓
RAG Prompt
   ↓
Gemini
   ↓
StrOutputParser
   ↓
Answer
The RAG pipeline is wrapped by deterministic analytics and caching.
📊 Deterministic Analytics
The application does not send every question to Gemini.
Questions that can be answered exactly with Pandas are handled locally.
Examples
What was the Revenue of record 96?

Which campaign record generated the highest Revenue?

Compare the Revenue of records 36 and 96.

What is the total Revenue of records 47 and 74?
Validated results
Query	Result
Record 96 Revenue	108000
Highest Revenue	Record 87 — Festive Flash Sale - South — 115000
Records 36 vs 96	100500 vs 108000
Total Revenue of 47 + 74	200500


For unavailable structured information:
Information not available in uploaded data.
🧪 Experiments
The project was developed through controlled experiments measuring retrieval quality, answer accuracy, and latency.
Baseline
Configuration:
Records:        100
Questions:      12
Chunk size:     1000
Overlap:        200
Embedding:      all-MiniLM-L6-v2
Vector DB:      ChromaDB
Top K:          5
LLM:            Gemini
Results:
Metric	Baseline
Recall@5	83.33%
MRR	0.5972
Answer Accuracy	83.33%
Average Latency	2.7858 s
P50	2.4875 s
P95	4.5102 s
P99	4.6585 s


Experiment 1 — Row-aware Chunking
Decision: KEEP
Metric	Result
Retrieval / Answer Accuracy	100%
Average Latency	1.2523 s
P50	1.2119 s
P95	1.4135 s
P99	1.4166 s


Average latency was approximately 55% lower than baseline.
Experiment 2 — Metadata-aware Retrieval
Decision: KEEP
Metric	Result
Recall	100%
MRR	1.0
Answer Accuracy	100%
Indexing Time	2.0932 s
Average Latency	1.6428 s
P95	2.0716 s
P99	2.2473 s


Experiment 3 — Query Transformation / Routing
Decision: KEEP
The rule-based router classified questions into:
exact_lookup
metadata_filter
semantic
multi_condition
comparison
aggregation
analytics
The tested benchmark achieved:
Retrieval / Answer Accuracy: 100%
MRR:                         1.0
Experiment 4 — Hybrid Dense + BM25 + RRF
Decision: REJECT
Results:
Metric	Result
Recall / Accuracy	100%
MRR	0.8194
Average Latency	3.9706 s
P95	5.8483 s


The additional retrieval complexity increased latency without providing a measured quality benefit sufficient to justify it.
Experiment 5 — Cross-encoder Reranking
Decision: REJECT
Model:
cross-encoder/ms-marco-MiniLM-L-6-v2
Results:
Metric	Result
Recall	100%
Answer Accuracy	100%
MRR	1.0
Average Latency	2.8743 s
P95	6.2437 s


Key lesson
A reranker can reorder retrieved candidates, but cannot recover a relevant document that was never retrieved into the candidate set.
Experiment 6 — Deterministic Analytics
Decision: KEEP
Pandas handles:
- Highest Revenue
- Highest ROI
- Exact record lookup
- Record comparisons
- Average Revenue
- Total Revenue
- Multi-condition filtering
This reduces unnecessary LLM calls and improves numerical reliability.
Experiment 7 — Exact-match Cache
Decision: KEEP
Controlled test
Requests:       6
Cache hits:     3
LLM calls:      3
Constructed hit rate: 50%
Real Gemini test
Requests:        8
Gemini calls:    1
Cache hits:      1
Average latency: 0.2603 s
P50:             0.0006 s
P95:             2.0749 s
P99:             2.0749 s
The 50% figure is a controlled benchmark result, not a production cache-hit-rate claim.
Experiment 8 — Streaming
Decision: KEEP
Streaming was evaluated using Time To First Token (TTFT) and total generation time.
Questions:        3

Average TTFT:     16.8035 s
P50 TTFT:         10.7482 s
P95 TTFT:         29.5543 s

Average Total:    19.9163 s
P50 Total:        13.5210 s
P95 Total:        33.0769 s
Conclusion
Streaming improves perceived responsiveness but does not necessarily reduce total generation latency.
Experiment 9 — Dataset-level Analytics Routing
Decision: KEEP
Questions:            14
Deterministic:        13
RAG / LLM fallback:   1
Deterministic Route:  92.86%
Answer Accuracy:      100%
Measurement caveat: the unified benchmark assigned perfect retrieval credit to deterministic queries. Therefore, these unified retrieval metrics should not be presented as pure RAG retrieval metrics.

💡 Engineering Lessons
1. RAG is not the right tool for every question
A dataset-wide numerical query should not depend on retrieving five chunks and asking an LLM to infer the maximum.
2. Deterministic computation improves reliability
If Pandas can calculate the answer exactly, there is no reason to make an LLM calculate it.
3. Retrieval quality and answer quality are different
Retrieving the correct context does not guarantee that the LLM will generate a correct answer.
4. Reranking has a candidate-set limitation
A reranker can only reorder documents that were already retrieved.
5. Every optimization needs a trade-off analysis
A retrieval technique should be evaluated using both:
Quality + Latency
not quality alone.
⚠️ Known Limitation
Consider:
Tell me the best revenue campaign,
why it performed well,
and how to improve the remaining campaigns.
A pure top-k RAG approach can retrieve only a small subset of the dataset and incorrectly infer the dataset-wide winner.
The verified highest-Revenue result in the evaluated dataset is:
Record:   87
Campaign: Festive Flash Sale - South
Revenue:  115000
Better architecture
User Question
      ↓
Deterministic Dataset Analysis
      ↓
Verified Metrics
      ↓
Gemini
      ↓
Business Explanation / Recommendations
This keeps factual calculations deterministic while using the LLM for explanation.
🧠 Interview Preparation
A detailed technical interview guide is available in:
TECHNICAL_DEEP_DIVE_QA.md
It covers:
- RAG
- embeddings
- ChromaDB
- chunking
- metadata retrieval
- query routing
- BM25
- RRF
- reranking
- deterministic analytics
- caching
- streaming
- Recall@5
- MRR
- latency
- benchmark design
- failure cases
- production scaling
- likely interviewer follow-ups
⚙️ Setup
1. Clone the repository
git clone <your-repository-url>
cd RetailGenAI
2. Create a virtual environment
python -m venv .venv
Windows
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure Gemini
Create .env:
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
Never commit .env.
5. Run
streamlit run app.py
🔐 GitHub Safety
Do not commit:
.env
.venv/
vector_store/
data/uploads/
.env.example should contain placeholders only.
🚀 Future Improvements
- Redis-based distributed caching
- Stronger intent classification
- Dataset-level metric extraction before LLM generation
- Automated faithfulness evaluation
- Retry and exponential backoff for transient LLM failures
- Production API layer
- Authentication and multi-user isolation
- Observability and tracing
- Persistent vector-store lifecycle management
⭐ Project Takeaway
RetailGenAI 2.0 is not just a RAG chatbot.
It is a hybrid analytics + RAG system where:
Structured questions → Deterministic computation
Semantic questions   → RAG + LLM
The project was evaluated experimentally and components were retained or rejected based on measured quality, latency, and engineering trade-offs.