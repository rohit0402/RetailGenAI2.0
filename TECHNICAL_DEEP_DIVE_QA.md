RetailGenAI 2.0 — Technical Deep Dive Q&A
This file is an interview-preparation guide based on the implemented and measured RetailGenAI 2.0 system.
1. What problem does RetailGenAI solve?
Answer: It lets users upload retail data and ask natural-language questions about campaigns, revenue, ROI, regions and business performance. Structured questions use deterministic Pandas analytics; semantic questions fall back to ChromaDB retrieval and Gemini.
2. Why RAG?
Answer: RAG retrieves relevant information from the uploaded data before generation, allowing the knowledge source to change without retraining the model. The flow is question → retrieval → context → Gemini → answer.
3. Why not fine-tuning?
Answer: The data changes with uploads, so injecting changing facts through retrieval is more appropriate than retraining a model for every dataset.
4. Why deterministic analytics?
Answer: LLMs are unnecessary for exact calculations. Pandas can reliably answer maximum Revenue, totals, averages, comparisons and exact record lookups, improving factual reliability while reducing LLM calls.
5. What is the final request flow?
Question → Deterministic Analytics
              ├─ handled → exact answer
              └─ not handled → Cache
                                  ├─ hit → cached answer
                                  └─ miss → Chroma → Gemini
6. Why all-MiniLM-L6-v2?
Answer: It is a lightweight sentence-transformer embedding model suitable for local semantic embeddings. It was kept constant across retrieval experiments for fair comparisons.
7. What are embeddings?
Answer: Embeddings convert text into numerical vectors. Semantically similar text can then be retrieved through vector similarity even when wording differs.
8. Why ChromaDB?
Answer: It provides simple local vector storage and similarity retrieval and was sufficient for experimenting with embeddings, top-k retrieval, metadata and evaluation.
9. How did you chunk data?
Answer: General text used RecursiveCharacterTextSplitter with chunk size 1000 and overlap 200. CSV rows were made into individual documents because each row represents a natural business entity.
10. Why row-aware chunking?
Answer: Keeping a retail row together preserves the relationship between record ID, campaign, region and metrics. The experiment reduced average latency from 2.7858 s to 1.2523 s and reached 100% on the evaluated retrieval/answer benchmark.
11. What is metadata-aware retrieval?
Answer: Metadata such as record ID, campaign name, region and start date is stored with documents, allowing more targeted retrieval. The evaluated experiment achieved 100% recall, MRR 1.0 and 100% answer accuracy.
12. What is query routing?
Answer: It classifies questions into categories such as exact lookup, metadata filter, semantic, multi-condition, comparison, aggregation and analytics. The project contains a rule-based router as a supporting module; the main application path uses deterministic analytics first and then RAG fallback.
13. Why not send every question to Gemini?
Answer: Deterministic computation is more exact, faster and avoids unnecessary API calls and cost.
14. What is Recall@5?
Answer: It measures whether the relevant target appears in the first five retrieved documents. Baseline Recall@5 was 83.33%; the row-aware and metadata experiments reached 100% on their evaluated sets.
15. What is MRR?
Answer: Mean Reciprocal Rank rewards retrieving the relevant item earlier. Rank 1 contributes 1, rank 2 contributes 1/2, etc. Baseline MRR was 0.5972; metadata retrieval reached 1.0 on the evaluated benchmark.
16. Why test BM25?
Answer: BM25 provides lexical retrieval while dense retrieval provides semantic retrieval. The project tested dense + BM25 + Reciprocal Rank Fusion.
17. Why reject hybrid retrieval?
Answer: It reached 100% recall/accuracy on the test set but increased average latency to 3.9706 s and P95 to 5.8483 s, with no sufficient quality gain.
18. What is RRF?
Answer: Reciprocal Rank Fusion combines rankings from multiple retrieval systems, rewarding documents that rank highly across lists.
19. What is cross-encoder reranking?
Answer: A cross-encoder scores a query and candidate document together after first-stage retrieval, then reorders the candidate set.
20. Why reject reranking?
Answer: Recall and accuracy were already 100% and MRR was 1.0, but average latency rose to 2.8743 s and P95 to 6.2437 s. The relevant document was already in the candidate set. A reranker cannot recover a document that was never retrieved.
21. How does caching work?
Answer: The normalized question is lowercased and whitespace-normalized, then hashed with SHA-256 and used as a dictionary key. A hit avoids another Gemini request.
22. Why SHA-256?
Answer: It provides a deterministic compact cache key. It is being used for key generation, not as a security mechanism.
23. What were the cache results?
Answer: A controlled test had 6 requests, 3 hits and 3 LLM calls. A separate real Gemini test had 8 requests, 1 Gemini call and 1 cache hit. The 50% figure is a controlled workload result, not a production claim.
24. What is streaming?
Answer: Streaming displays generated output incrementally using LangChain .stream() and Streamlit st.write_stream(). It improves perceived responsiveness but does not necessarily reduce total generation time.
25. What were the streaming measurements?
Answer: For 3 questions, average TTFT was 16.8035 s and average total generation time was 19.9163 s. P95 TTFT was 29.5543 s and P95 total was 33.0769 s.
26. Why measure P50 and P95?
Answer: Average latency can hide slow-tail behavior. P50 represents the median while P95 captures slower requests that affect user experience.
27. What was the dataset-level analytics result?
Answer: In the corrected 14-question benchmark, 13/14 were handled deterministically, 1/14 used RAG fallback, deterministic routing was 92.86%, and answer accuracy was 100%. Retrieval metrics from this unified benchmark are not pure RAG metrics because deterministic queries were assigned perfect retrieval credit.
28. What important RAG failure did you find?
Answer: A broad question asking for the best-revenue campaign and improvement recommendations was routed to top-k RAG. The model incorrectly identified Record 69 even though the verified dataset-wide maximum was Record 87 with Revenue 115000. The lesson is that dataset-wide numerical ranking should be deterministic.
29. How would you fix that?
Answer: First compute verified metrics with Pandas, then pass those metrics to the LLM for explanation and recommendations:
Question → Pandas metrics → verified facts → Gemini explanation
This keeps factual computation deterministic while using the LLM where natural-language reasoning adds value.
30. How did you handle Gemini streaming failures?
Answer: The project encountered httpx.RemoteProtocolError during a streaming request. The streaming wrapper was changed to catch failures, avoid caching incomplete responses, and return a friendly error instead of a Streamlit traceback.
31. How do you reduce hallucination?
Answer: Use deterministic analytics for structured questions, retrieval-grounded prompts, explicit unavailable-data instructions, and avoid unnecessary LLM calls. A prompt alone cannot guarantee zero hallucinations, so production would also need automated faithfulness evaluation.
32. What would you change for production?
Answer: Redis for distributed caching, persistent/managed vector storage, an API layer such as FastAPI, observability, retries with exponential backoff, authentication, authorization, secret management and user-level data isolation.
33. What is the biggest engineering lesson?
Answer: Do not use an LLM for work deterministic software can do exactly. Use structured computation for exact analytics and RAG + LLM generation for semantic questions.
34. How do you explain the project in 60 seconds?
Answer:
RetailGenAI 2.0 is a retail analytics application combining deterministic data analysis with RAG. Users upload CSV, JSON, PDF or TXT data and ask natural-language questions. Structured questions such as highest revenue, record lookups, comparisons and aggregations are answered directly with Pandas, while unsupported questions fall back to ChromaDB retrieval and Gemini. I evaluated row-aware chunking, metadata retrieval, hybrid BM25+dense retrieval, cross-encoder reranking, caching and streaming. Row-aware chunking reduced average latency from about 2.79 seconds to 1.25 seconds on the evaluated benchmark. I rejected hybrid retrieval and reranking because they increased latency without improving measured quality. The main engineering lesson was to use deterministic computation where exactness matters and RAG where semantic understanding is useful.

35. What should you NOT claim?
Answer: Do not claim that every question uses RAG, that the 50% cache hit rate is a production metric, that reranking or hybrid retrieval improved the system, that streaming reduced total latency, or that unified retrieval metrics are pure RAG metrics.
36. Why is this more than a basic RAG tutorial?
Answer: The project was evaluated through controlled experiments and trade-offs rather than adding components blindly. Components were retained or rejected based on measured quality and latency. This demonstrates engineering judgment.
37. Likely follow-up questions
Be ready to answer:
1. Why Chroma instead of FAISS?
2. How does cosine similarity work?
3. BM25 vs dense retrieval?
4. Why can't a reranker recover missing documents?
5. How would you scale the cache?
6. How would you evaluate hallucination?
7. How would you handle one million records?
8. How would you isolate multiple users' datasets?
9. How would you reduce Gemini latency?
10. How would you implement retries and backoff?
11. How would you monitor RAG quality in production?
12. What happens when the embedding model changes?
13. How would you re-index updated documents?
14. How would you handle duplicates?
15. How would you choose chunk size?
16. Why is top-k retrieval insufficient for aggregation?
17. When should you use deterministic analytics instead of an LLM?
Final interview principle
The project is best described as a hybrid analytics + RAG system, not simply a RAG chatbot:
Structured question → deterministic computation
Semantic question   → RAG + LLM
That distinction is the central engineering idea behind RetailGenAI 2.0.