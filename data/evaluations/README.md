# RetailGenAI Baseline Evaluation

This is a fixed synthetic benchmark for the first Gemini/local-embedding baseline.

Dataset:
- 100 retail campaign records
- Same file must be reused for every experiment

Evaluation:
- 12 deterministic fact questions
- Each question points to one known record
- Retrieval success can be checked by whether the relevant record appears in the retrieved context
- Generation quality will be evaluated separately

Baseline RAG configuration:
- RecursiveCharacterTextSplitter
- chunk_size = 1000
- chunk_overlap = 200
- all-MiniLM-L6-v2 embeddings
- ChromaDB
- semantic retrieval
- k = 5
- no reranker
- no hybrid search
- no query rewriting
