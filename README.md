# production-rag-pipeline-v2

A **production-grade RAG pipeline** built on LangGraph with four key upgrades over v1: hybrid retrieval, cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline.

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Retrieval | Vector search only (Chroma) | **Hybrid: BM25 + vector (EnsembleRetriever)** |
| Ranking | Cosine similarity | **Cross-encoder reranking (sentence-transformers)** |
| Citations | None | **Enforced — [Source: filename] in every answer** |
| CI | None | **GitHub Actions — eval gates block failing PRs** |
| Eval metrics | 4 (relevance, faithfulness, quality, recall) | **5 — adds citation recall** |
| Graph nodes | 7 | **9 — adds citation_check** |
| Source metadata | No | **Yes — source + chunk_id on every chunk** |

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph 0.2+ |
| LLM | Groq — llama-3.1-8b-instant (free tier) |
| Embeddings | FastEmbed BAAI/bge-small-en-v1.5 (ONNX) |
| Vector Store | Chroma (persistent on disk) |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| API | FastAPI + uvicorn |
| CI | GitHub Actions |
| Testing | pytest (mocked, no API key needed) |

## Architecture — 9-Node LangGraph Pipeline

```
Question
   │
   ▼
[retrieve] ── hybrid BM25 + vector → cross-encoder rerank → top-k
   │
   ▼
[filter_docs] ── LLM grades each chunk for relevance
   │
   ├── no relevant docs ──► [no_info] ──► END
   │
   ▼
[generate] ── RAG chain with [Source: filename] citation enforcement
   │
   ▼
[citation_check] ── verifies [Source: X] present in answer        ← v2
   │
   ├── no citation ──► [rewrite] ──► [retrieve]
   │
   ▼
[verify_answer] ── hallucination check (answer grounded in context?)
   │
   ├── not grounded ──► [rewrite] ──► [retrieve]
   │
   ▼
[evaluate] ── grades answer quality 1-10
   │
   ├── score < threshold ──► [rewrite] ──► [retrieve]
   │
   └── score >= threshold ──► END
```

## Why Hybrid + Reranking

**BM25 catches what vector search misses.** Exact keyword matches — acronyms, proper nouns, version numbers — often score low in cosine similarity because they're rare in embedding space. BM25 handles them natively.

**Cross-encoder reranking is more accurate than bi-encoder similarity.** A bi-encoder embeds query and document separately; the cross-encoder sees them together in one forward pass, letting it model query-document interactions directly. The tradeoff is speed — which is why we use it only on the top candidates from the ensemble, not across the whole corpus.

## Project Structure

```
app/
├── config.py       # pydantic-settings — all tunable params in .env
├── vectorstore.py  # hybrid retriever + cross-encoder reranker
├── chains.py       # 6 chains: rag (citation-enforced), grade, rewrite,
│                   #           relevance, hallucination, citation_check
├── graph.py        # 9-node LangGraph, lazy retriever init
└── api.py          # POST /chat/ — response includes sources[] list
ingest.py           # load docs → split+tag → Chroma + bm25_index.pkl
evaluate.py         # 5-metric RAGAS-style eval with CI gate summary
tests/
├── conftest.py     # in-memory fixtures, mock retriever
└── test_eval.py    # 14 unit tests — all mocked, no API key needed
.github/
└── workflows/
    └── eval.yml    # CI: pytest on every push/PR to main
```

## Quick Start

```bash
git clone https://github.com/akabhishek-kumar/production-rag-pipeline-v2
cd production-rag-pipeline-v2
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # add your Groq API key

# Step 1 — drop documents into docs/ then ingest
python ingest.py

# Step 2 — start the API
uvicorn main:app --reload
```

API live at **http://localhost:8000** · Docs at **http://localhost:8000/docs**

## API

```
POST /chat/
{
  "question": "What is Harness Engineering?",
  "session_id": "user-123"
}

Response:
{
  "answer": "Harness Engineering is... [Source: harness.txt].",
  "session_id": "user-123",
  "sources": ["harness.txt"]
}
```

## Evaluation

```bash
python evaluate.py
```

Output includes CI gate summary:

```
CI GATES
  Quality >= 5.0  : PASS
  Faithful >= 50% : PASS
  Citations >= 50%: PASS
```

## Running Tests (CI)

```bash
pytest tests/ -v
```

All 14 tests run without a Groq API key, Chroma DB, or model downloads.
GitHub Actions runs this on every push and PR — a failing gate blocks the merge.

---

← [v1: production-rag-pipeline](https://github.com/akabhishek-kumar/production-rag-pipeline) | Part of my AI Engineering series → [GitHub](https://github.com/akabhishek-kumar)
