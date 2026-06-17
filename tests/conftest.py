"""Test fixtures for production RAG pipeline v2.

All fixtures are in-memory — no Groq API key, no Chroma on disk,
no sentence-transformers download required. Safe to run in CI.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="LangGraph is a framework for building stateful multi-actor applications with LLMs.",
            metadata={"source": "langgraph_guide.txt", "chunk_id": 0},
        ),
        Document(
            page_content="RAG stands for Retrieval Augmented Generation. It combines search with LLM generation.",
            metadata={"source": "rag_intro.txt", "chunk_id": 0},
        ),
        Document(
            page_content="Harness Engineering patterns include Tool Registry, Guardrails, and Verification Steps.",
            metadata={"source": "harness.txt", "chunk_id": 0},
        ),
    ]


@pytest.fixture
def mock_retriever(sample_docs):
    retriever = MagicMock()
    retriever.invoke.return_value = sample_docs
    return retriever
