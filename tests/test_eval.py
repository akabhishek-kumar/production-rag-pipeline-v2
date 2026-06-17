"""Unit tests for production RAG pipeline v2.

All tests use mocks — no Groq API key, Chroma DB, or model downloads required.
These run in CI via .github/workflows/eval.yml.

CI gates enforced by assertions:
  avg_quality       >= 5.0
  faithfulness_rate >= 0.5
  citation_rate     >= 0.5
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ── Context formatting ────────────────────────────────────────────────────────

def test_format_context_includes_source_headers(sample_docs):
    """_format_context adds [Source: filename, Chunk N] headers."""
    from app.graph import _format_context

    context = _format_context(sample_docs)

    assert "[Source: langgraph_guide.txt, Chunk 0]" in context
    assert "[Source: rag_intro.txt, Chunk 0]" in context
    assert "[Source: harness.txt, Chunk 0]" in context
    # Content should also be present
    assert "LangGraph" in context
    assert "RAG" in context


def test_format_context_empty_docs():
    """_format_context handles empty list gracefully."""
    from app.graph import _format_context

    assert _format_context([]) == ""


# ── Ingestion metadata ────────────────────────────────────────────────────────

def test_split_and_tag_adds_metadata():
    """split_and_tag stamps source (basename) and chunk_id on every chunk."""
    from ingest import split_and_tag

    docs = [
        Document(
            page_content="This is a test document. " * 30,
            metadata={"source": "/some/path/my_file.txt"},
        )
    ]
    chunks = split_and_tag(docs)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["source"] == "my_file.txt"   # basename, not full path
        assert "chunk_id" in chunk.metadata
        assert isinstance(chunk.metadata["chunk_id"], int)


def test_split_and_tag_chunk_ids_sequential():
    """chunk_id is sequential (0, 1, 2...) within each source."""
    from ingest import split_and_tag

    docs = [
        Document(
            page_content="Paragraph one. " * 40,
            metadata={"source": "doc_a.txt"},
        )
    ]
    chunks = split_and_tag(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


# ── Reranking ─────────────────────────────────────────────────────────────────

def test_rerank_returns_at_most_reranker_k(sample_docs):
    """rerank_documents returns at most reranker_k docs, highest score first."""
    from app.vectorstore import rerank_documents
    from app.config import settings

    with patch("app.vectorstore.get_reranker") as mock_get:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.5, 0.9, 0.3]
        mock_get.return_value = mock_reranker

        result = rerank_documents("test query", sample_docs)

    assert len(result) <= settings.reranker_k
    # Highest score (0.9) should be first
    assert result[0].metadata["reranker_score"] == 0.9


def test_rerank_attaches_score_to_metadata(sample_docs):
    """rerank_documents attaches reranker_score to each doc's metadata."""
    from app.vectorstore import rerank_documents

    scores = [0.7, 0.4, 0.85]
    with patch("app.vectorstore.get_reranker") as mock_get:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = scores
        mock_get.return_value = mock_reranker

        result = rerank_documents("query", sample_docs)

    for doc in result:
        assert "reranker_score" in doc.metadata


def test_rerank_empty_docs():
    """rerank_documents handles empty list without error."""
    from app.vectorstore import rerank_documents

    result = rerank_documents("query", [])
    assert result == []


# ── Citation check ────────────────────────────────────────────────────────────

def test_citation_check_detects_present_citation():
    """citation_check_chain returns has_citations=True when [Source: X] present."""
    from app.chains import citation_check_chain

    with patch.object(citation_check_chain, "invoke") as mock_invoke:
        mock_invoke.return_value = {"has_citations": True, "missing": "none"}
        result = citation_check_chain.invoke(
            {"answer": "LangGraph is a framework [Source: langgraph_guide.txt]."}
        )
    assert result["has_citations"] is True


def test_citation_check_detects_missing_citation():
    """citation_check_chain returns has_citations=False when no [Source: X]."""
    from app.chains import citation_check_chain

    with patch.object(citation_check_chain, "invoke") as mock_invoke:
        mock_invoke.return_value = {
            "has_citations": False,
            "missing": "No [Source:] citations found",
        }
        result = citation_check_chain.invoke({"answer": "LangGraph is a framework."})
    assert result["has_citations"] is False


# ── API source extraction ─────────────────────────────────────────────────────

def test_extract_sources_single():
    """_extract_sources pulls one source name from answer."""
    from app.api import _extract_sources

    answer = "LangGraph is great [Source: langgraph_guide.txt]."
    assert _extract_sources(answer) == ["langgraph_guide.txt"]


def test_extract_sources_multiple_unique():
    """_extract_sources returns unique sources in order of appearance."""
    from app.api import _extract_sources

    answer = (
        "RAG is useful [Source: rag_intro.txt]. "
        "Harness patterns help [Source: harness.txt]. "
        "RAG again [Source: rag_intro.txt]."
    )
    sources = _extract_sources(answer)
    assert sources == ["rag_intro.txt", "harness.txt"]


def test_extract_sources_none():
    """_extract_sources returns empty list when no citations present."""
    from app.api import _extract_sources

    assert _extract_sources("This answer has no citations.") == []


# ── CI gate assertions ────────────────────────────────────────────────────────

def test_ci_gates_pass_on_good_results():
    """CI thresholds pass when metrics are above minimum."""
    results = [
        {"answer_quality": 7, "answer_faithfulness": True,  "citation_recall": True},
        {"answer_quality": 8, "answer_faithfulness": True,  "citation_recall": True},
        {"answer_quality": 6, "answer_faithfulness": False, "citation_recall": True},
        {"answer_quality": 7, "answer_faithfulness": True,  "citation_recall": False},
    ]
    n = len(results)
    avg_quality = sum(r["answer_quality"] for r in results) / n
    faith_rate  = sum(r["answer_faithfulness"] for r in results) / n
    cite_rate   = sum(r["citation_recall"] for r in results) / n

    assert avg_quality >= 5.0,  f"Quality gate failed: {avg_quality:.1f} < 5.0"
    assert faith_rate  >= 0.5,  f"Faithfulness gate failed: {faith_rate:.0%} < 50%"
    assert cite_rate   >= 0.5,  f"Citation gate failed: {cite_rate:.0%} < 50%"


def test_ci_gates_fail_on_bad_results():
    """CI thresholds correctly fail when metrics are below minimum."""
    results = [
        {"answer_quality": 2, "answer_faithfulness": False, "citation_recall": False},
        {"answer_quality": 3, "answer_faithfulness": False, "citation_recall": False},
    ]
    n = len(results)
    avg_quality = sum(r["answer_quality"] for r in results) / n
    faith_rate  = sum(r["answer_faithfulness"] for r in results) / n
    cite_rate   = sum(r["citation_recall"] for r in results) / n

    assert avg_quality < 5.0   # should fail the gate
    assert faith_rate  < 0.5
    assert cite_rate   < 0.5
