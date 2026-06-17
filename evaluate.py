"""Evaluation script — measures RAG pipeline quality.

Run with: python evaluate.py

v2 adds a 5th metric — Citation Recall:
  Does the answer cite the sources it used? Measured by running
  citation_check_chain on the generated answer.

Metrics:
  1. Context Relevance  — % of retrieved chunks relevant to the question
  2. Answer Faithfulness — answer is grounded in retrieved context (no hallucination)
  3. Answer Quality      — LLM grades answer relevance and accuracy 1-10
  4. Keyword Recall      — expected keywords present in the answer
  5. Citation Recall     — answer contains [Source: X] citations  ← v2

CI gate thresholds (also enforced in tests/test_eval.py):
  avg_quality       >= 5.0
  faithfulness_rate >= 0.5
  avg_citation      >= 0.5
"""

from app.chains import citation_check_chain, grade_chain, hallucination_chain, rag_chain, relevance_chain
from app.vectorstore import load_vectorstore, get_hybrid_retriever, load_bm25_documents, rerank_documents

EVAL_DATASET = [
    {
        "question": "What is a UiPath Coded Agent?",
        "expected_keywords": ["LangGraph", "UiPath SDK", "code-first"],
    },
    {
        "question": "What does MCP stand for and what does it do?",
        "expected_keywords": ["Model Context Protocol", "tools", "bind_tools"],
    },
    {
        "question": "What are the five components of Harness Engineering?",
        "expected_keywords": ["Tool Registry", "Model Management", "Guardrails"],
    },
    {
        "question": "What is the difference between ingestion and query phase in RAG?",
        "expected_keywords": ["chunk", "embed", "retriev"],
    },
]


def _format_context(doc_texts: list[str], sources: list[str]) -> str:
    parts = []
    for text, source in zip(doc_texts, sources):
        parts.append(f"[Source: {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def evaluate_single(question: str, expected_keywords: list[str]) -> dict:
    vectorstore = load_vectorstore()
    bm25_docs = load_bm25_documents()
    retriever = get_hybrid_retriever(vectorstore, bm25_docs)

    # Hybrid retrieve + rerank
    docs = retriever.invoke(question)
    reranked = rerank_documents(question, docs)

    doc_texts = [doc.page_content for doc in reranked]
    doc_sources = [doc.metadata.get("source", "unknown") for doc in reranked]

    # 1. Context Relevance
    relevance_scores = []
    for doc in doc_texts:
        result = relevance_chain.invoke({"question": question, "document": doc})
        relevance_scores.append(result.get("is_relevant", False))
    context_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    # Generate from relevant chunks only
    relevant_pairs = [(t, s) for t, s, r in zip(doc_texts, doc_sources, relevance_scores) if r]
    if not relevant_pairs:
        return {
            "question": question,
            "context_relevance": 0.0,
            "answer_faithfulness": False,
            "answer_quality": 0,
            "keyword_recall": 0.0,
            "citation_recall": False,
            "answer": "No relevant context found.",
        }

    relevant_texts, relevant_sources = zip(*relevant_pairs)
    context = _format_context(list(relevant_texts), list(relevant_sources))

    answer = rag_chain.invoke({"question": question, "context": context, "history": []})

    # 2. Answer Faithfulness
    hall_result = hallucination_chain.invoke({"context": context, "answer": answer})

    # 3. Answer Quality
    grade_result = grade_chain.invoke({"question": question, "answer": answer})

    # 4. Keyword Recall
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_recall = found / len(expected_keywords) if expected_keywords else 1.0

    # 5. Citation Recall — v2
    cite_result = citation_check_chain.invoke({"answer": answer})
    has_citations = cite_result.get("has_citations", False)

    return {
        "question": question,
        "context_relevance": round(context_relevance, 2),
        "answer_faithfulness": hall_result.get("grounded", False),
        "answer_quality": grade_result.score,
        "keyword_recall": round(keyword_recall, 2),
        "citation_recall": has_citations,           # v2
        "answer": answer,
    }


def main():
    print("\nRAG Pipeline Evaluation v2")
    print("=" * 60)

    results = []
    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"\n[{i}/{len(EVAL_DATASET)}] {item['question']}")
        result = evaluate_single(item["question"], item["expected_keywords"])
        results.append(result)
        print(f"  Context Relevance  : {result['context_relevance']:.0%}")
        print(f"  Answer Faithfulness: {'PASS' if result['answer_faithfulness'] else 'FAIL'}")
        print(f"  Answer Quality     : {result['answer_quality']}/10")
        print(f"  Keyword Recall     : {result['keyword_recall']:.0%}")
        print(f"  Citation Recall    : {'PASS' if result['citation_recall'] else 'FAIL'}")  # v2

    n = len(results)
    avg_quality = sum(r["answer_quality"] for r in results) / n
    faith_rate = sum(r["answer_faithfulness"] for r in results) / n
    avg_ctx = sum(r["context_relevance"] for r in results) / n
    avg_kw = sum(r["keyword_recall"] for r in results) / n
    cite_rate = sum(r["citation_recall"] for r in results) / n   # v2

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Avg Context Relevance  : {avg_ctx:.0%}")
    print(f"  Faithfulness Pass Rate : {faith_rate:.0%}")
    print(f"  Avg Answer Quality     : {avg_quality:.1f}/10")
    print(f"  Avg Keyword Recall     : {avg_kw:.0%}")
    print(f"  Citation Pass Rate     : {cite_rate:.0%}")          # v2

    # CI gates
    print("\nCI GATES")
    print(f"  Quality >= 5.0  : {'PASS' if avg_quality >= 5.0 else 'FAIL'}")
    print(f"  Faithful >= 50% : {'PASS' if faith_rate >= 0.5 else 'FAIL'}")
    print(f"  Citations >= 50%: {'PASS' if cite_rate >= 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
