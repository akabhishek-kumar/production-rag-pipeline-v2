"""LangGraph pipeline — production RAG v2.

v2 vs v1 — what changed:
  - retrieve_node: uses hybrid (BM25 + vector) retriever + cross-encoder reranking
  - retrieved_docs / filtered_docs: now carry full Document objects (metadata intact)
  - _format_context(): labels each chunk with [Source: filename, Chunk N] for citations
  - citation_check node: new — verifies answer contains [Source: X] before hallucination check
  - has_citations: new state field
  - Lazy retriever init: retriever built on first request, not at import, so tests can patch

9-node graph:
  retrieve → filter_docs → no_info
                         ↘ generate → citation_check → verify_answer → evaluate
                                    ↘ (no citation)         ↘              ↘
                                      rewrite ←────────────────────────────┘
"""

from typing import Annotated, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.chains import (
    GradeResult,
    citation_check_chain,
    grade_chain,
    hallucination_chain,
    rag_chain,
    relevance_chain,
    rewrite_chain,
)
from app.config import settings
from app.vectorstore import (
    get_hybrid_retriever,
    load_bm25_documents,
    load_vectorstore,
    rerank_documents,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: list[Document]
    filtered_docs: list[Document]
    current_question: str
    quality_score: int
    answer_grounded: bool
    has_citations: bool          # v2
    retry_count: int


# ── Lazy retriever — avoids import-time DB load (enables test patching) ───────
_retriever: Optional[object] = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = load_vectorstore()
        bm25_docs = load_bm25_documents()
        _retriever = get_hybrid_retriever(vectorstore, bm25_docs)
    return _retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format docs with [Source: filename, Chunk N] headers for citation enforcement."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[Source: {source}, Chunk {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    """Hybrid retrieve (BM25 + vector) then cross-encoder rerank."""
    question = state["current_question"]
    docs = _get_retriever().invoke(question)
    reranked = rerank_documents(question, docs)
    print(f"[retrieve] {len(docs)} candidates → {len(reranked)} after reranking")
    return {"retrieved_docs": reranked}


def filter_docs_node(state: AgentState) -> dict:
    """Secondary LLM relevance filter on top of reranking."""
    question = state["current_question"]
    relevant = []
    for doc in state["retrieved_docs"]:
        result: dict = relevance_chain.invoke({
            "question": question,
            "document": doc.page_content,
        })
        if result.get("is_relevant", False):
            relevant.append(doc)
        else:
            print(f"[filter] dropped: {doc.page_content[:60]}...")
    print(f"[filter] kept {len(relevant)}/{len(state['retrieved_docs'])} chunks")
    return {"filtered_docs": relevant}


def no_info_node(state: AgentState) -> dict:
    msg = (
        "I don't have enough information in my knowledge base to answer that. "
        "Try rephrasing, or add relevant documents and re-run ingest.py."
    )
    return {"messages": [AIMessage(content=msg)], "quality_score": 0}


def generate_node(state: AgentState) -> dict:
    context = _format_context(state["filtered_docs"])
    history = state["messages"][:-1]
    answer = rag_chain.invoke({
        "question": state["current_question"],
        "context": context,
        "history": history,
    })
    return {"messages": [AIMessage(content=answer)]}


def citation_check_node(state: AgentState) -> dict:
    """v2: verify answer contains [Source: ...] citations."""
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    result: dict = citation_check_chain.invoke({"answer": last_ai.content})
    has_citations = result.get("has_citations", False)
    print(f"[citation] has_citations={has_citations} — {result.get('missing', '')}")
    return {"has_citations": has_citations}


def verify_answer_node(state: AgentState) -> dict:
    """Hallucination check — is every claim grounded in retrieved context?"""
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    context = _format_context(state["filtered_docs"])
    result: dict = hallucination_chain.invoke({
        "context": context,
        "answer": last_ai.content,
    })
    grounded = result.get("grounded", False)
    print(f"[verify] grounded={grounded} — {result.get('explanation', '')}")
    return {"answer_grounded": grounded}


def evaluate_node(state: AgentState) -> dict:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    result: GradeResult = grade_chain.invoke({
        "question": state["current_question"],
        "answer": last_ai.content,
    })
    print(f"[evaluate] score={result.score}/10 — {result.reasoning}")
    return {"quality_score": result.score}


def rewrite_node(state: AgentState) -> dict:
    rewritten = rewrite_chain.invoke({
        "question": state["current_question"],
        "history": state["messages"],
    })
    print(f"[rewrite] '{state['current_question']}' -> '{rewritten}'")
    return {
        "current_question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_filter(state: AgentState) -> Literal["generate", "no_info"]:
    return "generate" if state["filtered_docs"] else "no_info"


def route_after_citation(
    state: AgentState,
) -> Literal["verify_answer", "rewrite", "__end__"]:
    if state["has_citations"]:
        return "verify_answer"
    if state["retry_count"] >= settings.max_retries:
        print("[route] no citations but max retries reached -> END")
        return "__end__"
    print("[route] no citations -> rewrite")
    return "rewrite"


def route_after_verify(
    state: AgentState,
) -> Literal["evaluate", "rewrite", "__end__"]:
    if state["answer_grounded"]:
        return "evaluate"
    if state["retry_count"] >= settings.max_retries:
        print("[route] not grounded but max retries reached -> END")
        return "__end__"
    print("[route] not grounded -> rewrite")
    return "rewrite"


def route_after_evaluate(
    state: AgentState,
) -> Literal["rewrite", "__end__"]:
    if state["quality_score"] >= settings.grade_threshold:
        print(f"[route] score {state['quality_score']} >= {settings.grade_threshold} -> END")
        return "__end__"
    if state["retry_count"] >= settings.max_retries:
        print("[route] max retries reached -> END anyway")
        return "__end__"
    print(f"[route] score {state['quality_score']} too low -> rewrite")
    return "rewrite"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve",       retrieve_node)
    builder.add_node("filter_docs",    filter_docs_node)
    builder.add_node("no_info",        no_info_node)
    builder.add_node("generate",       generate_node)
    builder.add_node("citation_check", citation_check_node)   # v2
    builder.add_node("verify_answer",  verify_answer_node)
    builder.add_node("evaluate",       evaluate_node)
    builder.add_node("rewrite",        rewrite_node)

    builder.add_edge(START,            "retrieve")
    builder.add_edge("retrieve",       "filter_docs")
    builder.add_edge("no_info",        END)
    builder.add_edge("generate",       "citation_check")      # v2
    builder.add_edge("rewrite",        "retrieve")

    builder.add_conditional_edges(
        "filter_docs", route_after_filter,
        {"generate": "generate", "no_info": "no_info"},
    )
    builder.add_conditional_edges(
        "citation_check", route_after_citation,               # v2
        {"verify_answer": "verify_answer", "rewrite": "rewrite", "__end__": END},
    )
    builder.add_conditional_edges(
        "verify_answer", route_after_verify,
        {"evaluate": "evaluate", "rewrite": "rewrite", "__end__": END},
    )
    builder.add_conditional_edges(
        "evaluate", route_after_evaluate,
        {"__end__": END, "rewrite": "rewrite"},
    )

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


def chat(question: str, session_id: str) -> str:
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.recursion_limit,
    }
    final = graph.invoke(
        {
            "messages":        [HumanMessage(content=question)],
            "current_question": question,
            "quality_score":   0,
            "retry_count":     0,
            "retrieved_docs":  [],
            "filtered_docs":   [],
            "answer_grounded": False,
            "has_citations":   False,
        },
        config=config,
    )
    last_ai = next(m for m in reversed(final["messages"]) if isinstance(m, AIMessage))
    return last_ai.content
