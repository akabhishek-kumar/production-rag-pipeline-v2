import re
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.graph import chat

router = APIRouter(prefix="/chat", tags=["agent"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is Harness Engineering?"])
    session_id: str = Field(default="default", examples=["session1"])


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = Field(
        default=[],
        description="Source filenames cited in the answer, extracted from [Source: X] tags.",
    )


def _extract_sources(answer: str) -> list[str]:
    """Pull unique source names from [Source: filename] citation tags."""
    matches = re.findall(r"\[Source:\s*([^\],]+)", answer)
    seen = []
    for m in matches:
        name = m.strip()
        if name not in seen:
            seen.append(name)
    return seen


@router.post("/", response_model=ChatResponse)
async def ask_agent(payload: ChatRequest) -> ChatResponse:
    """Send a question to the production RAG agent v2.

    Pipeline: hybrid retrieve (BM25 + vector) → rerank → filter →
              generate (with citations) → citation check → hallucination check →
              quality evaluate → retry if needed.
    """
    answer = chat(payload.question, payload.session_id)
    return ChatResponse(
        answer=answer,
        session_id=payload.session_id,
        sources=_extract_sources(answer),
    )
