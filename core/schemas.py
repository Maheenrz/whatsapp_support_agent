from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    content: str = Field(min_length=1, description="The content of the source chunk.")
    source: str = Field(min_length=1, description="The source of the chunk, e.g., a document name or URL.")
    score: float = Field(ge=0, le=1, description="Relevance score of this chunk to the query.")


class SupportQuery(BaseModel):
    query: str = Field(min_length=1, description="The user's support question.")
    user_id: Optional[str] = Field(default=None, description="Identifier of the user submitting the query, if known.")


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    NO_RELEVANT_DOCS = "no_relevant_docs"
    EXPLICIT_REQUEST = "explicit_request"


class AgentResponse(BaseModel):
    answer: str = Field(min_length=1, description="The agent's answer to the user's query.")
    sources: list[SourceChunk] = Field(description="Source chunks used to construct the answer. May be empty if no relevant documents were found.")
    faithfulness_score: Optional[float] = Field(default=None, ge=0, le=1, description="A score indicating how faithful the answer is to the provided sources.")
    should_escalate: bool = Field(description="Indicates whether the response should be escalated to a human agent.")
    escalation_reason: Optional[EscalationReason] = Field(default=None, description="The reason for escalation, if applicable.")