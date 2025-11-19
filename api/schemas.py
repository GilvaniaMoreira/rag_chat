"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema representing a chat query."""
    question: str = Field(..., min_length=3, description="Pergunta do usuário")
    top_k: int = Field(4, ge=1, le=10, description="Número de documentos a recuperar")
