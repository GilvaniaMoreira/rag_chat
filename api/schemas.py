"""Pydantic schemas for API requests and responses."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Schema para uma mensagem no histórico de conversa."""
    role: str = Field(..., description="Role da mensagem: 'user' ou 'assistant'")
    content: str = Field(..., description="Conteúdo da mensagem")


class QueryRequest(BaseModel):
    """Schema representing a chat query."""
    question: str = Field(..., min_length=3, description="Pergunta do usuário")
    top_k: int = Field(4, ge=1, le=10, description="Número de documentos a recuperar")
    user_id: Optional[str] = Field(None, description="ID do usuário para persistência do histórico")
    conversation_history: Optional[List[Message]] = Field(
        None, description="Histórico de conversas anteriores"
    )


class ConversationResponse(BaseModel):
    """Schema para resposta de uma conversa."""
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: Optional[int] = None
