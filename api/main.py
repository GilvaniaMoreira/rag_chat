"""FastAPI application exposing the RAG query endpoint."""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from .chroma_client import get_qa_chain
from .history import (
    delete_conversation_history,
    get_conversation_history,
    save_conversation,
)
from .schemas import ConversationResponse, QueryRequest

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Chat PDFs", version="1.0.0")


@app.get("/")
async def healthcheck() -> dict[str, str]:
    """Simple healthcheck endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=ConversationResponse)
async def query_documents(payload: QueryRequest) -> ConversationResponse:
    """Run a RAG pipeline to answer the provided question."""
    try:
        # Prepara o histórico de conversas para o chain
        conversation_history = []
        if payload.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in payload.conversation_history
            ]
        
        # Cria o chain com suporte a histórico se houver
        include_history = len(conversation_history) > 0
        chain = get_qa_chain(top_k=payload.top_k, include_history=include_history)
        
        # Invoca o chain com a pergunta e histórico
        chain_input = {"query": payload.question}
        if conversation_history:
            chain_input["chat_history"] = conversation_history
        
        result = chain.invoke(chain_input)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Erro ao processar consulta")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = result.get("result", "")
    sources = []
    for doc in result.get("source_documents", []):
        metadata = doc.metadata or {}
        sources.append(
            {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
            }
        )

    # Salva a conversa no histórico se user_id foi fornecido
    conversation_id = None
    if payload.user_id:
        try:
            conversation_id = save_conversation(
                user_id=payload.user_id,
                question=payload.question,
                answer=answer,
                sources=sources,
                top_k=payload.top_k,
            )
        except Exception as exc:
            logger.warning(f"Erro ao salvar conversa no histórico: {exc}")

    return ConversationResponse(
        answer=answer, sources=sources, conversation_id=conversation_id
    )


@app.get("/history/{user_id}")
async def get_history(
    user_id: str, limit: int = Query(None, ge=1, le=100, description="Limite de conversas a retornar")
) -> dict[str, object]:
    """Recupera o histórico de conversas de um usuário."""
    try:
        conversations = get_conversation_history(user_id, limit=limit)
        return {"user_id": user_id, "conversations": conversations, "count": len(conversations)}
    except Exception as exc:
        logger.exception("Erro ao recuperar histórico")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/history/{user_id}")
async def delete_history(user_id: str) -> dict[str, object]:
    """Deleta todo o histórico de conversas de um usuário."""
    try:
        deleted_count = delete_conversation_history(user_id)
        return {
            "user_id": user_id,
            "deleted_count": deleted_count,
            "message": f"Histórico deletado com sucesso. {deleted_count} conversa(s) removida(s).",
        }
    except Exception as exc:
        logger.exception("Erro ao deletar histórico")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
