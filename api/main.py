"""FastAPI application exposing the RAG query endpoint."""
from __future__ import annotations

import csv
import io
import logging
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, Response

from .chroma_client import get_qa_chain
from .history import (
    delete_conversation_history,
    get_conversation_history,
    save_conversation,
)
from .metrics import (
    get_document_usage_raw,
    get_error_stats,
    get_errors_raw,
    get_queries_raw,
    get_query_stats,
    get_time_series_data,
    get_top_documents,
    get_top_users,
    get_user_stats,
    record_error,
    record_query,
)
from .schemas import ConversationResponse, QueryRequest

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Chat PDFs", version="1.0.0")


def _serialize_to_csv(records: list[dict[str, object]], fieldnames: list[str]) -> str:
    """Serialize metric records to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field) for field in fieldnames})
    return output.getvalue()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware para capturar métricas de requisições HTTP."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            # Registra erros se o status code for >= 400
            if response.status_code >= 400:
                try:
                    record_error(
                        user_id=None,  # user_id será capturado no endpoint específico
                        endpoint=request.url.path,
                        error_type=f"HTTP_{response.status_code}",
                        error_message=f"HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                except Exception as exc:
                    logger.warning(f"Erro ao registrar métrica de erro: {exc}")
            
            return response
        except Exception as exc:
            # Registra exceção não tratada
            try:
                record_error(
                    user_id=None,
                    endpoint=request.url.path,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    status_code=500,
                )
            except Exception as metric_exc:
                logger.warning(f"Erro ao registrar métrica de erro: {metric_exc}")
            raise


app.add_middleware(MetricsMiddleware)


@app.get("/")
async def healthcheck() -> dict[str, str]:
    """Simple healthcheck endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=ConversationResponse)
async def query_documents(payload: QueryRequest) -> ConversationResponse:
    """Run a RAG pipeline to answer the provided question."""
    start_time = time.time()
    result = None
    success = False
    error_message = None
    sources = []
    
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
        success = True
        
        # Processa os resultados
        answer = result.get("result", "")
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
        
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Erro ao processar consulta")
        error_message = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Registra métricas da consulta
        response_time_ms = (time.time() - start_time) * 1000
        try:
            record_query(
                user_id=payload.user_id,
                question=payload.question,
                top_k=payload.top_k,
                response_time_ms=response_time_ms,
                success=success,
                sources=sources,
                error_message=error_message,
            )
        except Exception as exc:
            logger.warning(f"Erro ao registrar métrica de consulta: {exc}")


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


@app.get("/metrics/stats")
async def get_stats(
    user_id: str | None = Query(None, description="ID do usuário para filtrar estatísticas"),
    days: int = Query(30, ge=1, le=365, description="Número de dias para considerar"),
) -> dict[str, object]:
    """Retorna estatísticas gerais de uso."""
    try:
        stats = get_query_stats(user_id=user_id, days=days)
        return stats
    except Exception as exc:
        logger.exception("Erro ao recuperar estatísticas")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/user/{user_id}")
async def get_user_metrics(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Número de dias para considerar"),
) -> dict[str, object]:
    """Retorna estatísticas de um usuário específico."""
    try:
        stats = get_user_stats(user_id=user_id, days=days)
        return {"user_id": user_id, **stats}
    except Exception as exc:
        logger.exception("Erro ao recuperar estatísticas do usuário")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/top-users")
async def get_top_users_metrics(
    limit: int = Query(10, ge=1, le=100, description="Número de usuários a retornar"),
    days: int = Query(30, ge=1, le=365, description="Número de dias para considerar"),
) -> dict[str, object]:
    """Retorna os usuários mais ativos."""
    try:
        top_users = get_top_users(limit=limit, days=days)
        return {"top_users": top_users, "limit": limit, "days": days}
    except Exception as exc:
        logger.exception("Erro ao recuperar top usuários")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/top-documents")
async def get_top_documents_metrics(
    limit: int = Query(10, ge=1, le=100, description="Número de documentos a retornar"),
    days: int = Query(30, ge=1, le=365, description="Número de dias para considerar"),
) -> dict[str, object]:
    """Retorna os documentos mais consultados."""
    try:
        top_docs = get_top_documents(limit=limit, days=days)
        return {"top_documents": top_docs, "limit": limit, "days": days}
    except Exception as exc:
        logger.exception("Erro ao recuperar top documentos")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/errors")
async def get_errors_metrics(
    days: int = Query(30, ge=1, le=365, description="Número de dias para considerar"),
) -> dict[str, object]:
    """Retorna estatísticas de erros."""
    try:
        error_stats = get_error_stats(days=days)
        return {"days": days, **error_stats}
    except Exception as exc:
        logger.exception("Erro ao recuperar estatísticas de erros")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/time-series")
async def get_time_series_metrics(
    days: int = Query(7, ge=1, le=90, description="Número de dias para considerar"),
    user_id: str | None = Query(None, description="ID do usuário para filtrar"),
) -> dict[str, object]:
    """Retorna dados de séries temporais para gráficos."""
    try:
        time_series = get_time_series_data(days=days, user_id=user_id)
        return {"days": days, "user_id": user_id, "time_series": time_series}
    except Exception as exc:
        logger.exception("Erro ao recuperar séries temporais")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/export")
async def export_metrics_data(
    data_type: Literal["queries", "errors", "documents"] = Query(
        "queries", description="Tipo de dado para exportar"
    ),
    export_format: Literal["json", "csv"] = Query(
        "json", description="Formato do arquivo de saída"
    ),
    user_id: str | None = Query(
        None,
        description="Filtra por usuário (aplicável apenas para exportação de consultas)",
    ),
    days: int = Query(30, ge=1, le=365, description="Número de dias para considerar"),
) -> Response:
    """Exporta métricas em formatos CSV ou JSON."""
    try:
        if data_type == "queries":
            records = get_queries_raw(user_id=user_id, days=days)
            fieldnames = [
                "id",
                "user_id",
                "question",
                "top_k",
                "response_time_ms",
                "success",
                "error_message",
                "sources_count",
                "created_at",
            ]
        elif data_type == "errors":
            records = get_errors_raw(days=days)
            fieldnames = [
                "id",
                "user_id",
                "endpoint",
                "error_type",
                "error_message",
                "status_code",
                "created_at",
            ]
        else:
            records = get_document_usage_raw(days=days)
            fieldnames = [
                "id",
                "query_id",
                "user_id",
                "question",
                "source_path",
                "page",
                "created_at",
                "query_created_at",
            ]
    except Exception as exc:
        logger.exception("Erro ao preparar exportação de métricas")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = f"{data_type}_metrics_{days}d.{export_format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    if export_format == "json":
        return JSONResponse(content=records, headers=headers)

    csv_content = _serialize_to_csv(records, fieldnames)
    return Response(content=csv_content, media_type="text/csv", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
