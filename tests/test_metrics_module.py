"""Testes para o módulo de métricas."""
from __future__ import annotations

from api import metrics


def test_record_query_and_document_usage(metrics_db_path) -> None:
    """Garantir que consultas e uso de documentos sejam registrados corretamente."""
    sources = [{"source": "docs/manual.pdf", "page": 3}]

    query_id = metrics.record_query(
        user_id="alice",
        question="Qual é o procedimento?",
        top_k=4,
        response_time_ms=123.45,
        success=True,
        sources=sources,
    )

    queries = metrics.get_queries_raw(user_id="alice")
    assert len(queries) == 1
    query = queries[0]
    assert query["id"] == query_id
    assert query["user_id"] == "alice"
    assert query["success"] is True
    assert query["sources_count"] == len(sources)

    documents = metrics.get_document_usage_raw()
    assert len(documents) == 1
    doc = documents[0]
    assert doc["query_id"] == query_id
    assert doc["source_path"] == "docs/manual.pdf"
    assert doc["page"] == 3


def test_record_error_and_retrieval(metrics_db_path) -> None:
    """Garantir que erros sejam registrados e recuperados."""
    metrics.record_error(
        user_id="bob",
        endpoint="/metrics/export",
        error_type="ValidationError",
        error_message="Dados inválidos",
        status_code=400,
    )

    errors = metrics.get_errors_raw()
    assert len(errors) == 1
    error = errors[0]
    assert error["user_id"] == "bob"
    assert error["endpoint"] == "/metrics/export"
    assert error["status_code"] == 400

