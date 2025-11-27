"""Testes para os endpoints da API relacionados a métricas."""
from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main


def create_client(monkeypatch) -> TestClient:
    """Retorna um TestClient com dependências críticas mockadas."""
    # Evita escrita real no banco durante os testes
    monkeypatch.setattr(api_main, "record_error", lambda *_, **__: None)
    return TestClient(api_main.app)


def test_metrics_stats_endpoint(monkeypatch) -> None:
    """Deve retornar as estatísticas simuladas."""
    fake_stats = {
        "total_queries": 10,
        "successful_queries": 8,
        "failed_queries": 2,
        "success_rate": 80.0,
    }
    monkeypatch.setattr(api_main, "get_query_stats", lambda **_: fake_stats)

    client = create_client(monkeypatch)
    response = client.get("/metrics/stats")
    assert response.status_code == 200
    assert response.json() == fake_stats


def test_metrics_export_json(monkeypatch) -> None:
    """Exportação JSON deve retornar o payload simulado."""
    fake_data = [
        {"id": 1, "user_id": "alice", "question": "Pergunta?", "created_at": "2024-01-01"},
    ]
    monkeypatch.setattr(api_main, "get_queries_raw", lambda **_: fake_data)

    client = create_client(monkeypatch)
    response = client.get("/metrics/export", params={"data_type": "queries", "export_format": "json"})
    assert response.status_code == 200
    assert response.json() == fake_data
    assert "attachment" in response.headers.get("content-disposition", "")


def test_metrics_export_csv(monkeypatch) -> None:
    """Exportação CSV deve gerar conteúdo tabular."""
    fake_errors = [
        {"id": 1, "user_id": None, "endpoint": "/query", "error_type": "HTTP_500", "error_message": "Erro", "status_code": 500, "created_at": "2024-01-01"},
    ]
    monkeypatch.setattr(api_main, "get_errors_raw", lambda **_: fake_errors)

    client = create_client(monkeypatch)
    response = client.get(
        "/metrics/export",
        params={"data_type": "errors", "export_format": "csv"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    body = response.text
    assert "endpoint" in body
    assert "/query" in body

