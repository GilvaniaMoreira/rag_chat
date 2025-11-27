"""Fixtures compartilhadas para a suíte de testes."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from api import metrics


@pytest.fixture()
def metrics_db_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Garante um banco de métricas isolado para cada teste."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "metrics_test.db"
        monkeypatch.setattr(metrics, "METRICS_DB_PATH", db_path)
        metrics.init_metrics_db()
        yield db_path

