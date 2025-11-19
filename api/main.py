"""FastAPI application exposing the RAG query endpoint."""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .chroma_client import get_qa_chain
from .schemas import QueryRequest

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Chat PDFs", version="1.0.0")


@app.get("/")
async def healthcheck() -> dict[str, str]:
    """Simple healthcheck endpoint."""
    return {"status": "ok"}


@app.post("/query")
async def query_documents(payload: QueryRequest) -> dict[str, object]:
    """Run a RAG pipeline to answer the provided question."""
    try:
        chain = get_qa_chain(top_k=payload.top_k)
        result = chain.invoke({"query": payload.question})
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

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
