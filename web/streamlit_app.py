"""Streamlit frontend for chatting with PDFs using the RAG backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
API_URL = os.getenv("API_URL", "http://localhost:8000/query")


def save_uploaded_file(uploaded_file) -> Path | None:
    """Persist uploaded files into the shared storage directory."""
    if uploaded_file is None:
        return None

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    destination = STORAGE_DIR / uploaded_file.name

    with destination.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    return destination


def call_api(question: str, top_k: int) -> Dict[str, Any]:
    """Send a query request to the FastAPI backend."""
    response = requests.post(
        API_URL,
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="RAG Chat PDFs", page_icon="📄", layout="wide")
    st.title("📄 RAG Chat PDFs")
    st.write("Faça upload de PDFs, execute o ingest e converse com seus documentos.")

    with st.sidebar:
        st.header("Upload de PDF")
        uploaded_pdf = st.file_uploader("Selecione um arquivo (PDF)", type=["pdf"])
        if uploaded_pdf and st.button("Salvar PDF"):
            saved_path = save_uploaded_file(uploaded_pdf)
            st.success(f"Arquivo salvo em {saved_path.relative_to(BASE_DIR)}. Rode `python ingest.py`." )

    st.subheader("Chat")
    question = st.text_area("Pergunta", placeholder="Ex.: Quais são os pontos principais do documento?")
    top_k = st.slider("Número de chunks (top_k)", min_value=1, max_value=10, value=4)

    if st.button("Consultar"):
        if not question.strip():
            st.warning("Digite uma pergunta antes de consultar.")
            return
        with st.spinner("Consultando a API..."):
            try:
                response = call_api(question, top_k)
            except requests.HTTPError as http_err:
                st.error(f"Erro na API: {http_err.response.text}")
                return
            except Exception as exc:  # pragma: no cover - Streamlit feedback
                st.error(f"Erro ao consultar API: {exc}")
                return

        st.success("Resposta recebida!")
        st.markdown(f"**Resposta:** {response.get('answer', 'Sem resposta')}" )

        sources = response.get("sources", [])
        if sources:
            st.markdown("**Fontes:**")
            for idx, source in enumerate(sources, start=1):
                st.write(f"{idx}. {source.get('source')} (página {source.get('page')})")
        else:
            st.info("Nenhuma fonte retornada. Verifique se o ingest foi executado.")


if __name__ == "__main__":
    main()
