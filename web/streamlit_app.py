"""Streamlit frontend for chatting with PDFs using the RAG backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
API_URL = os.getenv("API_URL", "http://localhost:8000/query")
# Extrai a URL base da API removendo /query se presente
_base_url = API_URL.replace("/query", "").rstrip("/")
HISTORY_API_URL = _base_url if _base_url else "http://localhost:8000"


def save_uploaded_file(uploaded_file) -> Path | None:
    """Persist uploaded files into the shared storage directory."""
    if uploaded_file is None:
        return None

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    destination = STORAGE_DIR / uploaded_file.name

    with destination.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    return destination


def call_api(
    question: str, top_k: int, user_id: str | None = None, conversation_history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    """Send a query request to the FastAPI backend."""
    payload = {"question": question, "top_k": top_k}
    if user_id:
        payload["user_id"] = user_id
    if conversation_history:
        payload["conversation_history"] = conversation_history
    
    response = requests.post(
        API_URL,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_conversation_history(user_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Recupera o histórico de conversas de um usuário."""
    url = f"{HISTORY_API_URL}/history/{user_id}"
    params = {}
    if limit:
        params["limit"] = limit
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("conversations", [])
    except Exception:
        return []


def delete_conversation_history(user_id: str) -> bool:
    """Deleta o histórico de conversas de um usuário."""
    url = f"{HISTORY_API_URL}/history/{user_id}"
    try:
        response = requests.delete(url, timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False


def main() -> None:
    st.set_page_config(page_title="RAG Chat PDFs", page_icon="📄", layout="wide")
    st.title("📄 RAG Chat PDFs")
    st.write("Faça upload de PDFs, execute o ingest e converse com seus documentos.")

    # Inicializa o estado da sessão
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    with st.sidebar:
        st.header("Configurações do Usuário")
        user_id_input = st.text_input(
            "ID do Usuário",
            value=st.session_state.user_id,
            placeholder="Digite seu ID de usuário",
            help="O ID do usuário permite persistir o histórico de conversas",
        )
        
        if user_id_input != st.session_state.user_id:
            st.session_state.user_id = user_id_input
            if user_id_input:
                # Carrega o histórico quando o usuário é definido
                st.session_state.conversations = get_conversation_history(user_id_input)
                st.rerun()
        
        if st.session_state.user_id:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Atualizar Histórico"):
                    st.session_state.conversations = get_conversation_history(st.session_state.user_id)
                    st.rerun()
            with col2:
                if st.button("🗑️ Limpar Histórico"):
                    if delete_conversation_history(st.session_state.user_id):
                        st.session_state.conversations = []
                        st.success("Histórico deletado!")
                        st.rerun()
                    else:
                        st.error("Erro ao deletar histórico")
        
        st.divider()
        
        st.header("Upload de PDF")
        uploaded_pdf = st.file_uploader("Selecione um arquivo (PDF)", type=["pdf"])
        if uploaded_pdf and st.button("Salvar PDF"):
            saved_path = save_uploaded_file(uploaded_pdf)
            st.success(f"Arquivo salvo em {saved_path.relative_to(BASE_DIR)}. Rode `python ingest.py`." )

    # Exibe histórico de conversas se houver
    if st.session_state.user_id and st.session_state.conversations:
        with st.expander(f"📜 Histórico de Conversas ({len(st.session_state.conversations)} conversas)", expanded=st.session_state.show_history):
            for idx, conv in enumerate(reversed(st.session_state.conversations[-10:])):  # Mostra últimas 10
                with st.container():
                    st.markdown(f"**Conversa #{len(st.session_state.conversations) - idx}** - {conv.get('created_at', '')}")
                    st.markdown(f"**Pergunta:** {conv.get('question', '')}")
                    st.markdown(f"**Resposta:** {conv.get('answer', '')}")
                    sources = conv.get('sources', [])
                    if sources:
                        with st.expander("Ver fontes"):
                            for src_idx, source in enumerate(sources, start=1):
                                st.write(f"{src_idx}. {source.get('source')} (página {source.get('page')})")
                    st.divider()

    st.subheader("Chat")
    question = st.text_area("Pergunta", placeholder="Ex.: Quais são os pontos principais do documento?")
    top_k = st.slider("Número de chunks (top_k)", min_value=1, max_value=10, value=4)

    if st.button("Consultar"):
        if not question.strip():
            st.warning("Digite uma pergunta antes de consultar.")
            return
        
        # Prepara o histórico de conversas para enviar à API
        conversation_history = []
        if st.session_state.user_id and st.session_state.conversations:
            # Converte o histórico para o formato esperado pela API
            for conv in st.session_state.conversations[-10:]:  # Envia últimas 10 conversas
                conversation_history.append({
                    "role": "user",
                    "content": conv.get("question", "")
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": conv.get("answer", "")
                })
        
        with st.spinner("Consultando a API..."):
            try:
                response = call_api(
                    question,
                    top_k,
                    user_id=st.session_state.user_id if st.session_state.user_id else None,
                    conversation_history=conversation_history if conversation_history else None,
                )
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
        
        # Atualiza o histórico local se user_id foi usado
        if st.session_state.user_id:
            # Adiciona a nova conversa ao histórico local
            new_conv = {
                "question": question,
                "answer": response.get("answer", ""),
                "sources": sources,
                "top_k": top_k,
                "created_at": "Agora",
            }
            st.session_state.conversations.append(new_conv)
            # Recarrega o histórico completo do servidor para garantir sincronização
            st.session_state.conversations = get_conversation_history(st.session_state.user_id)


if __name__ == "__main__":
    main()
