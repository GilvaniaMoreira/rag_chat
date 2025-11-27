"""Utilities to build retrievers and QA chains backed by ChromaDB."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


@lru_cache(maxsize=1)
def _embedding_model() -> OpenAIEmbeddings:
    """Instantiate the OpenAI embeddings model once and reuse."""
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


@lru_cache(maxsize=1)
def _vector_store() -> Chroma:
    """Return a cached Chroma vector store instance."""
    return Chroma(
        collection_name="pdf_documents",
        embedding_function=_embedding_model(),
        persist_directory=os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db")),
    )


def get_retriever(top_k: int = 4):
    """Create a retriever with the configured top_k value."""
    return _vector_store().as_retriever(search_kwargs={"k": top_k})


def _build_prompt(include_history: bool = False) -> ChatPromptTemplate:
    """Return the chat prompt template used by the QA chain."""
    if include_history:
        # Template com histórico de conversas
        messages = [
            (
                "system",
                "Você é um assistente especialista em documentos. Use apenas as informações do contexto. "
                "Se não encontrar a resposta, diga que não foi possível responder. "
                "Use o histórico de conversas anteriores para dar respostas mais contextuais e consistentes.",
            ),
            ("human", "Histórico de conversas anteriores:\n{chat_history}\n\nContexto:\n{context}\n\nPergunta: {question}"),
        ]
    else:
        # Template sem histórico
        messages = [
            (
                "system",
                "Você é um assistente especialista em documentos. Use apenas as informações do contexto. "
                "Se não encontrar a resposta, diga que não foi possível responder.",
            ),
            ("human", "Contexto:\n{context}\n\nPergunta: {question}"),
        ]
    
    return ChatPromptTemplate.from_messages(messages)


class SimpleRetrievalQA:
    """Lightweight RAG chain compatible com a API esperada."""

    def __init__(self, retriever, llm: ChatOpenAI, prompt: ChatPromptTemplate) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["query"]
        conversation_history = inputs.get("chat_history", [])
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Formata o histórico de conversas se existir
        chat_history_str = "Nenhuma conversa anterior."
        if conversation_history:
            history_parts = []
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    history_parts.append(f"Usuário: {content}")
                elif role == "assistant":
                    history_parts.append(f"Assistente: {content}")
            if history_parts:
                chat_history_str = "\n".join(history_parts)
        
        # Formata as mensagens do prompt
        format_kwargs = {"context": context, "question": question}
        if conversation_history:
            format_kwargs["chat_history"] = chat_history_str
        
        messages = self.prompt.format_messages(**format_kwargs)
        answer = self.llm.invoke(messages).content
        return {"result": answer, "source_documents": docs}


def get_qa_chain(top_k: int = 4, include_history: bool = False) -> SimpleRetrievalQA:
    """Return a retrieval QA pipeline built manually para compatibilidade."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", 0.0)),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    retriever = get_retriever(top_k=top_k)
    prompt = _build_prompt(include_history=include_history)
    return SimpleRetrievalQA(retriever=retriever, llm=llm, prompt=prompt)
