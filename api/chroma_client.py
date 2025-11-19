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


def _build_prompt() -> ChatPromptTemplate:
    """Return the chat prompt template used by the QA chain."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente especialista em documentos. Use apenas as informações do contexto. "
                "Se não encontrar a resposta, diga que não foi possível responder.",
            ),
            ("human", "Contexto:\n{context}\n\nPergunta: {question}"),
        ]
    )


class SimpleRetrievalQA:
    """Lightweight RAG chain compatible com a API esperada."""

    def __init__(self, retriever, llm: ChatOpenAI, prompt: ChatPromptTemplate) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["query"]
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        messages = self.prompt.format_messages(context=context, question=question)
        answer = self.llm.invoke(messages).content
        return {"result": answer, "source_documents": docs}


def get_qa_chain(top_k: int = 4) -> SimpleRetrievalQA:
    """Return a retrieval QA pipeline built manually para compatibilidade."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", 0.0)),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    retriever = get_retriever(top_k=top_k)
    prompt = _build_prompt()
    return SimpleRetrievalQA(retriever=retriever, llm=llm, prompt=prompt)
