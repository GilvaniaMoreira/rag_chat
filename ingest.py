"""Ingest PDFs into a persisted ChromaDB collection using LangChain."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = BASE_DIR / "chroma_db"


def load_environment() -> None:
    """Load environment variables from a local .env if present."""
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def collect_pdf_paths(storage_dir: Path) -> List[Path]:
    """Return a list of PDF file paths available for ingestion."""
    if not storage_dir.exists():
        storage_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in storage_dir.glob("**/*.pdf") if p.is_file())


def load_documents(pdf_paths: List[Path]):
    """Load documents from PDF files using PDFPlumberLoader."""
    documents = []
    for pdf_path in pdf_paths:
        loader = PDFPlumberLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks to improve retrieval quality."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(documents):
    """Create or update the Chroma vector store with the provided documents."""
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vector_store = Chroma(
        collection_name="pdf_documents",
        embedding_function=embeddings,
        persist_directory=os.getenv("CHROMA_DB_DIR", str(CHROMA_DIR)),
    )
    vector_store.add_documents(documents)
    vector_store.persist()
    return vector_store


def main() -> None:
    """Execute the ingestion pipeline."""
    load_environment()
    pdf_paths = collect_pdf_paths(STORAGE_DIR)

    if not pdf_paths:
        print("Nenhum PDF encontrado em storage/. Adicione arquivos antes de rodar o ingest.")
        return

    print(f"Encontrados {len(pdf_paths)} PDFs. Iniciando carregamento...")
    documents = load_documents(pdf_paths)
    print(f"Total de {len(documents)} documentos extraídos. Realizando chunking...")
    chunks = split_documents(documents)
    print(f"Gerados {len(chunks)} chunks. Criando embeddings e persistindo no ChromaDB...")

    for _ in tqdm(range(1), desc="Ingestão"):
        build_vector_store(chunks)

    print("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    main()
