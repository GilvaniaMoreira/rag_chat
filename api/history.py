"""Módulo para gerenciar histórico de conversas persistido por usuário."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DB_PATH = BASE_DIR / "conversation_history.db"


def get_db_connection() -> sqlite3.Connection:
    """Cria e retorna uma conexão com o banco de dados SQLite."""
    # Garante que o diretório existe
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db() -> None:
    """Inicializa o banco de dados criando as tabelas necessárias."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                top_k INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_conversation(
    user_id: str,
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    top_k: int,
) -> int:
    """Salva uma conversa no banco de dados e retorna o ID."""
    init_history_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO conversations (user_id, question, answer, sources, top_k)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, question, answer, json.dumps(sources), top_k),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_conversation_history(
    user_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Recupera o histórico de conversas de um usuário."""
    init_history_db()
    conn = get_db_connection()
    try:
        query = """
            SELECT id, question, answer, sources, top_k, created_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY created_at ASC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        rows = conn.execute(query, (user_id,)).fetchall()
        conversations = []
        for row in rows:
            conversations.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": json.loads(row["sources"]) if row["sources"] else [],
                    "top_k": row["top_k"],
                    "created_at": row["created_at"],
                }
            )
        return conversations
    finally:
        conn.close()


def delete_conversation_history(user_id: str) -> int:
    """Deleta todo o histórico de conversas de um usuário. Retorna o número de registros deletados."""
    init_history_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE user_id = ?", (user_id,)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_conversation_count(user_id: str) -> int:
    """Retorna o número total de conversas de um usuário."""
    init_history_db()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM conversations WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()

