"""Módulo para gerenciar métricas de uso e monitoramento."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DB_PATH = BASE_DIR / "metrics.db"


def get_db_connection() -> sqlite3.Connection:
    """Cria e retorna uma conexão com o banco de dados SQLite de métricas."""
    # Garante que o diretório existe
    METRICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(METRICS_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_metrics_db() -> None:
    """Inicializa o banco de dados criando as tabelas necessárias."""
    conn = get_db_connection()
    try:
        # Tabela de consultas
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                top_k INTEGER,
                response_time_ms REAL,
                success BOOLEAN,
                error_message TEXT,
                sources_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        # Tabela de erros
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                endpoint TEXT,
                error_type TEXT,
                error_message TEXT,
                status_code INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        # Tabela de uso de documentos (sources)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                source_path TEXT,
                page INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES queries(id)
            )
            """
        )
        
        # Índices para melhor performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_user_id ON queries(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_success ON queries(success)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_errors_created_at ON errors(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_usage_source ON document_usage(source_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_usage_query_id ON document_usage(query_id)")
        
        conn.commit()
    finally:
        conn.close()


def record_query(
    user_id: Optional[str],
    question: str,
    top_k: int,
    response_time_ms: float,
    success: bool,
    sources: List[Dict[str, Any]],
    error_message: Optional[str] = None,
) -> int:
    """Registra uma consulta no banco de métricas e retorna o ID."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO queries (user_id, question, top_k, response_time_ms, success, error_message, sources_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, question, top_k, response_time_ms, success, error_message, len(sources)),
        )
        query_id = cursor.lastrowid
        
        # Registra o uso de documentos
        for source in sources:
            conn.execute(
                """
                INSERT INTO document_usage (query_id, source_path, page)
                VALUES (?, ?, ?)
                """,
                (query_id, source.get("source"), source.get("page")),
            )
        
        conn.commit()
        return query_id
    finally:
        conn.close()


def record_error(
    user_id: Optional[str],
    endpoint: str,
    error_type: str,
    error_message: str,
    status_code: Optional[int] = None,
) -> int:
    """Registra um erro no banco de métricas e retorna o ID."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO errors (user_id, endpoint, error_type, error_message, status_code)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, endpoint, error_type, error_message, status_code),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_query_stats(
    user_id: Optional[str] = None,
    days: Optional[int] = None,
) -> Dict[str, Any]:
    """Retorna estatísticas de consultas."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        where_clauses = []
        params = []
        
        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        
        if days:
            where_clauses.append("created_at >= datetime('now', '-' || ? || ' days')")
            params.append(days)
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Total de consultas
        total_row = conn.execute(
            f"SELECT COUNT(*) as count FROM queries {where_sql}",
            tuple(params),
        ).fetchone()
        total_queries = total_row["count"] if total_row else 0
        
        # Consultas bem-sucedidas
        success_row = conn.execute(
            f"SELECT COUNT(*) as count FROM queries {where_sql} AND success = 1",
            tuple(params),
        ).fetchone()
        successful_queries = success_row["count"] if success_row else 0
        
        # Consultas com erro
        error_row = conn.execute(
            f"SELECT COUNT(*) as count FROM queries {where_sql} AND success = 0",
            tuple(params),
        ).fetchone()
        failed_queries = error_row["count"] if error_row else 0
        
        # Tempo médio de resposta
        avg_time_row = conn.execute(
            f"SELECT AVG(response_time_ms) as avg_time FROM queries {where_sql} AND success = 1",
            tuple(params),
        ).fetchone()
        avg_response_time = round(avg_time_row["avg_time"], 2) if avg_time_row and avg_time_row["avg_time"] else 0.0
        
        # Tempo mínimo e máximo
        time_stats_row = conn.execute(
            f"""
            SELECT 
                MIN(response_time_ms) as min_time,
                MAX(response_time_ms) as max_time
            FROM queries 
            {where_sql} AND success = 1
            """,
            tuple(params),
        ).fetchone()
        min_response_time = round(time_stats_row["min_time"], 2) if time_stats_row and time_stats_row["min_time"] else 0.0
        max_response_time = round(time_stats_row["max_time"], 2) if time_stats_row and time_stats_row["max_time"] else 0.0
        
        # Top K mais usado
        top_k_row = conn.execute(
            f"""
            SELECT top_k, COUNT(*) as count
            FROM queries
            {where_sql}
            GROUP BY top_k
            ORDER BY count DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        most_used_top_k = top_k_row["top_k"] if top_k_row else None
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": round((successful_queries / total_queries * 100) if total_queries > 0 else 0, 2),
            "avg_response_time_ms": avg_response_time,
            "min_response_time_ms": min_response_time,
            "max_response_time_ms": max_response_time,
            "most_used_top_k": most_used_top_k,
        }
    finally:
        conn.close()


def get_user_stats(user_id: str, days: Optional[int] = None) -> Dict[str, Any]:
    """Retorna estatísticas de um usuário específico."""
    return get_query_stats(user_id=user_id, days=days)


def get_top_users(limit: int = 10, days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retorna os usuários mais ativos."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        where_clause = ""
        params = []
        
        if days:
            where_clause = "WHERE created_at >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        rows = conn.execute(
            f"""
            SELECT 
                user_id,
                COUNT(*) as query_count,
                AVG(response_time_ms) as avg_response_time,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries
            FROM queries
            {where_clause}
            GROUP BY user_id
            HAVING user_id IS NOT NULL
            ORDER BY query_count DESC
            LIMIT ?
            """,
            tuple(params + [limit]),
        ).fetchall()
        
        return [
            {
                "user_id": row["user_id"],
                "query_count": row["query_count"],
                "avg_response_time_ms": round(row["avg_response_time"], 2) if row["avg_response_time"] else 0.0,
                "successful_queries": row["successful_queries"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_top_documents(limit: int = 10, days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retorna os documentos mais consultados."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        join_clause = ""
        where_clause = ""
        params = []
        
        if days:
            join_clause = "JOIN queries q ON du.query_id = q.id"
            where_clause = "WHERE q.created_at >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        rows = conn.execute(
            f"""
            SELECT 
                du.source_path,
                COUNT(*) as usage_count,
                COUNT(DISTINCT du.query_id) as unique_queries
            FROM document_usage du
            {join_clause}
            {where_clause}
            GROUP BY du.source_path
            ORDER BY usage_count DESC
            LIMIT ?
            """,
            tuple(params + [limit]),
        ).fetchall()
        
        return [
            {
                "source_path": row["source_path"],
                "usage_count": row["usage_count"],
                "unique_queries": row["unique_queries"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_error_stats(days: Optional[int] = None) -> Dict[str, Any]:
    """Retorna estatísticas de erros."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        where_clause = ""
        params = []
        
        if days:
            where_clause = "WHERE created_at >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        # Total de erros
        total_row = conn.execute(
            f"SELECT COUNT(*) as count FROM errors {where_clause}",
            tuple(params),
        ).fetchone()
        total_errors = total_row["count"] if total_row else 0
        
        # Erros por tipo
        error_types_rows = conn.execute(
            f"""
            SELECT error_type, COUNT(*) as count
            FROM errors
            {where_clause}
            GROUP BY error_type
            ORDER BY count DESC
            """,
            tuple(params),
        ).fetchall()
        
        error_types = [
            {"error_type": row["error_type"], "count": row["count"]}
            for row in error_types_rows
        ]
        
        # Erros por endpoint
        endpoint_rows = conn.execute(
            f"""
            SELECT endpoint, COUNT(*) as count
            FROM errors
            {where_clause}
            GROUP BY endpoint
            ORDER BY count DESC
            """,
            tuple(params),
        ).fetchall()
        
        error_endpoints = [
            {"endpoint": row["endpoint"], "count": row["count"]}
            for row in endpoint_rows
        ]
        
        return {
            "total_errors": total_errors,
            "error_types": error_types,
            "error_endpoints": error_endpoints,
        }
    finally:
        conn.close()


def get_time_series_data(
    days: int = 7,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retorna dados de séries temporais para gráficos."""
    init_metrics_db()
    conn = get_db_connection()
    try:
        where_clause = "WHERE created_at >= datetime('now', '-' || ? || ' days')"
        params = [days]
        
        if user_id:
            where_clause += " AND user_id = ?"
            params.append(user_id)
        
        rows = conn.execute(
            f"""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as query_count,
                AVG(response_time_ms) as avg_response_time,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries
            FROM queries
            {where_clause}
            GROUP BY DATE(created_at)
            ORDER BY date ASC
            """,
            tuple(params),
        ).fetchall()
        
        return [
            {
                "date": row["date"],
                "query_count": row["query_count"],
                "avg_response_time_ms": round(row["avg_response_time"], 2) if row["avg_response_time"] else 0.0,
                "successful_queries": row["successful_queries"],
                "failed_queries": row["failed_queries"],
            }
            for row in rows
        ]
    finally:
        conn.close()

