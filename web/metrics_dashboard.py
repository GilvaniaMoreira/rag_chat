"""Dashboard Streamlit para visualizar métricas e estatísticas de uso."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000").replace("/query", "")


def get_metrics(endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Faz uma requisição GET para um endpoint de métricas."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Erro ao buscar métricas: {exc}")
        return {}


def main() -> None:
    st.set_page_config(
        page_title="Métricas e Monitoramento - RAG Chat PDFs",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 Métricas e Monitoramento")
    st.write("Visualize estatísticas de uso, performance e erros do sistema.")

    # Sidebar com filtros
    with st.sidebar:
        st.header("Filtros")
        days = st.slider(
            "Período (dias)",
            min_value=1,
            max_value=365,
            value=30,
            help="Período de tempo para considerar nas métricas",
        )
        user_id_filter = st.text_input(
            "Filtrar por Usuário (opcional)",
            placeholder="Digite o ID do usuário",
            help="Deixe vazio para ver todas as métricas",
        )

    # Estatísticas gerais
    st.header("📈 Estatísticas Gerais")
    stats_params = {"days": days}
    if user_id_filter:
        stats_params["user_id"] = user_id_filter

    stats = get_metrics("/metrics/stats", params=stats_params)

    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total de Consultas",
                stats.get("total_queries", 0),
            )

        with col2:
            success_rate = stats.get("success_rate", 0)
            st.metric(
                "Taxa de Sucesso",
                f"{success_rate}%",
                delta=f"{stats.get('successful_queries', 0)} sucessos",
            )

        with col3:
            avg_time = stats.get("avg_response_time_ms", 0)
            st.metric(
                "Tempo Médio de Resposta",
                f"{avg_time:.2f} ms",
            )

        with col4:
            st.metric(
                "Consultas com Erro",
                stats.get("failed_queries", 0),
                delta=f"-{stats.get('failed_queries', 0)}",
                delta_color="inverse",
            )

        # Métricas adicionais
        col5, col6 = st.columns(2)
        with col5:
            st.metric(
                "Tempo Mínimo",
                f"{stats.get('min_response_time_ms', 0):.2f} ms",
            )
        with col6:
            st.metric(
                "Tempo Máximo",
                f"{stats.get('max_response_time_ms', 0):.2f} ms",
            )

        if stats.get("most_used_top_k"):
            st.info(f"Top K mais usado: {stats.get('most_used_top_k')}")

    st.divider()

    # Séries temporais
    st.header("📅 Séries Temporais")
    time_series_params = {"days": min(days, 90)}
    if user_id_filter:
        time_series_params["user_id"] = user_id_filter

    time_series_data = get_metrics("/metrics/time-series", params=time_series_params)
    if time_series_data and time_series_data.get("time_series"):
        import pandas as pd

        df = pd.DataFrame(time_series_data["time_series"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Consultas por Dia")
                st.line_chart(
                    df.set_index("date")[["query_count", "successful_queries", "failed_queries"]],
                    use_container_width=True,
                )

            with col2:
                st.subheader("Tempo Médio de Resposta")
                st.line_chart(
                    df.set_index("date")[["avg_response_time_ms"]],
                    use_container_width=True,
                )

    st.divider()

    # Top usuários e documentos
    col1, col2 = st.columns(2)

    with col1:
        st.header("👥 Top Usuários")
        top_users_params = {"limit": 10, "days": days}
        top_users_data = get_metrics("/metrics/top-users", params=top_users_params)

        if top_users_data and top_users_data.get("top_users"):
            import pandas as pd

            df_users = pd.DataFrame(top_users_data["top_users"])
            if not df_users.empty:
                st.dataframe(
                    df_users.style.format(
                        {
                            "avg_response_time_ms": "{:.2f} ms",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Nenhum dado disponível")
        else:
            st.info("Nenhum usuário encontrado")

    with col2:
        st.header("📄 Top Documentos")
        top_docs_params = {"limit": 10, "days": days}
        top_docs_data = get_metrics("/metrics/top-documents", params=top_docs_params)

        if top_docs_data and top_docs_data.get("top_documents"):
            import pandas as pd

            df_docs = pd.DataFrame(top_docs_data["top_documents"])
            if not df_docs.empty:
                st.dataframe(
                    df_docs,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Nenhum dado disponível")
        else:
            st.info("Nenhum documento encontrado")

    st.divider()

    # Estatísticas de erros
    st.header("⚠️ Estatísticas de Erros")
    errors_params = {"days": days}
    errors_data = get_metrics("/metrics/errors", params=errors_params)

    if errors_data:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total de Erros", errors_data.get("total_errors", 0))

        with col2:
            if errors_data.get("error_types"):
                st.subheader("Erros por Tipo")
                import pandas as pd

                df_errors = pd.DataFrame(errors_data["error_types"])
                st.bar_chart(df_errors.set_index("error_type"), use_container_width=True)

        if errors_data.get("error_endpoints"):
            st.subheader("Erros por Endpoint")
            import pandas as pd

            df_endpoints = pd.DataFrame(errors_data["error_endpoints"])
            st.dataframe(df_endpoints, use_container_width=True, hide_index=True)

    # Estatísticas do usuário específico (se filtrado)
    if user_id_filter:
        st.divider()
        st.header(f"👤 Estatísticas do Usuário: {user_id_filter}")

        user_stats = get_metrics(
            f"/metrics/user/{user_id_filter}",
            params={"days": days},
        )

        if user_stats:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Consultas", user_stats.get("total_queries", 0))

            with col2:
                st.metric(
                    "Taxa de Sucesso",
                    f"{user_stats.get('success_rate', 0)}%",
                )

            with col3:
                st.metric(
                    "Tempo Médio",
                    f"{user_stats.get('avg_response_time_ms', 0):.2f} ms",
                )


if __name__ == "__main__":
    main()

