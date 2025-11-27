# RAG Chat PDFs

Sistema de chat com PDFs baseado em Retrieval-Augmented Generation (RAG). A aplicação combina ingestão de documentos, busca semântica com ChromaDB, geração de respostas com os modelos da OpenAI e uma interface amigável com FastAPI + Streamlit.

## Arquitetura
- `ingest.py`: prepara embeddings no ChromaDB a partir de PDFs.
- `api/`: expõe uma API FastAPI com um endpoint de consulta (`POST /query`).
- `web/`: interface Streamlit para upload de PDFs e chat.
- `storage/`: diretório observado pelo pipeline de ingestão.

## Tecnologias
- Python 3.11+
- LangChain + LangChain OpenAI
- ChromaDB (persistido em disco)
- FastAPI & Uvicorn
- Streamlit
- OpenAI API
- Docker & Docker Compose

## Funcionalidades principais
- Chat RAG com PDFs usando LangChain + OpenAI.
- Histórico de conversas persistido por usuário (API + Streamlit).
- Monitoramento completo de uso (métricas por usuário/documento, séries temporais).
- Exportação de métricas em CSV/JSON diretamente via API ou dashboard.
- Testes automatizados (pytest) e pipeline CI no GitHub Actions.

## Preparando o ambiente
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# preencha sua OPENAI_API_KEY e demais segredos
```

## Variáveis de ambiente
| Variável               | Obrigatória | Descrição                                                                 |
|------------------------|-------------|---------------------------------------------------------------------------|
| `OPENAI_API_KEY`       | Sim         | Chave usada pela ingestão e pelo backend para gerar embeddings/respostas.|
| `API_ACCESS_TOKEN`     | Sim         | Token exigido pelo endpoint `POST /query` (header `X-API-Key`).          |
| `STREAMLIT_API_TOKEN`  | Não         | Token pré-carregado na interface Streamlit (senão informe manualmente).  |
| `CHROMA_DB_DIR`        | Não         | Diretório onde o ChromaDB será persistido (padrão `chroma_db/`).         |
| `STORAGE_DIR`          | Não         | Diretório observado para ingestão (padrão `storage/`).                    |
| `OPENAI_EMBEDDING_MODEL`, `OPENAI_COMPLETION_MODEL`, `TEMPERATURE` | Não | Parametrizações opcionais para LangChain/OpenAI. |

## Autenticação e controle de acesso
- Defina `API_ACCESS_TOKEN` no `.env`. Esse token será exigido em todas as chamadas ao endpoint `POST /query` via header `X-API-Key`.
- Opcionalmente, configure `STREAMLIT_API_TOKEN` para pré-carregar o mesmo token na interface Streamlit. Caso contrário, informe manualmente na barra lateral antes de enviar perguntas ou realizar uploads.
- Sem um token válido, apenas o healthcheck (`GET /`) permanece acessível.

## Ingestão de PDFs
1. Salve os PDFs dentro de `storage/` (você pode usar o upload do Streamlit).
2. Execute o pipeline:
```bash
python ingest.py
```
Isso irá carregar os PDFs, quebrar os textos em chunks, gerar embeddings com OpenAI e persistir no diretório `chroma_db/`.

## API FastAPI
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET /` → healthcheck.
- `POST /query` → `{ "question": "...", "top_k": 4, "user_id": "...", "conversation_history": [...] }`.
- `GET /history/{user_id}` → recupera histórico de conversas de um usuário.
- `DELETE /history/{user_id}` → deleta histórico de conversas de um usuário.
- `GET /metrics/stats` → estatísticas gerais de uso.
- `GET /metrics/user/{user_id}` → estatísticas de um usuário específico.
- `GET /metrics/top-users` → lista os usuários mais ativos.
- `GET /metrics/top-documents` → lista os documentos mais consultados.
- `GET /metrics/errors` → estatísticas de erros.
- `GET /metrics/time-series` → dados de séries temporais para gráficos.
- `GET /metrics/export` → exporta métricas em CSV ou JSON (consultas, erros ou uso de documentos).

### Exemplo de requisição
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_ACCESS_TOKEN" \
  -d '{
        "question": "Quais são os prazos do contrato?",
        "top_k": 4,
        "user_id": "alice",
        "conversation_history": [
          {"role": "user", "content": "Resumo do documento?"},
          {"role": "assistant", "content": "O documento trata de..."}
        ]
      }'
```

Resposta (resumo):
```json
{
  "answer": "...",
  "sources": [
    {"source": "storage/contrato.pdf", "page": 5}
  ],
  "conversation_id": 42
}
```

## Execução local
1. **Ingestão**: `python ingest.py`
2. **API**: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
3. **Interface principal**:
```bash
cd web
streamlit run streamlit_app.py
```
Funcionalidades:
- Upload de PDFs para `storage/`.
- Campo de pergunta + slider `top_k`.
- Exibição da resposta e fontes retornadas pela API.
- Histórico de conversas persistido por usuário.
- Identificação de usuário para isolamento de dados.

### Dashboard de Métricas
```bash
cd web
streamlit run metrics_dashboard.py
```
Visualize métricas e estatísticas de uso:
- Estatísticas gerais (total de consultas, taxa de sucesso, tempo de resposta).
- Séries temporais (consultas por dia, tempo médio de resposta).
- Top usuários e documentos mais consultados.
- Estatísticas de erros.
- Filtros por período e usuário.
- Exportação direta de métricas (CSV/JSON) via link para o endpoint dedicado.

## Docker
### Build manual
```bash
docker build -t rag-chat-pdfs .
```

### Docker Compose (API + Streamlit)
```bash
docker-compose up --build
```
Serviços:
- API em `http://localhost:8000`.
- Streamlit em `http://localhost:8501`.
- Volumes montados: `storage/`, `chroma_db/`, `conversation_history.db`, `metrics.db`.
- Ajuste variáveis no `.env` antes de subir para garantir conexão com OpenAI e tokens de acesso.

## Screenshots (placeholder)
- `docs/screenshot-upload.png`
- `docs/screenshot-chat.png`

## Monitoramento e Métricas
O sistema inclui monitoramento completo de uso e performance:
- **Captura automática**: Todas as consultas são registradas automaticamente com métricas de tempo de resposta, sucesso/erro e uso de documentos.
- **Persistência**: Métricas são armazenadas em SQLite (`metrics.db`) para análise histórica.
- **Dashboard**: Interface Streamlit dedicada para visualização de métricas e estatísticas.
- **Endpoints de métricas**: API REST para consulta programática de estatísticas e exportação (CSV/JSON).

Métricas coletadas:
- Total de consultas e taxa de sucesso.
- Tempo de resposta (médio, mínimo, máximo).
- Top usuários e documentos mais consultados.
- Estatísticas de erros por tipo e endpoint.
- Séries temporais para análise de tendências.

## Testes Automatizados
- A suíte utiliza `pytest`. Execute:
  ```bash
  pytest
  ```
- Os testes cobrem as principais rotinas do módulo de métricas e os endpoints críticos da API (stats/export).
- Utilize um ambiente virtual para isolar dependências antes de rodar os testes.

## CI/CD
- Pipeline automatizado via GitHub Actions (`.github/workflows/ci.yml`).
- Para cada push/pull request:
  1. Checkout do repositório.
  2. Configuração do Python 3.11.
  3. Instalação das dependências (`pip install -r requirements.txt`).
  4. Execução da suíte (`pytest`).
- Expanda facilmente adicionando lint/coverage conforme necessário.

## Roadmap
- Suporte a formatos adicionais (DOCX, Markdown, HTML).
- Alertas e notificações baseados em métricas (por exemplo, erros acima do normal).
- Integração com ferramentas de observabilidade externas (Datadog, Grafana, etc.).
- Automação de ingestão (watcher para executar `ingest.py` ao detectar novos arquivos).
