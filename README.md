# 🇪🇺 EU Regulatory RAG System

A **Retrieval-Augmented Generation (RAG)** platform that automatically ingests EU regulatory texts — DORA and NIS2 — directly from official EU sources, stores them as semantic embeddings in a vector database, and exposes a conversational interface for compliance teams, legal analysts, and IT security officers.

## Overview

| Component | Description |
|---|---|
| **Ingestion** | Fetches and parses DORA & NIS2 from EUR-Lex / official EU portals |
| **Embeddings** | Chunks regulatory text and stores as vector embeddings |
| **Vector DB** | Semantic search over regulatory corpus (Qdrant) |
| **LLM Interface** | Conversational Q&A grounded in retrieved regulatory context |
| **API** | REST + streaming API for integration with compliance tooling |
| **UI** | Web interface for non-technical compliance teams |

## Supported Regulations

- **DORA** — Digital Operational Resilience Act (Regulation (EU) 2022/2554)
- **NIS2** — Network and Information Systems Directive 2 (Directive (EU) 2022/2555)

## Architecture

```
Ingestion Pipeline
EUR-Lex SOAP / REST → Parser → Chunker → Embed
                          ↓
                     Vector DB (Qdrant)
                          ↓
                   RAG Query Layer
   User Query → Embed → Retrieve → Augment Prompt → LLM
                          ↓
                 REST API / Streaming
                          ↓
           Web UI / CLI / Integrations
```

## Project Structure

```
eu-regulatory-rag/
├── config.py                    # Pydantic Settings (loads .env)
├── ingestion/
│   ├── sources/
│   │   ├── regulations.py       # Regulation registry (DORA, NIS2)
│   │   └── eurlex_client.py     # EUR-Lex SOAP + REST client
│   ├── parsers/
│   │   └── eurlex_parser.py     # HTML → structured sections
│   ├── chunker.py               # Semantic chunking + metadata
│   └── pipeline.py              # Full orchestration CLI
├── embeddings/
│   └── service.py               # local (sentence-transformers) or OpenAI
├── vectordb/
│   └── qdrant_client.py         # Qdrant wrapper
├── rag/
│   └── retriever.py             # Semantic search + citation context
├── api/
│   └── main.py                  # FastAPI application
├── tests/
│   ├── test_parser.py
│   ├── test_chunker.py
│   └── test_regulations.py
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)

### 2. Clone & install

```bash
git clone <repo-url>
cd eu-regulatory-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package (with dev extras for testing)
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum, leave defaults for local embedding + Qdrant
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `EURLEX_USERNAME` | _(empty)_ | EUR-Lex SOAP username (optional) |
| `EURLEX_PASSWORD` | _(empty)_ | EUR-Lex SOAP password (optional) |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |
| `OPENAI_API_KEY` | _(empty)_ | Required when `EMBEDDING_PROVIDER=openai` |

### 4. Start Qdrant

```bash
docker compose up qdrant -d
```

## Running the Ingestion Pipeline

```bash
# Ingest all supported regulations (DORA + NIS2)
python -m ingestion.pipeline --regulation all

# Ingest DORA only
python -m ingestion.pipeline --regulation DORA

# Force re-ingest (skip idempotency check)
python -m ingestion.pipeline --regulation DORA --force

# Dry run — parse & chunk but do not write to Qdrant
python -m ingestion.pipeline --regulation NIS2 --dry-run
```

Or use the installed script:

```bash
eurlex-ingest --regulation all
```

> **EUR-Lex credentials:** If `EURLEX_USERNAME` and `EURLEX_PASSWORD` are set, the pipeline
> uses the SOAP API (more structured data). Otherwise it falls back to the public HTML endpoint.

## Using the RAG Retriever

```python
from rag.retriever import RegulatoryRetriever

retriever = RegulatoryRetriever()

# Retrieve top-5 chunks relevant to the query
results = retriever.retrieve(
    "What are the ICT risk management requirements for financial entities?",
    regulation="DORA",
    section_type="article",
    top_k=5,
)

# Format as LLM context with citations
context = retriever.build_context(results)
print(context)
# → [1] According to DORA — Article 6 (ICT risk management framework):
#   Financial entities shall have in place a sound, comprehensive and well-documented …
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Embeddings | sentence-transformers / OpenAI text-embedding-3-small |
| Vector DB | Qdrant |
| LLM | OpenAI GPT-4o / Ollama |
| API | FastAPI |
| Ingestion | httpx, beautifulsoup4, lxml, zeep |
| Token counting | tiktoken |
| Infra | Docker Compose |

## License

MIT
