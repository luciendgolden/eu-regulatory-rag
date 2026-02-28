# 🇪🇺 EU Regulatory RAG System

A **Retrieval-Augmented Generation (RAG)** platform that automatically ingests EU regulatory texts — DORA and NIS2 — directly from official EU sources, stores them as semantic embeddings in a vector database, and exposes a conversational interface for compliance teams, legal analysts, and IT security officers.

## Overview

| Component | Description |
|---|---|
| **Ingestion** | Fetches and parses DORA & NIS2 from EUR-Lex / official EU portals |
| **Embeddings** | Chunks regulatory text and stores as vector embeddings |
| **Vector DB** | Semantic search over regulatory corpus (Qdrant / pgvector) |
| **LLM Interface** | Conversational Q&A grounded in retrieved regulatory context |
| **API** | REST + streaming API for integration with compliance tooling |
| **UI** | Web interface for non-technical compliance teams |

## Supported Regulations

- **DORA** — Digital Operational Resilience Act (Regulation (EU) 2022/2554)
- **NIS2** — Network and Information Systems Directive 2 (Directive (EU) 2022/2555)

## Architecture

```
Ingestion Pipeline
EUR-Lex / Official EU Sources → Parser → Chunker → Embed
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
├── ingestion/          # Scrapers, parsers, chunking logic
│   ├── sources/        # EUR-Lex connectors
│   ├── parsers/        # PDF/HTML → structured text
│   └── chunker.py
├── embeddings/         # Embedding model wrappers
├── vectordb/           # Vector DB client & schema
├── rag/                # Retrieval + prompt assembly
├── api/                # FastAPI application
├── ui/                 # Frontend (Next.js)
├── tests/
├── docs/
├── docker-compose.yml
└── pyproject.toml
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Embeddings | sentence-transformers / OpenAI text-embedding-3-small |
| Vector DB | Qdrant |
| LLM | OpenAI GPT-4o / Ollama |
| API | FastAPI |
| UI | Next.js + Tailwind |
| Ingestion | httpx, pdfplumber, beautifulsoup4 |
| Infra | Docker Compose |

## License

MIT
