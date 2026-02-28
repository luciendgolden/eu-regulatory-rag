# Architecture Decision Records

## ADR-001: Vector Database — Qdrant
Chosen for: self-hostable, strong Python SDK, payload filtering for regulation metadata.

## ADR-002: Chunking Strategy
Regulatory texts have numbered articles/paragraphs. Strategy: chunk at article level,
with 1-article overlap for context continuity. Max ~512 tokens per chunk.

## ADR-003: Embedding Model
Default: `sentence-transformers/all-MiniLM-L6-v2` for self-hosted.
Optional: OpenAI `text-embedding-3-small` for higher accuracy.

## ADR-004: Source of Truth
DORA: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R2554
NIS2: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022L2555
