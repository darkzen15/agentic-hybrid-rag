# ARCHITECTURE.md — System Architecture & Module Guide

This document maps out how the hybrid GraphRAG system is structured, how
the modules connect, and where the key design decisions live. It's meant
for someone new to the codebase who needs to orient themselves, or for
future-you when you come back to this after a break.

---

## System overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Qdrant    │     │    Neo4j    │     │   Ollama    │
│  (vectors)  │     │   (graph)  │     │   (LLMs)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┘───────────────────┘
                   │
            ┌──────┴──────┐
            │  FastAPI    │
            │   (api/)    │
            └──────┬──────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────┴──────┐        ┌───────┴───────┐
│ Upload page │        │  OpenWebUI    │
│ (port 8000) │        │  (port 3000)  │
└─────────────┘        └───────────────┘
```

Five containers, defined in `docker-compose.yml`:

| Container | Role | Data persistence |
|---|---|---|
| **qdrant** | Vector store — chunk embeddings + full-text payload | `qdrant_data` volume |
| **neo4j** | Knowledge graph — entities, relationships, document metadata | `neo4j_data` volume |
| **ollama** | Local LLM host — chat, embedding, and extraction models | `ollama_data` volume |
| **api** | FastAPI application — this is the codebase itself | `api_uploads` volume (uploaded files + job state) |
| **openwebui** | Chat UI — talks to the API's OpenAI-compatible endpoints | `openwebui_data` volume |

The API is the only service with application logic. Qdrant, Neo4j, and
Ollama are stock images used as-is; OpenWebUI is stock plus three env vars
pointing its internal embedding subsystem at Ollama instead of HuggingFace.

---

## Module map

The 17 Python modules in `api/` fall into four layers. Dependencies flow
downward — no module imports from a layer above it.

```
┌─────────────────────── ENTRY POINT ───────────────────────┐
│                        main.py                            │
│  FastAPI app, all HTTP endpoints, request/response models │
└────────────────┬──────────────┬────────────────────────────┘
                 │              │
    ┌────────────┴──┐     ┌────┴─────────────────┐
    │  INGESTION    │     │     RETRIEVAL         │
    │               │     │                       │
    │  ingest.py    │     │  retrieval.py         │
    │  jobs.py      │     │  agentic_retrieval.py │
    └───┬───────────┘     └───┬───────────────────┘
        │                     │
┌───────┴─────────────────────┴───────────────────────────┐
│                    EXTRACTION / PARSING                   │
│                                                          │
│  file_parsers.py          graph_extraction.py            │
│  structural_extraction.py                                │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────────┐
│                     INFRASTRUCTURE                       │
│                                                          │
│  config.py        ollama_client.py    retry.py           │
│  embeddings.py    vector_store.py     graph_store.py     │
│  llm.py                                                 │
└──────────────────────────────────────────────────────────┘
```

---

## Module-by-module reference

### Infrastructure layer

**`config.py`** (237 lines, 57 settings)
All configuration, read from environment variables with defaults. Every
setting that exists anywhere in the system is defined here — there are no
scattered `os.getenv` calls in other modules. The `docker-compose.yml`
`environment:` block and `_env.example` are kept in strict sync with this
file (verified programmatically during development).

**`ollama_client.py`** (59 lines)
Shared, reused `httpx.Client` (sync) and `httpx.AsyncClient` (async) for
all Ollama communication. Every Ollama call in the codebase — embedding,
extraction, chat, retrieval LLM-assist — goes through these two pooled
clients rather than opening a fresh TCP connection per request. Shutdown
cleanup is wired into `main.py`'s FastAPI shutdown event.

**`retry.py`** (87 lines)
Generic retry-with-backoff wrappers (`with_retry` sync, `with_retry_async`)
used by every Ollama call. Retries transient failures (connection errors,
timeouts, 5xx) but never retries 4xx (e.g. unknown model → would just
fail the same way every time). Backoff is exponential with jitter.

**`embeddings.py`** (65 lines)
Wraps Ollama's `/api/embed` endpoint. Supports batch embedding (multiple
texts in one call) and optional prefix strings for asymmetric embedding
models (query vs. document prefixes). Used by both ingestion (embed chunks)
and retrieval (embed the user's query).

**`vector_store.py`** (316 lines)
Qdrant client. Handles collection creation (with in-process caching so it
only checks once per server lifetime, not once per document), chunk
upsert, semantic search, keyword/full-text search, neighbor-window
retrieval (chunks adjacent to a given index in the same document), and
document deletion. The full-text index for keyword search is created
alongside the collection.

**`graph_store.py`** (603 lines — the largest infrastructure module)
Neo4j client. The most complex module because it handles several things
that interact:

- **Schema setup:** constraints + full-text index for fuzzy entity-name
  matching (also cached in-process).
- **Entity upsert with type management:** entities merge on a normalized
  key (lowercased, whitespace-collapsed). The `type` property only ever
  upgrades (CONCEPT → specific type, never the reverse). Each entity also
  gets a second Neo4j *label* matching its type (`:Entity:PERSON`,
  `:Entity:THREAT_ACTOR`) via `apoc.create.setLabels`, so Neo4j Browser
  colors them by type — a property alone wouldn't do this.
- **Relationship upsert with direction-dedup:** checks BOTH directions
  before creating a new edge, so the same fact extracted with
  source/target reversed across different chunks reinforces one edge
  instead of fragmenting into two. Tracks `weight` (how many times seen)
  and `chunk_ids` (provenance) per edge.
- **Batched writes:** `upsert_batch` processes a whole embedding batch
  in one Neo4j transaction rather than one session per chunk.
- **Graph expansion:** `expand_from_chunks` does 1-hop traversal from
  seed entities, ranked by relationship weight, with configurable type
  exclusions (DATE entities excluded by default to prevent hub-entity
  pollution).
- **Fuzzy entity matching:** `find_entities_in_text` uses a Lucene
  full-text index with scaled fuzzy tolerance (`~1` for short words like
  years, `~2` for longer words) — catches typos and near-misses.
- **Relationship-path lookup:** `find_relationship_paths` finds the
  shortest path between two named entities via `shortestPath`.
- **Document deletion:** removes a document's entities/relationships
  from the graph, keeping entities shared with other documents.

**`llm.py`** (162 lines)
Ollama chat client — both single-shot (`ollama_chat`) and streaming
(`ollama_chat_stream`). Builds the RAG system prompt with retrieved
context injected, handles conversation history, and maps generation
parameters (temperature, top_p, max_tokens) to Ollama's option format.
Also provides `list_ollama_models` for the health check and model-listing
endpoints (with a short per-request timeout override so a health check
doesn't hang for minutes if Ollama is down).

### Extraction / parsing layer

**`file_parsers.py`** (530 lines)
Converts uploaded files into a list of text blocks with metadata. Each
format has its own extractor:

| Format | Method | Notes |
|---|---|---|
| PDF | `pdfplumber` per page, OCR fallback via `pytesseract`/`pdf2image` | Scanned pages only get OCR when native text extraction finds nothing |
| DOCX | `python-docx` | Splits on headings into sections |
| DOC | LibreOffice headless conversion to text | Legacy binary format, no pure-Python parser exists |
| CSV/TSV | `csv` module, batched ~20 rows per block | Formatted as `header: value` pairs, not raw rows |
| XLSX | `openpyxl`, batched ~20 rows per block per sheet | Sheet name + row range in metadata |
| HTML | `BeautifulSoup` | Tables become their own blocks; headings are section boundaries |
| XML | Recursive flattening to `path: value` lines | Each child of root = one block |
| EML | `email` module | Headers and body as separate blocks |
| JSON | `json` / `ijson` (streaming for large files) | Top-level array = one block per record |
| TXT/LOG/MD | Raw read with encoding detection | |

Every format returns **blocks** (not one big string), so the chunker in
`ingest.py` never silently spans two unrelated pages/sections/records.

**`graph_extraction.py`** (220 lines)
LLM-based entity/relationship extraction. Sends each chunk's text to
Ollama with a detailed system prompt specifying the domain entity taxonomy
(16 types: PERSON, ORGANIZATION, THREAT_ACTOR, MALWARE, VULNERABILITY,
INDICATOR, PRODUCT, LOCATION, FACILITY, MILITARY_UNIT, WEAPON, OPERATION,
LAW, EVENT, DATE, CONCEPT) and relationship rules. Validates the LLM's
output: checks entity types against the allowed list (invalid → CONCEPT),
logs diagnostic counts per chunk (missing type / invalid type / model
chose CONCEPT) so type-quality problems surface in logs rather than
silently accumulating.

**`structural_extraction.py`** (302 lines)
Rule-based entity/relationship extraction for JSON files, skipping the
LLM entirely. Walks the parsed JSON tree and derives entities from key
names (via a hint table mapping field names to entity types) and
relationships from nesting/sibling structure. Special-cases: numeric year
values (JSON ints) for DATE entities; bare `"name"` fields biased toward
EVENT when a date sibling is present (incident-log pattern). Returns a
`coverage` ratio (fraction of scalar fields that matched any hint) so
`ingest.py` can fall back to LLM extraction per-record when coverage is
too low — the self-correcting safety net for unrecognized schemas.

### Ingestion layer

**`ingest.py`** (325 lines)
Orchestrates the full ingestion pipeline for a document:

```
file_parsers.extract_text(file)
    → list of text blocks with metadata
        → chunk each block (character-window splitter, paragraph/sentence-aware)
            → batch processing (EMBED_BATCH_SIZE chunks per batch):
                1. Embed the batch (one Ollama call for all chunks)
                2. Extract entities/relationships per chunk:
                   - JSON with structural_extraction result + sufficient
                     coverage → use it directly (no LLM call)
                   - Otherwise → LLM extraction, concurrent across the
                     batch (GRAPH_EXTRACTION_CONCURRENCY threads)
                3. Write graph data to Neo4j (one transaction per batch,
                   fail-soft — a Neo4j hiccup doesn't kill the whole job)
                4. Write vectors + payloads to Qdrant
                5. Update job progress
```

Content-hash deduplication: computes a hash of the full text before
chunking and warns (+ populates `duplicate_of` in the job record) if the
same content was already ingested under a different doc_id.

**`jobs.py`** (150 lines)
In-memory + JSON-file-backed job tracking for background ingestion. Job
records survive container restarts (persisted to the `api_uploads` volume).
Progress-only updates are throttled (every 5th batch) to avoid rewriting
the entire job file hundreds of times during a large ingest. Supports
cooperative cancellation: `request_cancel` sets a flag that `ingest.py`'s
batch loop checks between batches. History is capped at 500 finished jobs.

### Retrieval layer

**`retrieval.py`** (526 lines)
The core retrieval engine. `hybrid_retrieve` runs four independent search
channels and fuses their results:

```
Query
  │
  ├─── [concurrent, ThreadPoolExecutor] ──────────────────────┐
  │                                                           │
  │  VECTOR: embed query → Qdrant semantic search             │
  │    (+ optional query expansion: LLM generates alternate   │
  │     phrasings, each embedded and searched separately)     │
  │                                                           │
  │  ENTITY_NAME: fuzzy full-text lookup in Neo4j             │
  │    (catches "tell me about <specific name>" queries)      │
  │                                                           │
  │  KEYWORD: Qdrant full-text filter on chunk text           │
  │    (catches exact strings — CVE IDs, acronyms)            │
  │                                                           │
  ├───────────────────────────────────────────────────────────┘
  │
  │  GRAPH: 1-hop expansion in Neo4j, seeded from vector +
  │    entity_name results (must wait for those two channels)
  │
  ▼
 RRF Fusion (Reciprocal Rank Fusion)
  │  Per-channel weights: vector=1.0, entity=1.3, graph=0.8, keyword=0.6
  │
  ▼
 Post-processing
  │  Per-document cap (RAG_MAX_CHUNKS_PER_DOC)
  │  Near-duplicate suppression (word-overlap ratio)
  │  Neighbor-window expansion (chunks before/after strong hits)
  │  Relationship-path lookup (if query names 2+ known entities)
  │
  ▼
 Optional LLM-assist steps (off by default, each adds latency):
    - RAG_RERANK_ENABLED: LLM re-scores fused candidates
    - RAG_CONDENSE_ENABLED: rewrites follow-up questions using history
    - RAG_QUERY_EXPANSION_ENABLED: generates alternate query phrasings
```

Also provides `_ollama_chat_json` (shared LLM-call helper used by the
agentic module) and `format_as_context` (flattens results into a single
context string for the generation prompt).

**`agentic_retrieval.py`** (353 lines)
Multi-step reasoning loop on top of `hybrid_retrieve`, toggled by
`RAG_AGENTIC_ENABLED`. Designed for strong models (120B+):

```
Query
  │
  ▼
 DECOMPOSE: LLM breaks query into focused sub-questions
  │
  ▼ (for each sub-question)
 RETRIEVE: hybrid_retrieve (all four channels)
  │
  ▼
 GRADE: LLM scores each chunk's relevance (0-10),
        discards below RAG_AGENTIC_RELEVANCE_THRESHOLD
  │
  ├─── enough relevant chunks survived? → done for this sub-question
  │
  ▼ (not enough)
 REFORMULATE: LLM broadens/rephrases the sub-question
  │
  └─── retry RETRIEVE + GRADE (up to RAG_AGENTIC_MAX_RETRIES)
  │
  ▼
 MERGE: deduplicate chunks across all sub-questions,
        sort by relevance score
```

Every failure mode falls back gracefully: decomposition failure → original
query as single sub-question; grading failure → keep all chunks; 
reformulation failure → stop retrying with whatever survived.

### Entry point

**`main.py`** (777 lines)
FastAPI application. All HTTP endpoints, request/response Pydantic
models, and the OpenAI-compatible `/v1/chat/completions` streaming
interface (what OpenWebUI connects to). Routes chat endpoints through
`_smart_retrieve` which picks agentic or single-shot based on the toggle.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Basic liveness check |
| GET | `/health/detailed` | Checks Qdrant, Neo4j, Ollama connectivity + model tag verification |
| GET | `/` | Upload webpage (`static/index.html`) |
| POST | `/ingest` | Ingest raw text (synchronous) |
| POST | `/ingest/file` | Upload and ingest a file (async background job) |
| GET | `/jobs/{job_id}` | Poll ingestion job status |
| GET | `/jobs` | List recent jobs |
| POST | `/jobs/{job_id}/cancel` | Cancel a running ingestion job |
| GET | `/documents` | List ingested documents |
| DELETE | `/documents/{doc_id}` | Delete a document and all its data |
| POST | `/retrieve` | Retrieve context (always single-shot, for debugging) |
| GET | `/retrieve` | Same, via query params |
| POST | `/rag/chat` | Retrieve + generate answer |
| GET | `/v1/models` | OpenAI-compatible model listing |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (what OpenWebUI uses) |

---

## Data flow: ingestion

```
User uploads file
  → main.py streams to disk, creates job record
  → ingest.py runs in background:
      file_parsers.extract_text(path)
        → blocks (page/section/record/batch of rows)
      chunk each block (character window, ~1000 chars, ~200 overlap)
      for each batch of EMBED_BATCH_SIZE chunks:
        embeddings.embed_texts(batch)
          → vectors (via ollama_client → Ollama /api/embed)
        for each chunk, one of:
          structural_extraction (JSON with sufficient coverage)
            → entities + relationships (no LLM call)
          graph_extraction.extract_graph(chunk_text)
            → entities + relationships (via ollama_client → Ollama /api/chat)
        graph_store.upsert_batch(entities, relationships)
          → Neo4j (Entity nodes, typed relationships, Document node)
        vector_store.upsert_chunks(vectors, payloads)
          → Qdrant (vectors + text + metadata as payload)
      jobs.update_job(status="done")
```

**Where text lives:** Qdrant only. Neo4j stores entity names, types,
relationship types/weights, and lists of chunk_ids — never chunk text.
This was a deliberate design decision (user's first request).

**Where entities connect to chunks:** Entity nodes carry a `chunk_ids`
list property and a `doc_ids` list property. `expand_from_chunks` uses
`chunk_ids` to find entities mentioned in the seed chunks, then walks
one hop to find other entities' `chunk_ids`. Retrieval fetches actual
text for those chunk_ids from Qdrant via `vector_store.get_by_ids`.

---

## Data flow: retrieval + generation

```
User asks a question (via /rag/chat, /v1/chat/completions, or OpenWebUI)
  → main.py extracts query + history
  → _smart_retrieve picks agentic or single-shot:
      agentic: decompose → per-sub-question retrieve+grade+retry → merge
      single-shot: hybrid_retrieve directly
  → hybrid_retrieve runs four channels (three concurrent, then graph):
      vector: embeddings.embed_text(query) → vector_store.search
      entity_name: graph_store.find_entities_in_text(query)
      keyword: vector_store.keyword_search(extracted_keywords)
      graph: graph_store.expand_from_chunks(seed_chunk_ids)
  → RRF fusion across all channels
  → post-processing (dedup, per-doc cap, neighbor expansion, paths)
  → format_as_context(results) → context string
  → llm.build_rag_messages(query, context, history) → prompt
  → llm.ollama_chat or ollama_chat_stream → answer
  → response (answer + sources + entities + timings)
```

---

## Key design decisions and why

**Entities-only graph (no chunk nodes in Neo4j).**
User's explicit first request. Qdrant is the single source of truth for
chunk text. Neo4j holds entities, relationships, and Document nodes.
Entity→chunk mapping is via a `chunk_ids` list property on the Entity
node, not a separate Chunk node with its own edges.

**Specific verbs as Neo4j relationship types (not broad categories).**
We tried a category system (AFFILIATION, CREATION, etc.) as the Neo4j
type with specific verbs as properties. It produced massive OTHER
overuse with smaller models. Reverted to the specific verb (FOUNDED,
WORKS_AT, etc.) directly as the relationship type. Accepted tradeoff:
synonym fragmentation (FOUNDED vs ESTABLISHED = two edges, not one).

**Four-channel RRF fusion rather than single vector search + graph expand.**
Each channel independently retrieves candidates; they're fused by
Reciprocal Rank Fusion so a chunk found by multiple channels outranks
one found by only one. Channels fail independently (fail-soft for
entity/graph/keyword; fail-hard for vector since that's the minimum
viable retrieval).

**Structural extraction for JSON (with coverage-based LLM fallback).**
JSON's own key names and nesting encode relationships directly — no
LLM inference needed. But key-name heuristics can't generalize to
schemas they haven't seen. The `coverage` ratio (fraction of fields
matching any hint) auto-detects unrecognized schemas and falls back to
LLM extraction per-record rather than silently producing thin results.

**Domain-specific entity taxonomy (16 types).**
Designed for cybersecurity/threat-intelligence/military/political content.
Includes THREAT_ACTOR (distinct from ORGANIZATION), MALWARE, VULNERABILITY,
INDICATOR, MILITARY_UNIT, WEAPON, OPERATION. DATE entities are rounded to
the year (low-cardinality by design, for clean merging) and excluded from
graph expansion traversal (to prevent hub-entity pollution).

**Agentic retrieval as an opt-in layer, not a replacement.**
The agentic loop (decompose → grade → self-correct) adds real value on
complex queries with a strong model but adds latency and lower-quality
decisions with a weak one. It sits entirely on top of hybrid_retrieve
(which is the engine under each sub-question) and is toggled independently.
The `/retrieve` endpoint always stays single-shot for debugging.

---

## Standalone tools (project root)

| File | Purpose |
|---|---|
| `eval_retrieval.py` | Hit-rate/MRR evaluation against a labeled query set |
| `dedupe_entities.py` | Find likely-duplicate entities (string similarity), merge confirmed pairs |

---

## Configuration reference

All 57 settings live in `config.py`, documented in `_env.example`, and
forwarded by `docker-compose.yml`. They're organized into groups:

- **Models:** `OLLAMA_MODEL`, `EMBEDDING_MODEL`, `GRAPH_EXTRACTION_MODEL`, `RAG_ASSIST_MODEL`
- **Ollama tuning:** timeout, retry attempts/delay, batch size, extraction concurrency
- **File parsing:** conversion timeouts, streaming thresholds, OCR toggle
- **JSON structural extraction:** enable toggle, minimum coverage for LLM fallback
- **Retrieval fusion:** RRF K, per-channel weights, candidate multiplier, per-doc cap, dedup threshold
- **Graph expansion:** neighbor window, type exclusions, path max hops
- **Optional LLM-assist:** rerank, condense, query expansion (each a separate toggle)
- **Agentic retrieval:** enable toggle, max sub-questions, max retries, relevance threshold
- **Chat:** show-sources toggle

The consistency between config.py / docker-compose.yml / _env.example
has been verified programmatically — every `os.getenv` in config.py
has a corresponding entry in the compose file's environment block.

---

## Common operations

**Rebuild after code changes:**
```bash
docker compose build api
docker compose up -d --force-recreate api
```

**Wipe graph and re-ingest** (needed after extraction/schema changes):
```bash
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n) DETACH DELETE n"
```

**Check what's in the graph:**
```cypher
MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count(*) DESC
```

**Diagnose startup/connectivity issues:**
```bash
curl http://localhost:8000/health/detailed | python3 -m json.tool
```

**Check extraction quality:**
```bash
docker compose logs api | grep "graph_extraction:"
```
