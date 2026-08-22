# Hybrid GraphRAG (Qdrant + Neo4j + Ollama + OpenWebUI + FastAPI)

A minimal hybrid Retrieval-Augmented Generation stack:

- **Qdrant** — vector store for chunk embeddings (semantic recall)
- **Neo4j** — knowledge graph of `Entity` nodes only (linked to `Document`),
  used to expand vector search results with related chunks; no chunk text
  is duplicated into Neo4j — entities carry a `chunk_ids` list that points
  back to Qdrant, which stays the single source of truth for chunk text
- **Ollama** — runs your local LLMs (generation) and embedding model
  (`nomic-embed-text` by default)
- **OpenWebUI** — chat interface, pre-wired to the RAG API so every model in
  its dropdown is retrieval-augmented out of the box
- **FastAPI** — ties it together: `/ingest`, `/retrieve`, `/rag/chat`, plus an
  OpenAI-compatible `/v1/chat/completions` (what OpenWebUI and tools like
  promptfoo talk to)

## How retrieval works

Retrieval runs four independent channels and fuses them with **Reciprocal
Rank Fusion (RRF)**, rather than a single vector-search-then-expand
pipeline — each channel gets a real chance to contribute regardless of how
the others perform:

- **vector** — Qdrant semantic search over chunk embeddings (optionally
  widened by LLM-generated query reformulations, see below)
- **entity_name** — literal entity-name matches from the query, looked up
  directly in Neo4j via an indexed n-gram lookup (catches "tell me about
  &lt;specific name&gt;" queries that vector search alone can miss)
- **graph** — 1-hop relationship expansion in Neo4j, seeded from both the
  vector and entity_name channels, weighted by how well-established each
  connecting relationship is
- **keyword** — Qdrant full-text filter on chunk text (catches exact
  strings — IDs, codes, acronyms — that embeddings blur)

A chunk found by multiple channels outranks one found strongly by only
one. Results then go through a per-document cap (so one document can't
crowd out everything else), near-duplicate suppression (adjacent
overlapping chunks don't both take a slot), and — for the top few hits —
**neighbor-window expansion**, pulling in the chunk immediately
before/after each strong hit so context a chunk boundary cut off isn't
lost.

Separately, if the query names two or more known entities, a direct
relationship-path lookup between them (`A -[FOUNDED]-> B -[LOCATED_IN]-> C`)
is added to the context — answering from graph structure directly rather
than hoping some chunk happens to state the connection in prose.

Entities and relationships are extracted per chunk by your local LLM (see
`graph_extraction.py`), not a fixed NER model — it returns typed entities
(PERSON, ORGANIZATION, ...) and typed relationships (`WORKS_AT`, `FOUNDED`,
etc.), which become real Neo4j relationship types via the APOC plugin. This
catches relevant chunks that vector search alone would miss (e.g. two chunks
about the same person/org that use different wording), and gives graph
edges actual semantic meaning rather than a generic "these two things
appeared near each other" link. The tradeoff is ingestion speed: extraction
is one extra LLM call per chunk on top of embedding, run concurrently in
batches (`GRAPH_EXTRACTION_CONCURRENCY`) to keep large-file ingestion
reasonable. Set `GRAPH_EXTRACTION_MODEL` to a smaller/faster model than
your chat model if that matters for your hardware.

### Optional LLM-assist steps (off by default — each adds latency)

Three further retrieval improvements exist but are opt-in, since each
costs at least one extra LLM call per query:

- **`RAG_RERANK_ENABLED`** — after RRF fusion, asks the LLM to score each
  candidate's relevance directly and re-sorts by that score before
  truncating to `top_k`. Usually improves precision.
- **`RAG_CONDENSE_ENABLED`** — for multi-turn chat, rewrites a follow-up
  like "what about its founder?" into a standalone question using recent
  history before retrieval. Without this, a follow-up's embedding alone
  often carries little meaning.
- **`RAG_QUERY_EXPANSION_ENABLED`** — generates alternate phrasings of the
  query and searches with each too, improving recall on queries whose
  specific wording happens to embed weakly.

See `_env.example` for the full list of tunable weights, thresholds, and
these toggles.

## Run it

```bash
cp .env.example .env    # optionally change passwords / models
docker compose up --build
```

This starts:
- Qdrant on `localhost:6333`
- Neo4j Browser on `localhost:7474` (bolt on `7687`)
- Ollama on `localhost:11434`
- API + upload webpage on `localhost:8000` (docs at `localhost:8000/docs`)
- **OpenWebUI on `localhost:3000`**

The first build takes a few minutes mainly for the `libreoffice-writer`
system package (needed for legacy `.doc` conversion) — no Python-side model
download happens at build time; embedding and extraction models are pulled
into Ollama separately (see below).

**Pull models into Ollama** (not included automatically — they're large):

```bash
docker compose exec ollama ollama pull llama3.1:8b      # or any chat model you like
docker compose exec ollama ollama pull nomic-embed-text # embedding model — required
```

`OLLAMA_MODEL` / `EMBEDDING_MODEL` / `GRAPH_EXTRACTION_MODEL` in `.env` must
match the **exact tag** `ollama list` shows (e.g. `llama3.1:8b`, not just
`llama3.1`) — Ollama treats an untagged name as `:latest`, a different tag
than `:8b` unless you specifically pulled `:latest`. A mismatch here doesn't
fail loudly at startup; it shows up later as an unexplained `404 Not Found`
from Ollama the first time that model is actually called. Run
`docker compose exec api env | grep OLLAMA_MODEL` and compare against
`docker compose exec ollama ollama list` if you ever see that.

If you use a different chat model name, set `OLLAMA_MODEL` in `.env` to
match. If you use a different embedding model, set `EMBEDDING_MODEL` —
just be aware that changing it after you've already ingested documents means
old vectors (dimension/space of the old model) won't be comparable to new
ones; re-ingest if you switch. `GRAPH_EXTRACTION_MODEL` is optional and
falls back to `OLLAMA_MODEL` if unset — set it separately if you want a
smaller/faster model doing entity extraction than the one doing chat.

Open **`http://localhost:3000`**, create the first (admin) account, and
start chatting — the model dropdown shows whatever you've pulled into
Ollama, and every message is retrieval-augmented automatically.

## Ingest a document

### Via the upload webpage

Open **`http://localhost:8000`** in a browser. Drag & drop (or click to
choose) one or more files — PDF (including scanned/image-only pages, via
OCR), DOCX, legacy DOC, CSV, TSV, XLSX, HTML, XML, JSON, EML, TXT, LOG, or
MD. Each upload streams straight to disk (so large files don't sit in
memory), kicks off a background ingestion job, and the page polls for
status until it shows `done` with the chunk count, or `error` with what
went wrong.

Not currently supported: PPTX and archive files (`.zip`) containing
multiple documents — see [Known limitations](#known-limitations).

### Via the API

Raw text:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "acme-handbook",
    "title": "Acme Employee Handbook",
    "text": "Acme Corp was founded by Jane Smith in Melbourne in 2015. ..."
  }'
```

A file (any supported extension — run `GET /` or check `SUPPORTED_EXTENSIONS`
in `file_parsers.py` for the current list) — returns immediately with a
`job_id` while ingestion runs in the background:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf" \
  -F "doc_id=report-1"

# {"job_id": "...", "status": "queued", ...}

curl http://localhost:8000/jobs/<job_id>
# {"status": "processing" | "done" | "error", "chunks_ingested": 42, ...}
```

**How each format is turned into text:**
- **PDF** — `pdfplumber` extracts text per page (page breaks become chunk
  boundaries, and each chunk's payload carries a `page` number). Pages with
  no extractable text (scanned/image-only) fall back to OCR
  (`pytesseract` + `pdf2image`, via the `tesseract-ocr`/`poppler-utils`
  system packages) when `OCR_ENABLED=true` (the default).
- **DOCX** — `python-docx` pulls paragraph text (split into blocks by
  heading, carried through as a `section` tag) and table cell text.
- **DOC** (legacy binary format) — converted to text via headless
  LibreOffice, since no pure-Python library reliably parses old `.doc`
  files. This is why the API image includes `libreoffice-writer` (adds a
  few hundred MB to the image and a few minutes to the first build). The
  conversion timeout scales with file size (`DOC_CONVERT_TIMEOUT_*`).
- **CSV / TSV / XLSX** — rows are batched into blocks of ~20 rows,
  formatted as `header: value` pairs (like the JSON flattening below)
  rather than chunked by raw character count, which would slice a row in
  half at an arbitrary point. XLSX blocks carry the sheet name and row
  range as metadata.
- **HTML** — parsed with BeautifulSoup; tables become their own blocks
  (row cells joined with ` | `), headings (`h1`-`h6`) become section
  boundaries for the surrounding paragraph text, same as DOCX.
- **XML** — flattened into `path: value` lines the same way as JSON; each
  direct child of the root becomes one block/record.
- **EML** (email) — headers (From/To/Subject/Date) and body text
  (`text/plain`, falling back to stripped `text/html`) become separate
  blocks.
- **JSON** — flattened into `path: value` lines; a top-level array of
  objects is treated as one block per record so records stay separate
  chunks rather than blurring together. Files above
  `JSON_STREAMING_THRESHOLD_MB` (default 20MB) are parsed with a
  streaming parser instead of loading the whole file into memory.
- **TXT / LOG / MD** — read as-is (tries UTF-8, then UTF-16, then Latin-1).

Every format above returns a list of **blocks** (a page, a section, a
batch of rows, a record) rather than one giant string — the chunker splits
each block independently, so a chunk never silently spans two unrelated
pages/sections, and page/section/sheet metadata rides along into the
chunk payload.

**Large files:** uploads are streamed to disk in 1 MB chunks rather than
loaded fully into memory, so multi-hundred-MB PDFs are fine. Default cap is
300 MB (`MAX_UPLOAD_MB` env var) — raise it in `docker-compose.yml` if you
need bigger. The heavy work (embedding, entity extraction) happens in a
FastAPI `BackgroundTask` after the HTTP response returns, so the upload
doesn't hang the request — that's what the job-polling UI/endpoints are for.

Embedding and writing happen in batches of `EMBED_BATCH_SIZE` chunks
(default 32) rather than all at once — a 50MB file can easily chunk into
tens of thousands of pieces, and embedding all of them in a single request
would take far longer than any reasonable timeout. The upload page shows
live progress (`1,240 / 8,600 chunks`) for exactly this reason — large
files are expected to take a while, this is how you tell "still working"
from "stuck." Within each batch, entity/relationship extraction for every
chunk runs concurrently (`GRAPH_EXTRACTION_CONCURRENCY`, default 3) rather
than one Ollama round-trip at a time, since that's the dominant per-chunk
cost — pair this with raising `OLLAMA_NUM_PARALLEL` on the `ollama`
service or requests just queue there anyway. Per-batch timing (embed /
extract / upsert) is logged at INFO level, so if ingestion feels slow you
can see which stage is actually the bottleneck instead of guessing.

If you re-ingest the same content under a different `doc_id` (or a renamed
file), ingestion still proceeds but logs a warning and reports
`duplicate_of` in the job/response — it detects identical content via a
hash, without silently blocking what you asked it to do.

## Managing ingested documents

```bash
# List everything currently ingested
curl http://localhost:8000/documents

# Delete a document and all its data - removes its chunks from Qdrant,
# then cleans up Neo4j: entities that only ever appeared in this document
# are deleted, relationships whose evidence was solely this document are
# deleted, entities/relationships shared with other documents are kept
curl -X DELETE http://localhost:8000/documents/report-1
```

## Query it

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Who founded Acme Corp?", "top_k": 5, "graph_expand": 5}'
```

Response includes a ready-to-use `context` string plus structured `results`
(each `source` tagged with the contributing channel(s) — `vector`,
`entity_name`, `graph`, `keyword`, `neighbor`, `path`, or a `+`-joined
combination when more than one channel found the same chunk) and
`related_entities`. A `timings` field breaks down how long each retrieval
channel took, for diagnosing slow queries.

## Ask a question (retrieval + generation via Ollama)

```bash
curl -X POST http://localhost:8000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Who founded Acme Corp?"}'
```

```json
{
  "answer": "According to the Acme Employee Handbook, Acme Corp was founded by Jane Smith...",
  "context": "...",
  "sources": [...],
  "related_entities": [...],
  "timings": {"vector": 0.08, "entity_name": 0.01, "graph": 0.02, "keyword": 0.03, "total": 0.15}
}
```

Pass `"model": "llama3.2"` (or whichever you've pulled) to override the
default `OLLAMA_MODEL`, and `"temperature"`, `"top_p"`, or `"max_tokens"` to
tune generation per request. Pass `"history": [{"role": "user", "content":
"..."}, ...]` for multi-turn context — used both as generation context and,
if `RAG_CONDENSE_ENABLED=true`, to rewrite follow-up questions into
standalone queries before retrieval.

## Confirming the LLM is actually using your data

A model can *sound* grounded while just answering from its own training —
here's how to check, from easiest to most rigorous:

**1. Look at the sources footer (easiest, works from any chat UI).**
Every `/v1/chat/completions` response — including in OpenWebUI — now ends
with a `---\n**Sources retrieved:**` block listing which documents were
actually pulled into context for that answer, tagged 🔎 vector / 🕸️ graph.
If it says *"No matching context was found"*, the model answered from its
own knowledge, not your data — worth being skeptical of that answer.
Turn it off with `"include_sources": false` per request, or globally via
`SHOW_SOURCES_IN_CHAT=false`.

**2. Check the raw retrieval, separately from generation.**
```bash
curl "http://localhost:8000/retrieve?query=Who+founded+Acme+Corp"
```
This shows exactly which chunks matched and their scores, with no LLM in
the loop — useful for isolating whether a bad answer is a *retrieval*
problem (wrong/no chunks found) or a *generation* problem (right chunks,
model ignored them anyway).

**3. Compare `/rag/chat`'s `context` field against the `answer`.**
`/rag/chat` (unlike the OpenWebUI-facing endpoint) returns the full
`context` string alongside the `answer` in one JSON response — read
through it and check the answer's claims actually appear in there.

**4. The real test: ask something the base model can't already know.**
Pick a fact that only exists in something you've ingested — an invented
product name, an internal policy number, a made-up person's role — and ask
about it. If the model gets it right, it had to have come from retrieval,
since it can't be in its training data. If it hallucinates something
plausible-sounding instead, either retrieval isn't finding the chunk (check
step 2) or the model is ignoring the context (try a more insistent system
prompt via `RAG_SYSTEM_PROMPT`, or a smaller/more instruction-following
model).

**5. Ask something *not* in your data and confirm it admits that.**
The default system prompt asks the model to say when context doesn't
answer the question. If it confidently answers anyway from general
knowledge, that's a sign the prompt isn't being followed strictly enough —
worth tightening `RAG_SYSTEM_PROMPT` if this matters for your use case.

## OpenWebUI (bundled)

OpenWebUI is included in `docker-compose.yml`, already pointed at the RAG
API (`OPENAI_API_BASE_URLS=http://api:8000/v1`) with its direct Ollama
connection disabled (`ENABLE_OLLAMA_API=false`). That means:

- Every model shown in OpenWebUI's dropdown is proxied from your Ollama
  instance via the API's `/v1/models`.
- Every chat turn is automatically retrieval-augmented — the latest message
  triggers hybrid vector+graph search, the result is injected as context,
  then sent to whichever model you picked. No manual setup required beyond
  pulling models into Ollama.
- You can change **temperature** (and top-p, max tokens, seed) per chat from
  OpenWebUI's model settings (the "Advanced Params" panel) — these are
  forwarded straight through to Ollama on every request.

If you'd rather talk to a model directly with no retrieval, add a second
connection in OpenWebUI pointed at `http://ollama:11434` and re-enable
`ENABLE_OLLAMA_API=true`, or just call `POST /v1/chat/completions` on the
API directly with retrieval disabled (`"graph_expand": 0` doesn't fully
disable vector search, so for a true bypass query Ollama's own OpenAI-
compatible endpoint at `http://ollama:11434/v1` instead).

## Testing with promptfoo

Since `/v1/chat/completions` is a standard OpenAI-compatible endpoint with
dynamic `model` and `temperature`, you can point promptfoo straight at it
and sweep both per test case:

```yaml
# promptfooconfig.yaml
providers:
  - id: openai:chat:llama3.1
    config:
      apiBaseUrl: http://localhost:8000/v1
      apiKey: not-checked
      temperature: 0.2
  - id: openai:chat:llama3.2
    config:
      apiBaseUrl: http://localhost:8000/v1
      apiKey: not-checked
      temperature: 0.7

prompts:
  - "{{query}}"

tests:
  - vars:
      query: "Who founded Acme Corp?"
    assert:
      - type: contains
        value: "Jane Smith"
```

Each test call goes through the full hybrid RAG pipeline before hitting the
named Ollama model, so you're evaluating retrieval + generation together,
across models and temperatures, in one sweep. You can also override
retrieval per test case with the non-standard `top_k` / `graph_expand` /
`doc_id` fields this endpoint accepts, if your promptfoo provider config
supports passing extra body fields.

Responses include a "Sources retrieved" footer by default (see [Confirming
the LLM is actually using your data](#confirming-the-llm-is-actually-using-your-data)),
which will break exact-match or strict-format assertions — pass
`"include_sources": false` in the provider config's body, or set
`SHOW_SOURCES_IN_CHAT=false` in `.env`, to get a clean answer-only response
for automated testing.

## Connecting an external OpenWebUI or other client

If you're running OpenWebUI (or anything else) outside this compose file
instead of using the bundled one:

### As a chat model (retrieval + generation, in one)

1. **Settings → Connections → Add Connection** (OpenAI API type).
2. **API Base URL**: `http://<host>:8000/v1`
3. **API Key**: anything non-empty (not checked, but required).
4. Pick a model from the dropdown (proxied from Ollama) and chat — every
   turn is retrieval-augmented, streaming included.

### As a Tool Server (model decides when to retrieve)

1. **Settings → Tools → Add Tool Server**.
2. OpenAPI URL: `http://<host>:8000/openapi.json`
3. Enable the discovered `retrieve_context` tool for whichever model you're
   chatting with — the model calls it only when it decides retrieval helps,
   rather than on every turn.

You can also just call `POST /rag/chat` directly from any script — it does
retrieval + generation server-side and returns `{answer, context, sources,
related_entities}` in one response, no OpenWebUI required.

### Sharing a Docker network

If your external OpenWebUI runs in its own compose file, put both on the
same external network so they can reach each other by container name:

```bash
docker network create graphrag-net
```

Add to both compose files:

```yaml
networks:
  default:
    external:
      name: graphrag-net
```

## Project layout

```
docker-compose.yml
_env.example
eval_retrieval.py    # retrieval hit-rate/MRR evaluation harness (see below)
api/
  Dockerfile
  requirements.txt
  config.py          # env-based settings
  retry.py            # shared retry/backoff helper for Ollama HTTP calls
  embeddings.py         # Ollama embeddings (nomic-embed-text by default)
  vector_store.py        # Qdrant client
  graph_store.py           # Neo4j client + Cypher queries
  file_parsers.py            # PDF / DOCX / DOC / CSV / XLSX / HTML / XML / EML / JSON / TXT extraction
  jobs.py                      # background job tracking (in-memory + JSON-file backed)
  llm.py                         # Ollama chat client (streaming, options)
  graph_extraction.py              # LLM-based entity/relationship extraction
  ingest.py                          # chunking + entity extraction + writes
  retrieval.py                         # multi-channel RRF retrieval fusion
  main.py                                # FastAPI app / endpoints
  static/index.html                        # upload webpage
```

## Evaluating retrieval quality

With five weighted channels and a handful of tunable thresholds, "does this
feel better" isn't a great way to judge a config change. `eval_retrieval.py`
runs a small hand-labeled query set against `/retrieve` and reports
hit-rate@k and MRR:

```bash
cat > eval_set.json << 'EOF'
[
  {"query": "who founded Acme Corp?", "expected_doc_id": "acme-history"},
  {"query": "what is the return policy?", "expected_doc_id": "policies-2024"}
]
EOF

python eval_retrieval.py eval_set.json
```

Build the set from queries you've actually asked (or expect to) and the
document you know the answer lives in — 15-30 cases is enough to start
noticing whether a `RAG_WEIGHT_*` or `RAG_MIN_SCORE` change helped or hurt.
A hit counts if `expected_doc_id` appears anywhere in the top-k results,
not at the individual-chunk level — coarser, but much less tedious to
hand-label and maintain.

## Troubleshooting: Ollama fails to load a model / 502 errors

If `/v1/chat/completions` (or OpenWebUI) returns a 502, check
`docker compose logs ollama` — if you see something like:

```
llama_model_load: error loading model: read error: Cannot allocate memory
```

that's Ollama (or the VM Docker Desktop runs in, on Mac/Windows) running out
of free RAM while trying to load the model into memory. This is a host
resource limit, not an API bug — the API is correctly surfacing Ollama's
real failure as a 502 rather than hiding it.

**Rough memory needed per model** (Q4 quantization, the Ollama default):
roughly 0.6–0.7 GB per billion parameters, plus a couple GB of overhead —
so `llama3.1:8b` needs ~6GB free, `qwen2.5:14b` needs ~10GB free. Add
another 1–2GB for Qdrant + Neo4j + the API running alongside it.

**To check what's actually available:**
```bash
docker stats                      # live memory per container
docker compose exec ollama ollama ps   # which model(s) are currently loaded
free -h                            # Linux: total/free host RAM
```
On Mac/Windows, Docker Desktop runs everything inside a VM with its own
memory cap — check **Docker Desktop → Settings → Resources → Memory** and
raise it if it's below what your models need.

**Fixes, roughly in order of effort:**
1. `docker-compose.yml` now sets `OLLAMA_MAX_LOADED_MODELS=1`, so Ollama
   unloads the previous model before loading a new one instead of trying to
   hold multiple in memory at once — this alone fixes it if the *sum* of
   models you were switching between didn't fit, even though each
   individually would.
2. If even one model at a time won't fit, use a smaller one:
   `llama3.2:3b`, `qwen2.5:3b`, or `phi3.5` all run comfortably in ~4GB.
3. Raise Docker's memory allocation (Mac/Windows) or add swap (Linux) if
   you want to keep using larger models.
4. If you're switching models rapidly (e.g. testing several back-to-back
   in OpenWebUI or promptfoo), expect a short pause on each switch while
   the old one unloads and the new one loads from disk — that's expected
   with `OLLAMA_MAX_LOADED_MODELS=1`, not a hang.

### Variant: `httpx.ReadTimeout` instead of "Cannot allocate memory"

If the API logs show `httpx.ReadTimeout` (traceback ending in
`raise mapped_exc(message) from exc`) rather than an explicit OOM message,
and it happens right around your `OLLAMA_TIMEOUT` value, check
`docker compose logs ollama` for this line around the same time:

```
level=INFO source=sched.go:... msg="disabling mmap for llama-server
```

This means the same underlying problem — not enough free memory to
memory-map the model file — but Ollama is falling back to a much slower
loading path instead of failing outright, so it just runs past the
request timeout instead of erroring immediately. Two things to do:

1. **Raise `OLLAMA_TIMEOUT`** (default is now 600s / 10 minutes) so slow
   loads aren't cut off mid-way — set it higher in `.env` if even that
   isn't enough on your hardware.
2. **Don't stop at raising the timeout.** "Disabling mmap" affects
   per-token generation speed too, not just load time — so even once it
   succeeds, expect it to be noticeably slower than normal for as long as
   memory stays this tight. If that's the case, the fixes from the section
   above (smaller model, or more memory) address the actual cause; a
   longer timeout just stops it from failing partway through.

## Improving retrieval quality (if RAG seems to hurt more than it helps)

Three things were fixed to address this directly — worth understanding what
changed and what to still tune yourself:

**1. Embedding task prefixes (fixed automatically, but re-ingest to benefit).**
`nomic-embed-text` was trained with task prefixes — `"search_document: "`
for indexed text, `"search_query: "` for queries — and Nomic's own model
card notes retrieval quality is measurably worse without them. Ingestion
and querying now apply these automatically
(`EMBEDDING_DOC_PREFIX`/`EMBEDDING_QUERY_PREFIX` in `.env` if you ever need
to change or disable it for a different embedding model). **This only
affects newly-embedded text** — documents ingested before this change used
no prefix, so their vectors are somewhat inconsistent with new ones.
Re-ingest existing documents (same `doc_id`) to get the full benefit.

**2. A relevance floor, so irrelevant chunks aren't force-fed (off by default — needs tuning).**
Vector search always returns the top-k *closest* chunks, even if none of
them are actually relevant — there's no "no good match" case by default.
`RAG_MIN_SCORE` (0-1, cosine similarity) lets you set a floor below which
matches are dropped entirely, so a question unrelated to your knowledge
base returns no context instead of forcing in whatever ranked highest by
default. It's **disabled by default (0)** because the right threshold
depends entirely on your embedding model and data — a wrong guess could
just as easily filter out genuinely relevant matches. To find a good value:

```bash
# Try a query you know SHOULD match your data well:
curl -G "http://localhost:8000/retrieve" --data-urlencode "query=<something clearly in your docs>"
# note the vector scores in the response

# Try a query you know should NOT match anything in your data:
curl -G "http://localhost:8000/retrieve" --data-urlencode "query=<something unrelated>"
# note those scores too
```

Pick a threshold between the two clusters (e.g. if relevant queries score
0.6-0.8 and irrelevant ones score 0.2-0.4, try `RAG_MIN_SCORE=0.5`), set it
in `.env`, and re-test both queries. You can also override it per-request
via `"min_score"` on `/retrieve`, `/rag/chat`, or `/v1/chat/completions`
without changing the global setting, which is the fastest way to iterate
on a value.

**3. A system prompt that tells the model to ignore irrelevant context.**
Even with good retrieval, small/quantized local models can get derailed by
context that's present but not actually relevant — instead of ignoring it,
they sometimes try to force it into the answer. The default
`RAG_SYSTEM_PROMPT` now explicitly instructs: if the context isn't
relevant, ignore it and answer normally. If you're still seeing this after
the above, it's worth testing whether a different/larger model follows
that instruction more reliably — this is a real limitation of smaller
models, not something a prompt can fully guarantee.

**Other levers worth knowing about:**
- `graph_expand` (default 5) adds chunks that merely *share an entity*
  with a vector match, which can pull in tangentially-related content.
  Try `graph_expand=0` per-request (or lower the default) to isolate
  whether graph expansion specifically is the source of noisy context.
- `top_k` (default 5) — fewer, higher-confidence chunks are often better
  than more, weaker ones. Try lowering it before raising it.
- The sources footer (see below) shows exactly what was retrieved for any
  given answer — the fastest way to confirm whether a bad answer traces
  back to bad retrieval or the model ignoring good retrieval.

## Troubleshooting: ingestion times out on large files

If a large file's job status shows `error` with something like "timed out"
after processing a while — this used to be a real scaling gap: earlier
versions embedded and wrote an entire document's chunks in one request. A
50MB file can chunk into 60,000+ pieces at the default `CHUNK_SIZE`, and
embedding all of them in a single Ollama call could take far longer than
any timeout, however high.

This is fixed — ingestion now processes chunks in batches of
`EMBED_BATCH_SIZE` (default 32), and the upload page shows live progress
(`processed / total chunks`) instead of just a spinner, so you can tell
it's actively working rather than stuck. Every Ollama call (embedding,
extraction) also retries transient failures automatically
(`OLLAMA_RETRY_ATTEMPTS`/`OLLAMA_RETRY_BASE_DELAY`) instead of failing the
whole job on one blip, and per-batch timing (`embed=`/`extract=`/`upsert=`
seconds) is logged at INFO level so you can see exactly which stage is
slow rather than guessing. If you still hit timeouts on very large files:

1. **Lower `EMBED_BATCH_SIZE`** (e.g. `8`–`16`) if individual batches are
   slow on constrained hardware — smaller batches mean each Ollama request
   finishes faster, at the cost of more total requests.
2. **Check `docker compose logs api`** — ingestion failures are now logged
   server-side with a full traceback (`logger.exception(...)`), even when
   the job's `error` field is terse or empty, so the real cause (Ollama
   overload, Neo4j slowness, a malformed chunk) should be visible there.
   The per-batch timing lines tell you whether it's the embedding call,
   entity extraction, or the Qdrant/Neo4j writes that's actually slow.
3. **Raise `GRAPH_EXTRACTION_CONCURRENCY`** (and `OLLAMA_NUM_PARALLEL` on
   the `ollama` service) if extraction time dominates each batch — this
   is usually the biggest lever, since extraction is one Ollama call per
   chunk versus one batched call for embedding.
4. If it's consistently failing partway through a very large file rather
   than just being slow, that's more likely the underlying memory pressure
   from the sections above than a batching issue — check `docker stats`
   while ingestion is running.

## Diagnosing "nothing works" — `/health/detailed`

Most of the setup issues that come up with this stack (an untagged model
name resolving to the wrong tag, APOC not enabled, `.env` not actually
reaching the container) look identical from the outside: retrieval or chat
just silently returns nothing useful. `GET /health/detailed` checks Qdrant,
Neo4j, and Ollama connectivity directly, and specifically checks whether
`OLLAMA_MODEL` / `EMBEDDING_MODEL` / `GRAPH_EXTRACTION_MODEL` are pulled
under the *exact* tag configured — the single most common cause of a
confusing 404 further downstream:

```bash
curl http://localhost:8000/health/detailed | python3 -m json.tool
```

A `"missing_models"` entry under `checks.ollama` means a configured model
isn't pulled under that exact tag — compare against
`docker compose exec ollama ollama list` and fix whichever's wrong (the
model name, or the tag).

<a id="known-limitations"></a>
## Known limitations

- **Archives and PPTX aren't supported.** A `.zip` containing multiple
  documents, or a `.pptx` slide deck, will be rejected — supporting either
  is a larger addition (recursive ingestion for archives; a new parser
  dependency for PPTX) than fits the current file-type list.
- **Entity resolution is exact-key only.** `"Bob Smith"` and `"Robert
  Smith"` will end up as two separate Entity nodes — there's no fuzzy/
  semantic alias resolution.
- **No auth.** Anyone who can reach the API can ingest, retrieve, delete
  documents, or trigger generation — fine for local/private-network use,
  not for anything exposed more broadly without a reverse proxy or API
  key layer in front of it.
- **Keyword search is bounded, not exhaustive.** The keyword channel ranks
  client-side over a wide-but-finite candidate pool
  (`RAG_KEYWORD_SCROLL_LIMIT`, default 500) rather than a true full-text
  search engine — fine unless a single query's keywords match hundreds of
  chunks across your corpus.
- **The optional LLM-assist retrieval steps (rerank/condense/expand) add
  real latency** — each is one extra Ollama round-trip per query when
  enabled. They're off by default for this reason; turn them on
  deliberately once you've confirmed the baseline fusion pipeline behaves
  the way you want.

## Notes / simplifications

- Embeddings run through Ollama (`nomic-embed-text` by default) rather than
  a bundled Python library — smaller Docker image, but it means the API
  can't create vectors until that model is pulled. On first boot, if you
  haven't pulled it yet, the Qdrant collection just isn't created until the
  first successful ingest/retrieve after you do — the API itself starts
  fine either way.
- Entities and relationships are extracted by your local LLM per chunk
  (`graph_extraction.py`), not a fixed NER model — relation types are
  whatever the model returns (`WORKS_AT`, `FOUNDED`, ...), stored as real
  Neo4j relationship types via the APOC plugin (`NEO4J_PLUGINS=["apoc"]` in
  `docker-compose.yml`). Quality depends on the extraction model — a small
  quantized model will produce noisier/sparser relations than a larger one;
  bad or unparseable output for a chunk is skipped (logged as a warning),
  not a fatal ingest error, so vector search for that chunk still works
  even if graph extraction fails.
- Entity names are merged on a normalized (lowercased, whitespace-collapsed)
  key so `"Microsoft"` / `"microsoft"` collapse into one node — see
  [Known limitations](#known-limitations) for what this doesn't catch.
- Chunking is a simple character-window splitter with paragraph/sentence-
  aware boundaries, not a semantic chunker — but it now runs per-block
  (per PDF page, per docx section, per batch of spreadsheet rows, ...)
  rather than across the whole document at once, so a chunk never spans
  two unrelated blocks even though the splitter itself stays simple.
- No auth on the API or the upload page — add an API key check / put it
  behind a reverse proxy with auth before exposing this beyond
  localhost/a private network.
- To re-ingest a document, upload/POST it again with the same `doc_id`;
  chunk IDs are deterministic (`uuid5` of `doc_id:index`) so vectors get
  overwritten. If the new version is *shorter* than the old one, though,
  the extra chunks/graph data from the longer version aren't automatically
  pruned — `DELETE /documents/{doc_id}` first, then re-ingest, if you need
  a clean replace rather than an overwrite-in-place.
- Job status is tracked in memory plus persisted to a small JSON file on
  the `api_uploads` volume — so it survives container restarts/rebuilds,
  but it's still a single-process store (a file, not a database), and the
  upload page stops polling and shows an error after a few consecutive
  404s (e.g. if a job record ever does go missing) instead of polling
  forever. Fine for a single-container setup; move to Redis or a DB table
  if you scale to multiple API replicas.
- PDF extraction uses the text layer first, falling back to OCR
  (`pytesseract` + `pdf2image`) only for pages where that finds nothing —
  set `OCR_ENABLED=false` to skip OCR entirely (native-text pages are
  unaffected either way).
- Ollama generation defaults to non-streaming for `/rag/chat`; the
  OpenAI-compatible `/v1/chat/completions` supports streaming (`"stream":
  true`), which is what OpenWebUI uses by default for a responsive feel.
- Temperature, top-p, max tokens, and seed are all passed straight through
  to Ollama's `options` object per-request — nothing is hardcoded, so
  OpenWebUI's per-chat settings and promptfoo's per-test-case config both
  take effect immediately, no restart needed.
- The RAG system prompt (`RAG_SYSTEM_PROMPT` env var) asks the model to
  answer from context and admit when it can't — it's a prompt-level nudge,
  not a hard guarantee against hallucination; treat outputs accordingly for
  anything high-stakes.
- No GPU is configured by default for Ollama — CPU inference works but is
  slow for anything beyond small models. Uncomment the GPU section in
  `docker-compose.yml` if you have an NVIDIA GPU + the NVIDIA Container
  Toolkit installed.
