# Agentic RAG Retrieval — Addon Guide

## What it does

Adds a multi-step reasoning loop on top of the existing hybrid retrieval
pipeline. Instead of a single retrieve-then-answer pass, the system now:

1. **Decomposes** a complex query into focused sub-questions
2. **Retrieves** independently for each sub-question (using all four
   existing channels: vector, entity-name, graph expansion, keyword)
3. **Grades** every retrieved chunk for relevance, discarding weak ones
4. **Self-corrects** — if too few relevant chunks survive grading, the
   LLM reformulates the sub-question (broader terms, different angle)
   and retries retrieval, up to a configurable cap
5. **Merges** all surviving chunks across sub-questions into one
   deduplicated, relevance-ranked result set

The existing single-shot retrieval is completely untouched. The agentic
layer sits on top of it — `hybrid_retrieve` is still the engine under
each sub-question's retrieval step.

---

## Files involved

| File | What changed |
|---|---|
| `api/agentic_retrieval.py` | **New.** The full agentic pipeline |
| `api/config.py` | Added `RAG_AGENTIC_*` settings |
| `api/main.py` | `/rag/chat` and `/v1/chat/completions` route through agentic retrieval when enabled |
| `docker-compose.yml` | Forwards the new env vars to the container |
| `_env.example` | Documents all new settings |

---

## How to enable

```bash
# In your .env file
RAG_AGENTIC_ENABLED=true
```

Then rebuild and restart:

```bash
docker compose build api
docker compose up -d --force-recreate api
```

No re-ingestion needed — this only changes the query path, not how
documents are stored.

---

## Settings

| Setting | Default | What it controls |
|---|---|---|
| `RAG_AGENTIC_ENABLED` | `false` | Master toggle. When off, everything behaves exactly as before |
| `RAG_AGENTIC_MAX_SUBQUESTIONS` | `4` | Cap on how many sub-questions the decomposition step can produce |
| `RAG_AGENTIC_MAX_RETRIES` | `2` | How many reformulate-and-retry cycles per sub-question when grading says results are weak |
| `RAG_AGENTIC_RELEVANCE_THRESHOLD` | `5.0` | Minimum relevance score (0–10) from the grading LLM for a chunk to survive |
| `RAG_AGENTIC_MIN_RELEVANT_RATIO` | `0.4` | Fraction of `top_k` that must survive grading before the agent stops retrying. 0.4 with top_k=5 = at least 2 relevant chunks needed |
| `RAG_ASSIST_MODEL` | (empty) | Model used for decomposition, grading, and reformulation. Falls back to `GRAPH_EXTRACTION_MODEL`, then `OLLAMA_MODEL`. On production hardware, point this at your strongest model |

---

## What stays single-shot

`/retrieve` always uses the original `hybrid_retrieve` regardless of the
toggle — it's a debugging/testing endpoint where you want predictable,
fast, single-pass behavior. The agentic toggle only affects the chat
paths (`/rag/chat` and `/v1/chat/completions`).

---

## Hardware considerations

This is designed for strong models (120B+ class). Each query can involve:

- 1 decomposition call
- N sub-questions × (1 retrieval + 1 grading + up to M retries × (1
  reformulation + 1 retrieval + 1 grading))

Worst case with defaults (4 sub-questions, 2 retries each): ~25 LLM
round-trips before generation starts. On a 120B model with decent
hardware, this is seconds. On an 8B model on a dev machine, it will
be noticeably slow and the decomposition/grading decisions will be
lower quality — which is why `RAG_AGENTIC_ENABLED` defaults to `false`.

Recommendation: **test the toggle on your dev machine** to confirm the
wiring works, but **judge the quality of the agentic loop's decisions**
on your production hardware with the stronger model.

---

## Observability

Every step logs at INFO level with the `graphrag` logger:

```
agentic_decompose: "compare APT28 and APT29" -> ["What is APT28?", "What is APT29?"]
agentic_grade: 3/5 chunks passed threshold 5.0 for "What is APT28?"
agentic_reformulate: "What is APT28?" -> "APT28 threat actor group Fancy Bear GRU" (attempt 1)
agentic_retrieve: "compare APT28 and APT29" -> 2 sub-questions, 6 total chunks, 3.42s
```

The API response also includes an `agentic` field with structured
metadata — sub-questions, per-subquestion attempt counts, and which
queries were tried:

```json
{
  "agentic": {
    "sub_questions": [
      {
        "sub_question": "What is APT28?",
        "chunks_kept": 3,
        "attempts": [
          {"query": "What is APT28?", "retrieved": 5, "passed_grading": 1},
          {"query": "APT28 Fancy Bear GRU threat actor", "retrieved": 5, "passed_grading": 3}
        ]
      }
    ]
  }
}
```

---

## Failure modes (all tested and handled)

| What fails | What happens |
|---|---|
| Decomposition LLM returns garbage or fails | Falls back to treating the original query as a single sub-question |
| Grading LLM returns garbage or fails | All chunks kept unfiltered for that sub-question (no crash, no empty results) |
| Reformulation LLM fails | Stops retrying for that sub-question, keeps whatever chunks survived so far |
| One sub-question finds nothing after all retries | Other sub-questions' results still returned normally |
| All sub-questions find nothing | Returns empty results (same as single-shot would) |
| `hybrid_retrieve` itself fails for one sub-question | Standard fail-soft: vector failure → 502, other channels degrade gracefully |

---

## Offline deployment

No new dependencies, no network calls beyond the existing Ollama
endpoints. All LLM calls go through the same shared `ollama_client.py`
connection pool with the same retry/backoff logic used everywhere else
in the system. The agentic module adds zero new external dependencies
to `requirements.txt` — it's pure Python on top of the existing stack.
