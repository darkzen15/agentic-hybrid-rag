import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from config import settings
from embeddings import embed_text
import vector_store
import graph_store
from retry import with_retry, is_retryable_ollama_error
from ollama_client import get_sync_client

logger = logging.getLogger("graphrag")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "with", "about", "tell", "me", "what",
    "who", "when", "where", "why", "how", "does", "do", "did", "this",
    "that", "these", "those", "it", "its", "as", "by", "from", "be",
    "been", "has", "have", "had", "can", "could", "would", "should",
}


def _extract_keywords(query: str, min_len: int = 3, max_keywords: int = 8) -> List[str]:
    """Coarse keyword extraction for the lexical retrieval channel: lowercase
    word tokens, drop stopwords and anything too short to be a meaningful
    exact-match term, dedupe while preserving order."""
    words = re.findall(r"[A-Za-z0-9_]+", query.lower())
    keywords = []
    for w in words:
        if len(w) < min_len or w in _STOPWORDS:
            continue
        if w not in keywords:
            keywords.append(w)
    return keywords[:max_keywords]


def _rrf_merge(
    channel_ranks: Dict[str, List[str]], weights: Dict[str, float], rrf_k: int
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion: each channel contributes
    weight / (rrf_k + rank_in_channel + 1) per chunk_id, summed across
    channels. A chunk that multiple independent channels agree on rises to
    the top naturally - this is the standard way to combine rankings from
    sources whose raw scores aren't on comparable scales (cosine
    similarity vs. entity-overlap counts vs. keyword-overlap counts).
    """
    scores: Dict[str, float] = {}
    for channel, ids in channel_ranks.items():
        weight = weights.get(channel, 1.0)
        for rank, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank + 1)
    return scores


def _is_near_duplicate(text: str, existing_texts: List[str], min_overlap_ratio: float) -> bool:
    """
    Cheap near-duplicate check for the final merged list: catches the
    common case where two adjacent/overlapping chunks (CHUNK_OVERLAP means
    they legitimately share text) both make it into results, wasting
    context-window budget on largely redundant content. Not a general
    similarity measure - just checks whether a large fraction of one
    chunk's words already appear in an already-selected chunk's text.
    """
    words = set(text.lower().split())
    if not words:
        return False
    for existing in existing_texts:
        existing_words = set(existing.lower().split())
        if not existing_words:
            continue
        overlap = len(words & existing_words) / len(words)
        if overlap >= min_overlap_ratio:
            return True
    return False


def _ollama_chat_json(prompt: str, model: Optional[str], max_tokens: int, temperature: float, label: str) -> Optional[str]:
    """
    Shared helper for retrieval.py's optional LLM-assist steps (query
    condensation, expansion, reranking) - a single-message, JSON-or-plain
    sync call to Ollama with retry, matching the pattern already used in
    graph_extraction.py. Returns None (rather than raising) on failure, so
    every caller can fall back to its non-LLM-assisted behavior.
    """
    payload = {
        "model": model or settings.GRAPH_EXTRACTION_MODEL or settings.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    def _do_request():
        client = get_sync_client()
        resp = client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    try:
        data = with_retry(
            _do_request,
            attempts=settings.OLLAMA_RETRY_ATTEMPTS,
            base_delay=settings.OLLAMA_RETRY_BASE_DELAY,
            retryable=is_retryable_ollama_error,
            label=label,
        )
        return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.warning(f"{label}: Ollama request failed after retries, skipping ({e})")
        return None


def _condense_query(query: str, history: List[Dict[str, str]]) -> str:
    """
    Rewrites a follow-up question into a standalone query using recent
    conversation history, before embedding/searching. Without this, a
    follow-up like "what about its founder?" embeds almost meaninglessly
    on its own - retrieval needs the antecedent from prior turns that the
    embedding of the follow-up alone doesn't carry. Only called when
    RAG_CONDENSE_ENABLED is set, since it's one extra LLM call per turn.
    """
    if not history:
        return query
    recent = history[-settings.RAG_CONDENSE_HISTORY_TURNS :]
    convo = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in recent)
    prompt = (
        f"Conversation so far:\n{convo}\n\nFollow-up question: {query}\n\n"
        f"Rewrite the follow-up question as a standalone question that "
        f"makes sense without the conversation history. If it's already "
        f"standalone, return it unchanged. Return ONLY the rewritten "
        f"question, nothing else."
    )
    raw = _ollama_chat_json(prompt, settings.RAG_ASSIST_MODEL, max_tokens=128, temperature=0, label="condense_query")
    rewritten = (raw or "").strip()
    if rewritten:
        logger.info(f"condense_query: {query!r} -> {rewritten!r}")
        return rewritten
    return query


def _expand_queries(query: str, n: int) -> List[str]:
    """
    Generates n alternate phrasings of the query via the LLM (multi-query
    expansion) so retrieval isn't fully dependent on one specific wording
    embedding well. Only called when RAG_QUERY_EXPANSION_ENABLED is set,
    since each variant costs its own embed+search round-trip.
    """
    prompt = (
        f"Generate {n} alternate phrasings of this question that preserve "
        f"its meaning but use different words/structure, to improve "
        f"search recall. Return ONLY a JSON array of {n} strings, nothing "
        f"else.\n\nQuestion: {query}"
    )
    raw = _ollama_chat_json(prompt, settings.RAG_ASSIST_MODEL, max_tokens=256, temperature=0.3, label="expand_queries")
    if not raw:
        return []
    try:
        variants = json.loads(raw)
        return [v.strip() for v in variants if isinstance(v, str) and v.strip()][:n]
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"expand_queries: model returned non-JSON output, skipping: {raw[:200]!r}")
        return []


def _rerank_with_llm(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Asks the LLM to score each fused candidate's relevance to the query on
    a 0-10 scale, then re-sorts by that score. RRF fusion combines
    *rankings* well but never actually re-reads the candidates against the
    query the way this does - typically improves precision on the final
    top_k. Only called when RAG_RERANK_ENABLED is set, since it's one
    extra LLM call per retrieval. Falls back to the existing RRF order on
    any failure (bad JSON, request failure) rather than dropping results.
    """
    if not candidates:
        return candidates
    numbered = "\n".join(f"[{i}] {(c['text'] or '')[:400]}" for i, c in enumerate(candidates))
    prompt = (
        f"Question: {query}\n\nCandidate passages:\n{numbered}\n\n"
        f'Return ONLY a JSON array of objects like [{{"index": 0, "score": 8}}, ...] '
        f"- one entry per candidate above, scoring 0-10 how relevant each "
        f"passage is to answering the question. No other text."
    )
    raw = _ollama_chat_json(prompt, settings.RAG_ASSIST_MODEL, max_tokens=512, temperature=0, label="rerank")
    if not raw:
        return candidates
    try:
        scores_data = json.loads(raw)
        score_by_index = {int(item["index"]): float(item.get("score", 0)) for item in scores_data if "index" in item}
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        logger.warning(f"rerank: model returned unusable output, keeping RRF order: {raw[:200]!r}")
        return candidates

    scored = [(score_by_index.get(i, 0.0), c) for i, c in enumerate(candidates)]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    graph_expand: int = 5,
    doc_id: str = None,
    min_score: Optional[float] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Runs several independent retrieval channels and fuses them with RRF,
    rather than a single vector-search-then-expand pipeline. Each channel
    gets a real chance to contribute regardless of how the others perform:

    - vector: Qdrant semantic search over chunk embeddings (optionally
      widened by LLM query expansion - see RAG_QUERY_EXPANSION_ENABLED)
    - entity_name: literal entity-name matches from the query, looked up
      directly in Neo4j (catches "tell me about <specific name>" queries
      that vector search alone can miss if the semantic match is weak)
    - graph: 1-hop relationship expansion in Neo4j, seeded from *both* the
      vector and entity_name channels, weighted by relationship strength
    - keyword: Qdrant full-text filter on chunk text (catches exact
      strings - IDs, codes, acronyms - that embeddings blur)

    A chunk found by multiple channels outranks one found strongly by only
    one. A per-document cap keeps one document from crowding out the
    merged result set, and a cheap near-duplicate check drops candidates
    that mostly overlap an already-selected chunk. Graph/entity/keyword
    channels each fail soft (log + continue with an empty channel) so a
    Neo4j hiccup degrades to vector-only instead of failing the whole
    request - only the vector channel itself is allowed to raise, since
    that's the one case main.py already surfaces as a clear 502.

    Optional (off by default, each adds LLM-call latency): RAG_CONDENSE_
    ENABLED rewrites multi-turn follow-ups into standalone queries using
    `history` before retrieval; RAG_QUERY_EXPANSION_ENABLED widens the
    vector channel with alternate phrasings; RAG_RERANK_ENABLED re-scores
    the fused candidate pool with the LLM before truncating to top_k.

    Separately, if the query names two or more known entities, a direct
    relationship-path lookup between them is added to the results as its
    own context entry (see graph_store.find_relationship_paths).
    """
    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    threshold = min_score if min_score is not None else settings.RAG_MIN_SCORE
    threshold = threshold if threshold and threshold > 0 else None
    candidate_limit = max(top_k * settings.RAG_CANDIDATE_MULTIPLIER, top_k)

    # --- Optional: condense a multi-turn follow-up into a standalone query ---
    effective_query = query
    if settings.RAG_CONDENSE_ENABLED and history:
        t0 = time.perf_counter()
        effective_query = _condense_query(query, history)
        timings["condense"] = round(time.perf_counter() - t0, 3)

    payload_cache: Dict[str, Dict[str, Any]] = {}

    # --- Independent channels (vector, entity_name, keyword) run
    # concurrently ---
    # These three have no data dependency on each other - they're
    # blocking I/O calls to two different backends (Qdrant for vector +
    # keyword, Neo4j for entity_name), so running them in a thread pool
    # overlaps the network waits instead of paying them end to end. Only
    # the graph channel below genuinely has to wait, since it's seeded
    # from the vector + entity_name results. Each channel keeps the exact
    # same fail-soft behavior it had when sequential: it returns its own
    # (possibly empty) result and its own timing, and an exception inside
    # one channel never takes down the others. The vector channel is the
    # one exception to fail-soft - it's allowed to raise (main.py surfaces
    # it as a clear 502), so its exception is re-raised after the pool
    # joins rather than swallowed.

    def _run_vector() -> Dict[str, Any]:
        t0 = time.perf_counter()
        query_vector = embed_text(effective_query, prefix=settings.EMBEDDING_QUERY_PREFIX)
        hits = vector_store.search(
            query_vector, top_k=candidate_limit, doc_id=doc_id, score_threshold=threshold
        )
        ids = [h["chunk_id"] for h in hits]
        cache = {h["chunk_id"]: h for h in hits}

        # Query expansion is folded into the vector channel (extra
        # embed+search per variant), so it belongs inside this channel's
        # concurrent slot rather than adding a serial step after it.
        expansion_time = None
        if settings.RAG_QUERY_EXPANSION_ENABLED:
            te = time.perf_counter()
            try:
                for variant in _expand_queries(effective_query, settings.RAG_QUERY_EXPANSION_COUNT):
                    variant_vector = embed_text(variant, prefix=settings.EMBEDDING_QUERY_PREFIX)
                    variant_hits = vector_store.search(
                        variant_vector, top_k=candidate_limit, doc_id=doc_id, score_threshold=threshold
                    )
                    for h in variant_hits:
                        if h["chunk_id"] not in ids:
                            ids.append(h["chunk_id"])
                            cache.setdefault(h["chunk_id"], h)
            except Exception:
                logger.warning("hybrid_retrieve: query expansion failed, continuing without it", exc_info=True)
            expansion_time = round(time.perf_counter() - te, 3)
        return {"ids": ids, "cache": cache, "time": round(time.perf_counter() - t0, 3), "expansion_time": expansion_time}

    def _run_entity_name() -> Dict[str, Any]:
        t0 = time.perf_counter()
        named: List[Dict[str, Any]] = []
        ids: List[str] = []
        try:
            named = graph_store.find_entities_in_text(effective_query)
            for ent in named:
                for cid in ent["chunk_ids"]:
                    if cid not in ids:
                        ids.append(cid)
        except Exception:
            logger.warning("hybrid_retrieve: entity-name lookup failed, continuing without it", exc_info=True)
        return {"named_entities": named, "ids": ids, "time": round(time.perf_counter() - t0, 3)}

    def _run_keyword() -> Dict[str, Any]:
        t0 = time.perf_counter()
        ids: List[str] = []
        cache: Dict[str, Dict[str, Any]] = {}
        try:
            keywords = _extract_keywords(effective_query)
            if keywords:
                keyword_hits = vector_store.keyword_search(keywords, limit=candidate_limit, doc_id=doc_id)
                for h in keyword_hits:
                    ids.append(h["chunk_id"])
                    cache[h["chunk_id"]] = h
        except Exception:
            logger.warning("hybrid_retrieve: keyword search failed, continuing without it", exc_info=True)
        return {"ids": ids, "cache": cache, "time": round(time.perf_counter() - t0, 3)}

    t_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_vector = executor.submit(_run_vector)
        f_entity = executor.submit(_run_entity_name)
        f_keyword = executor.submit(_run_keyword)
        # .result() re-raises any exception from inside the channel. Only
        # the vector future is allowed to propagate (fail-hard -> 502);
        # the other two already swallow their own exceptions internally
        # and can only return normally.
        vector_res = f_vector.result()
        entity_res = f_entity.result()
        keyword_res = f_keyword.result()

    vector_channel_ids = vector_res["ids"]
    payload_cache.update(vector_res["cache"])
    timings["vector"] = vector_res["time"]
    if vector_res["expansion_time"] is not None:
        timings["query_expansion"] = vector_res["expansion_time"]

    named_entities = entity_res["named_entities"]
    entity_channel_ids = entity_res["ids"]
    timings["entity_name"] = entity_res["time"]

    keyword_channel_ids = keyword_res["ids"]
    for cid, h in keyword_res["cache"].items():
        payload_cache.setdefault(cid, h)
    timings["keyword"] = keyword_res["time"]
    timings["parallel_channels_wall"] = round(time.perf_counter() - t_parallel, 3)

    # --- Channel: graph expansion, seeded from vector + entity-name hits ---
    # Genuinely dependent on the two channels above, so it runs after the
    # pool joins rather than inside it.
    t0 = time.perf_counter()
    graph_channel_ids: List[str] = []
    try:
        expand_seed_ids = list(dict.fromkeys(vector_channel_ids + entity_channel_ids))
        graph_expansion = graph_store.expand_from_chunks(expand_seed_ids, limit=graph_expand)
        graph_channel_ids = [g["chunk_id"] for g in graph_expansion]
    except Exception:
        logger.warning("hybrid_retrieve: graph expansion failed, continuing without it", exc_info=True)
    timings["graph"] = round(time.perf_counter() - t0, 3)

    # --- Fetch text/doc_id for chunk_ids not already cached from vector/keyword payloads ---
    missing_ids = [
        cid for cid in dict.fromkeys(entity_channel_ids + graph_channel_ids) if cid not in payload_cache
    ]
    if missing_ids:
        fetched = vector_store.get_by_ids(missing_ids)
        for cid, payload in fetched.items():
            payload_cache[cid] = {
                "chunk_id": cid,
                "text": payload.get("text"),
                "doc_id": payload.get("doc_id"),
                "index": payload.get("index"),
            }

    # --- RRF fusion across all channels ---
    channel_ranks = {
        "vector": vector_channel_ids,
        "entity_name": entity_channel_ids,
        "graph": graph_channel_ids,
        "keyword": keyword_channel_ids,
    }
    weights = {
        "vector": settings.RAG_WEIGHT_VECTOR,
        "entity_name": settings.RAG_WEIGHT_ENTITY,
        "graph": settings.RAG_WEIGHT_GRAPH,
        "keyword": settings.RAG_WEIGHT_KEYWORD,
    }
    rrf_scores = _rrf_merge(channel_ranks, weights, settings.RAG_RRF_K)

    contributing: Dict[str, List[str]] = {}
    for channel, ids in channel_ranks.items():
        for cid in ids:
            contributing.setdefault(cid, []).append(channel)

    ranked_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # --- Build final merged list: per-document cap + near-dup suppression ---
    # When reranking is enabled, the pool going into rerank is wider than
    # top_k, since reranking should be able to change *which* candidates
    # survive, not just reorder an already-truncated list.
    pool_size = top_k * settings.RAG_RERANK_POOL_MULTIPLIER if settings.RAG_RERANK_ENABLED else top_k

    merged: List[Dict[str, Any]] = []
    per_doc_count: Dict[str, int] = {}
    selected_texts: List[str] = []
    for cid in ranked_chunk_ids:
        payload = payload_cache.get(cid)
        if not payload or not payload.get("text"):
            # A graph/entity hit whose chunk_id no longer exists in Qdrant
            # (e.g. the doc was re-ingested) - skip rather than surface
            # empty text.
            continue
        text = payload["text"]
        if _is_near_duplicate(text, selected_texts, settings.RAG_DEDUP_OVERLAP_THRESHOLD):
            continue
        d_id = payload.get("doc_id") or "?"
        if per_doc_count.get(d_id, 0) >= settings.RAG_MAX_CHUNKS_PER_DOC:
            continue
        per_doc_count[d_id] = per_doc_count.get(d_id, 0) + 1
        selected_texts.append(text)
        merged.append(
            {
                "chunk_id": cid,
                "text": text,
                "doc_id": d_id,
                "score": round(rrf_scores[cid], 5),
                "source": "+".join(sorted(set(contributing.get(cid, [])))),
            }
        )
        if len(merged) >= pool_size:
            break

    # --- Optional: LLM rerank of the fused pool, then truncate to top_k ---
    if settings.RAG_RERANK_ENABLED:
        t0 = time.perf_counter()
        merged = _rerank_with_llm(effective_query, merged)[:top_k]
        timings["rerank"] = round(time.perf_counter() - t0, 3)

    # --- Neighbor-window expansion for the top few strongest hits ---
    if settings.RAG_NEIGHBOR_WINDOW > 0 and merged:
        t0 = time.perf_counter()
        try:
            seen_ids = {m["chunk_id"] for m in merged}
            neighbor_entries = []
            for m in merged[: settings.RAG_NEIGHBOR_EXPAND_TOP_N]:
                idx = payload_cache.get(m["chunk_id"], {}).get("index")
                if idx is None:
                    continue
                neighbors = vector_store.get_neighbors(m["doc_id"], idx, window=settings.RAG_NEIGHBOR_WINDOW)
                for n in neighbors:
                    if n["chunk_id"] in seen_ids or not n.get("text"):
                        continue
                    seen_ids.add(n["chunk_id"])
                    neighbor_entries.append(
                        {
                            "chunk_id": n["chunk_id"],
                            "text": n["text"],
                            "doc_id": n["doc_id"],
                            "score": m["score"],
                            "source": "neighbor",
                        }
                    )
            merged.extend(neighbor_entries)
        except Exception:
            logger.warning("hybrid_retrieve: neighbor expansion failed, continuing without it", exc_info=True)
        timings["neighbor_expand"] = round(time.perf_counter() - t0, 3)

    # --- Relationship-path lookup between named entities in the query ---
    t0 = time.perf_counter()
    try:
        path_entries = graph_store.find_relationship_paths(named_entities, max_hops=settings.RAG_PATH_MAX_HOPS)
    except Exception:
        logger.warning("hybrid_retrieve: relationship-path lookup failed, continuing without it", exc_info=True)
        path_entries = []
    timings["relationship_paths"] = round(time.perf_counter() - t0, 3)

    for i, p in enumerate(path_entries):
        merged.append(
            {
                "chunk_id": f"path:{i}",
                "text": p["description"],
                "doc_id": "(graph relationship)",
                "score": 1.0,
                "source": "path",
            }
        )

    try:
        entities = graph_store.entities_for_chunks(list(dict.fromkeys(vector_channel_ids + entity_channel_ids)))
    except Exception:
        logger.warning("hybrid_retrieve: entities_for_chunks failed, continuing without it", exc_info=True)
        entities = []

    timings["total"] = round(time.perf_counter() - t_total_start, 3)
    logger.info(f"hybrid_retrieve timings for {query!r}: {timings}")

    return {
        "query": query,
        "results": merged,
        "related_entities": entities,
        "timings": timings,
    }


def format_as_context(retrieval_result: Dict[str, Any]) -> str:
    """Flatten retrieval results into a single context string for RAG prompting."""
    parts = []
    for r in retrieval_result["results"]:
        tag = r.get("source", "vector").upper()
        parts.append(f"[{tag} | doc:{r['doc_id']}] {r['text']}")
    return "\n\n---\n\n".join(parts)
