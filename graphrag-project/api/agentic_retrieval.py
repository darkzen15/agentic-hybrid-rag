"""
Agentic retrieval pipeline - sits on top of hybrid_retrieve rather than
replacing it. The existing single-shot retrieval path stays untouched;
this module adds a multi-step reasoning loop that decomposes complex
queries, grades results, and self-corrects when retrieval is weak.

Flow:
  1. DECOMPOSE: the LLM breaks the user's query into 1-N focused
     sub-questions (a simple factual query stays as one; a comparative
     or multi-part question gets split into its natural parts).
  2. RETRIEVE: each sub-question runs through hybrid_retrieve
     independently (all four channels: vector/entity/graph/keyword).
  3. GRADE: the LLM scores each retrieved chunk's relevance to the
     sub-question that pulled it in. Chunks below the relevance
     threshold are discarded.
  4. SELF-CORRECT: if too few chunks survived grading for a given
     sub-question, the LLM reformulates (broader terms, different
     angle, synonyms) and hybrid_retrieve runs again with the new
     query. This retry loop is capped (RAG_AGENTIC_MAX_RETRIES).
  5. MERGE: all surviving chunks across all sub-questions are
     deduplicated and merged into one result set.

Designed for strong models (120B+ class) where the LLM loop decisions
are reliable and the per-round-trip cost is worth the improved recall
on complex queries. On a weaker model the grading/reformulation steps
may add latency without improving quality — that's why this is behind
a toggle (RAG_AGENTIC_ENABLED) and the existing fast single-shot path
(/rag/chat) always stays available.
"""
import json
import logging
import time
from typing import List, Dict, Any, Optional

from config import settings
import retrieval as _retrieval

# Access through the module rather than binding at import time, so the
# functions can be patched/mocked on the retrieval module and the change
# is visible here too (a direct `from retrieval import X` would snapshot
# the reference at import time and never see a later patch).
hybrid_retrieve = lambda *a, **kw: _retrieval.hybrid_retrieve(*a, **kw)


def _ollama_chat_json(*a, **kw):
    return _retrieval._ollama_chat_json(*a, **kw)

logger = logging.getLogger("graphrag")


def _decompose_query(query: str) -> List[str]:
    """
    Asks the LLM to break a potentially complex query into focused
    sub-questions. A simple factual query ("who is APT28?") should come
    back as a single sub-question (itself); a comparative or multi-part
    query ("compare APT28 and APT29's TTPs and their known malware
    families") should decompose into its natural parts.

    Returns the original query as a single-element list if decomposition
    fails or if the model determines the query is already focused enough.
    """
    prompt = (
        f"You are a search query planner. Break the following question into "
        f"focused sub-questions that can each be answered independently via "
        f"document retrieval. If the question is already simple and focused, "
        f"return it as-is in a single-element list.\n\n"
        f"Rules:\n"
        f"- Return ONLY a JSON array of strings, nothing else.\n"
        f"- Each sub-question must be self-contained (no pronouns referring to other sub-questions).\n"
        f"- Do not add sub-questions the original doesn't ask for.\n"
        f"- Maximum {settings.RAG_AGENTIC_MAX_SUBQUESTIONS} sub-questions.\n\n"
        f"Question: {query}"
    )
    raw = _ollama_chat_json(
        prompt,
        model=settings.RAG_ASSIST_MODEL,
        max_tokens=256,
        temperature=0,
        label="agentic_decompose",
    )
    if not raw:
        return [query]
    try:
        result = json.loads(raw)
        if isinstance(result, list) and result:
            subs = [s.strip() for s in result if isinstance(s, str) and s.strip()]
            if subs:
                logger.info(f"agentic_decompose: {query!r} -> {subs}")
                return subs[: settings.RAG_AGENTIC_MAX_SUBQUESTIONS]
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"agentic_decompose: non-JSON response, using original query: {raw[:200]!r}")
    return [query]


def _grade_chunks(
    sub_question: str, chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Asks the LLM to score each chunk's relevance to the sub-question on
    a 0-10 scale, then keeps only those above the threshold. This is
    different from the optional RAG_RERANK_ENABLED step in retrieval.py:
    reranking re-sorts candidates but keeps all of them, while grading
    *discards* irrelevant ones to decide whether self-correction is needed.

    Returns the chunks that passed, each with a 'relevance_score' added.
    Falls back to returning all chunks unfiltered on any failure (the
    agent loop still works, it just can't discard weak results that round).
    """
    if not chunks:
        return []

    numbered = "\n".join(
        f"[{i}] {(c.get('text') or '')[:500]}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"You are a relevance grader for a retrieval-augmented system.\n\n"
        f"Sub-question: {sub_question}\n\n"
        f"Retrieved passages:\n{numbered}\n\n"
        f"Score each passage 0-10 for how relevant it is to answering the "
        f"sub-question. 0 = completely irrelevant, 10 = directly answers it.\n"
        f'Return ONLY a JSON array like [{{"index": 0, "score": 8}}, ...] '
        f"with one entry per passage. No other text."
    )
    raw = _ollama_chat_json(
        prompt,
        model=settings.RAG_ASSIST_MODEL,
        max_tokens=512,
        temperature=0,
        label="agentic_grade",
    )
    if not raw:
        return chunks  # fallback: keep all

    try:
        scores = json.loads(raw)
        score_map = {
            int(item["index"]): float(item.get("score", 0))
            for item in scores
            if "index" in item
        }
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        logger.warning(f"agentic_grade: unusable response, keeping all chunks: {raw[:200]!r}")
        return chunks

    threshold = settings.RAG_AGENTIC_RELEVANCE_THRESHOLD
    graded = []
    for i, chunk in enumerate(chunks):
        score = score_map.get(i, 0)
        if score >= threshold:
            chunk["relevance_score"] = score
            graded.append(chunk)

    logger.info(
        f"agentic_grade: {len(graded)}/{len(chunks)} chunks passed "
        f"threshold {threshold} for {sub_question!r}"
    )
    return graded


def _reformulate_query(original_sub: str, attempt: int) -> Optional[str]:
    """
    Asks the LLM to broaden or rephrase a sub-question that didn't
    retrieve enough relevant results. Each retry should try a genuinely
    different angle, not just swap one synonym.

    Returns the reformulated query, or None if the LLM fails (in which
    case the agent loop stops retrying for this sub-question).
    """
    prompt = (
        f"A document search for the following question returned poor results.\n\n"
        f"Original question: {original_sub}\n"
        f"Retry attempt: {attempt}\n\n"
        f"Rewrite the question to improve search recall. Strategies:\n"
        f"- Use broader or more general terms\n"
        f"- Try synonyms or alternate names for key entities\n"
        f"- Remove overly specific constraints\n"
        f"- Ask about the topic from a different angle\n\n"
        f"Return ONLY the rewritten question, nothing else."
    )
    raw = _ollama_chat_json(
        prompt,
        model=settings.RAG_ASSIST_MODEL,
        max_tokens=128,
        temperature=0.3,
        label="agentic_reformulate",
    )
    reformulated = (raw or "").strip()
    if reformulated and reformulated != original_sub:
        logger.info(f"agentic_reformulate: {original_sub!r} -> {reformulated!r} (attempt {attempt})")
        return reformulated
    return None


def _retrieve_and_grade_subquestion(
    sub_question: str,
    top_k: int,
    graph_expand: int,
    doc_id: Optional[str],
    min_score: Optional[float],
    history: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """
    Runs the full retrieve-grade-retry loop for a single sub-question.
    Returns the final set of graded chunks plus metadata about the loop
    (how many attempts it took, which reformulations it tried).
    """
    min_relevant = max(1, int(top_k * settings.RAG_AGENTIC_MIN_RELEVANT_RATIO))
    attempts = []
    all_graded: List[Dict[str, Any]] = []
    current_query = sub_question

    for attempt in range(1 + settings.RAG_AGENTIC_MAX_RETRIES):
        retrieval = hybrid_retrieve(
            query=current_query,
            top_k=top_k,
            graph_expand=graph_expand,
            doc_id=doc_id,
            min_score=min_score,
            history=history,
        )
        graded = _grade_chunks(sub_question, retrieval["results"])
        attempts.append({
            "query": current_query,
            "retrieved": len(retrieval["results"]),
            "passed_grading": len(graded),
        })

        # Merge graded chunks, deduplicating by chunk_id across retries
        # (a reformulated query might re-find something the original
        # already found, just ranked differently).
        seen = {c["chunk_id"] for c in all_graded}
        for c in graded:
            if c["chunk_id"] not in seen:
                all_graded.append(c)
                seen.add(c["chunk_id"])

        if len(all_graded) >= min_relevant:
            logger.info(
                f"agentic_retrieve: sub-question {sub_question!r} satisfied "
                f"after {attempt + 1} attempt(s) with {len(all_graded)} relevant chunks"
            )
            break

        # Self-correct: reformulate and retry
        if attempt < settings.RAG_AGENTIC_MAX_RETRIES:
            reformulated = _reformulate_query(sub_question, attempt + 1)
            if reformulated:
                current_query = reformulated
            else:
                logger.info(
                    f"agentic_retrieve: reformulation failed for {sub_question!r}, "
                    f"stopping retries with {len(all_graded)} chunks"
                )
                break

    return {
        "sub_question": sub_question,
        "chunks": all_graded,
        "attempts": attempts,
        "related_entities": retrieval.get("related_entities", []),
    }


def agentic_retrieve(
    query: str,
    top_k: int = 5,
    graph_expand: int = 5,
    doc_id: Optional[str] = None,
    min_score: Optional[float] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Full agentic retrieval pipeline: decompose -> retrieve -> grade ->
    self-correct -> merge. Returns the same top-level shape as
    hybrid_retrieve (results, related_entities, timings) so callers
    can use either interchangeably, plus extra agentic metadata
    (sub_questions, per-subquestion attempt logs).
    """
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    # --- Step 1: Decompose ---
    t0 = time.perf_counter()
    sub_questions = _decompose_query(query)
    timings["decompose"] = round(time.perf_counter() - t0, 3)

    # --- Steps 2-4: Retrieve + grade + self-correct per sub-question ---
    t0 = time.perf_counter()
    sub_results = []
    for sub_q in sub_questions:
        sub_result = _retrieve_and_grade_subquestion(
            sub_question=sub_q,
            top_k=top_k,
            graph_expand=graph_expand,
            doc_id=doc_id,
            min_score=min_score,
            history=history,
        )
        sub_results.append(sub_result)
    timings["retrieve_grade_loop"] = round(time.perf_counter() - t0, 3)

    # --- Step 5: Merge across sub-questions ---
    seen_ids = set()
    merged: List[Dict[str, Any]] = []
    all_entities: List[Dict[str, Any]] = []

    for sr in sub_results:
        for chunk in sr["chunks"]:
            if chunk["chunk_id"] not in seen_ids:
                # Tag each chunk with which sub-question found it, for
                # transparency in the response
                chunk["sub_question"] = sr["sub_question"]
                merged.append(chunk)
                seen_ids.add(chunk["chunk_id"])
        all_entities.extend(sr.get("related_entities", []))

    # Sort by relevance score (from grading), falling back to RRF score
    merged.sort(
        key=lambda c: c.get("relevance_score", c.get("score", 0)),
        reverse=True,
    )

    # Deduplicate entities by name
    seen_entity_names = set()
    deduped_entities = []
    for e in all_entities:
        if e.get("name") and e["name"].lower() not in seen_entity_names:
            seen_entity_names.add(e["name"].lower())
            deduped_entities.append(e)

    timings["total"] = round(time.perf_counter() - t_total, 3)

    logger.info(
        f"agentic_retrieve: {query!r} -> {len(sub_questions)} sub-questions, "
        f"{len(merged)} total chunks, {timings['total']:.2f}s"
    )

    return {
        "query": query,
        "results": merged,
        "related_entities": deduped_entities,
        "timings": timings,
        "agentic": {
            "sub_questions": [
                {
                    "sub_question": sr["sub_question"],
                    "chunks_kept": len(sr["chunks"]),
                    "attempts": sr["attempts"],
                }
                for sr in sub_results
            ],
        },
    }
