#!/usr/bin/env python3
"""
Lightweight retrieval evaluation harness.

Measures hit-rate@k and Mean Reciprocal Rank (MRR) for the /retrieve
endpoint against a small hand-labeled set of (query, expected_doc_id)
pairs - so changes to RAG_WEIGHT_*, RAG_MIN_SCORE, RAG_RRF_K, etc. can be
judged by a number instead of eyeballing chat responses.

Usage:
    python eval_retrieval.py eval_set.json
    python eval_retrieval.py eval_set.json --api-base http://localhost:8000 --top-k 5

eval_set.json format:
[
    {"query": "who founded Acme Corp?", "expected_doc_id": "acme-history"},
    {"query": "what is the return policy?", "expected_doc_id": "policies-2024"}
]

A hit counts if expected_doc_id appears anywhere in the top_k results'
doc_id field. This is deliberately coarse (document-level, not
chunk-level) since that's usually the granularity you actually care about
- "did retrieval find the right document" - and hand-labeling at the
individual-chunk level is a lot more tedious to build and maintain.

Build the eval set from queries you've actually asked (or expect to ask)
and the document you know the answer lives in. 15-30 cases is enough to
start noticing whether a config change helped or hurt; it doesn't need to
be exhaustive.
"""
import argparse
import json
import sys

import httpx


def evaluate(eval_set, api_base: str, top_k: int):
    total = len(eval_set)
    hits = 0
    reciprocal_ranks = []

    for i, case in enumerate(eval_set, 1):
        query = case["query"]
        expected = case["expected_doc_id"]
        try:
            resp = httpx.post(
                f"{api_base}/retrieve",
                json={"query": query, "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        except Exception as e:
            print(f"[{i}/{total}] ERROR for query {query!r}: {e}")
            reciprocal_ranks.append(0.0)
            continue

        doc_ids = [r["doc_id"] for r in results]
        if expected in doc_ids:
            hits += 1
            rank = doc_ids.index(expected) + 1
            reciprocal_ranks.append(1.0 / rank)
            status = f"HIT  (rank {rank})"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"

        print(f"[{i}/{total}] {status} - {query!r} (expected doc_id={expected!r}, got {doc_ids})")

    hit_rate = hits / total if total else 0.0
    mrr = sum(reciprocal_ranks) / total if total else 0.0

    print("\n--- Summary ---")
    print(f"Hit-rate@{top_k}: {hit_rate:.1%} ({hits}/{total})")
    print(f"MRR@{top_k}:      {mrr:.3f}")
    return hit_rate, mrr


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate /retrieve hit-rate and MRR against a labeled query set."
    )
    parser.add_argument("eval_set", help="Path to a JSON file of {query, expected_doc_id} pairs")
    parser.add_argument("--api-base", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--top-k", type=int, default=5, help="top_k to request from /retrieve")
    args = parser.parse_args()

    with open(args.eval_set) as f:
        eval_set = json.load(f)

    if not eval_set:
        print("eval_set is empty - nothing to evaluate.")
        sys.exit(1)

    evaluate(eval_set, args.api_base, args.top_k)


if __name__ == "__main__":
    main()
