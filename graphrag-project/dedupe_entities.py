#!/usr/bin/env python3
"""
Finds likely-duplicate Entity nodes in Neo4j - the same real-world thing
stored under two different name spellings/casings/punctuation - and
reports them for review. Also supports merging a specific, confirmed pair.

Deliberately NOT automatic: string similarity cannot tell "Acme Corp."
(a punctuation variant of "Acme Corp") apart from "Georgia" the US state
vs "Georgia" the country (identical string, genuinely different entities)
or two different people who happen to share a name. Auto-merging on a
threshold would silently corrupt the graph the moment it hit one of those
cases. So: candidate-finding is read-only and safe to run anytime; merging
always requires you to name the specific pair after reviewing it.

Usage:
    # Report candidates only (safe, read-only, default)
    python dedupe_entities.py

    # Only report pairs above a similarity threshold (default 0.85)
    python dedupe_entities.py --threshold 0.9

    # List entities with very little supporting evidence (possible noise -
    # a one-off bad extraction, not necessarily a duplicate of anything)
    python dedupe_entities.py --low-evidence

    # Merge one specific confirmed pair - ABSORB's relationships and
    # evidence move onto KEEP, then ABSORB is deleted. Irreversible.
    python dedupe_entities.py --merge "Acme Corp" "Acme Corp."

Reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD from the environment, same
as the API itself - run this with the same .env loaded, or pass
equivalent values directly via those env vars.
"""
import argparse
import difflib
import os
from collections import defaultdict

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")


def _normalize_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def fetch_entities(driver):
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Entity) RETURN e.key AS key, e.name AS name, e.type AS type, "
            "size(coalesce(e.chunk_ids, [])) AS evidence"
        )
        return [dict(r) for r in result]


def find_candidates(entities, threshold: float):
    """
    Groups entities by type (comparing a PERSON to an ORGANIZATION is
    almost never useful and just multiplies the search space for nothing),
    then does a pairwise string-similarity comparison within each group
    using stdlib difflib - no new dependency, and simple ratio-based
    similarity is enough to catch case/punctuation/typo variants, which is
    what this is actually for. It will NOT catch semantic aliases (APT28
    vs Fancy Bear) - that's a fundamentally different problem needing
    domain knowledge, not string comparison; see the note this script
    prints about that.
    """
    by_type = defaultdict(list)
    for e in entities:
        by_type[e["type"]].append(e)

    candidates = []
    for etype, group in by_type.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["key"] == b["key"]:
                    continue
                ratio = difflib.SequenceMatcher(None, a["key"], b["key"]).ratio()
                if ratio >= threshold:
                    candidates.append((ratio, a, b))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates


def merge_entities(driver, keep_key: str, absorb_key: str):
    """
    Merges `absorb_key` into `keep_key`: unions their chunk_ids/doc_ids,
    redirects every one of absorb's relationships onto keep, then deletes
    absorb. Uses the same "check both directions before creating" pattern
    the main ingestion path uses (see graph_store.py's upsert_batch) -
    apoc.merge.relationship only matches its own fixed direction, so
    redirecting naively could create a second, reversed-direction edge
    instead of reusing one that already exists between keep and that
    neighbor. Weight is always recomputed as the size of the (deduped)
    chunk_ids list after merging, rather than summed - self-consistent,
    no risk of drifting from the true evidence count.
    """
    with driver.session() as session:
        exists = session.run(
            "MATCH (e:Entity {key: $key}) RETURN e.name AS name", key=keep_key
        ).single()
        if not exists:
            raise ValueError(f"No entity found with normalized key '{keep_key}' - check the exact name")
        absorb_exists = session.run(
            "MATCH (e:Entity {key: $key}) RETURN e.name AS name", key=absorb_key
        ).single()
        if not absorb_exists:
            raise ValueError(f"No entity found with normalized key '{absorb_key}' - check the exact name")

        session.run(
            """
            MATCH (keep:Entity {key: $keep_key}), (absorb:Entity {key: $absorb_key})
            SET keep.chunk_ids = apoc.coll.toSet(coalesce(keep.chunk_ids, []) + coalesce(absorb.chunk_ids, [])),
                keep.doc_ids = apoc.coll.toSet(coalesce(keep.doc_ids, []) + coalesce(absorb.doc_ids, []))
            """,
            keep_key=keep_key,
            absorb_key=absorb_key,
        )

        redirects = session.run(
            """
            MATCH (absorb:Entity {key: $absorb_key})-[r]-(other:Entity)
            WHERE other.key <> $absorb_key AND other.key <> $keep_key
            RETURN DISTINCT elementId(r) AS rel_id, type(r) AS rel_type,
                   other.key AS other_key,
                   startNode(r).key = $absorb_key AS was_outgoing,
                   coalesce(r.chunk_ids, []) AS chunk_ids
            """,
            keep_key=keep_key,
            absorb_key=absorb_key,
        )
        redirect_list = [dict(r) for r in redirects]

        for r in redirect_list:
            existing = session.run(
                """
                MATCH (keep:Entity {key: $keep_key})-[rel]-(other:Entity {key: $other_key})
                WHERE type(rel) = $rel_type
                RETURN elementId(rel) AS rel_id
                LIMIT 1
                """,
                keep_key=keep_key,
                other_key=r["other_key"],
                rel_type=r["rel_type"],
            ).single()

            if existing:
                session.run(
                    """
                    MATCH ()-[rel]-() WHERE elementId(rel) = $rel_id
                    SET rel.chunk_ids = apoc.coll.toSet(coalesce(rel.chunk_ids, []) + $chunk_ids),
                        rel.weight = size(apoc.coll.toSet(coalesce(rel.chunk_ids, []) + $chunk_ids))
                    """,
                    rel_id=existing["rel_id"],
                    chunk_ids=r["chunk_ids"],
                )
            else:
                start_key = keep_key if r["was_outgoing"] else r["other_key"]
                end_key = r["other_key"] if r["was_outgoing"] else keep_key
                session.run(
                    """
                    MATCH (a:Entity {key: $start_key}), (b:Entity {key: $end_key})
                    CALL apoc.merge.relationship(a, $rel_type, {}, {}, b, {})
                    YIELD rel
                    SET rel.chunk_ids = $chunk_ids,
                        rel.weight = size($chunk_ids)
                    """,
                    start_key=start_key,
                    end_key=end_key,
                    rel_type=r["rel_type"],
                    chunk_ids=r["chunk_ids"],
                )

        session.run("MATCH (absorb:Entity {key: $absorb_key}) DETACH DELETE absorb", absorb_key=absorb_key)


def report_low_evidence(entities, max_evidence: int = 1):
    thin = [e for e in entities if e["evidence"] <= max_evidence]
    thin.sort(key=lambda e: e["evidence"])
    print(f"{len(thin)} entities with <= {max_evidence} supporting chunk(s) - possibly a one-off extraction, not necessarily wrong:\n")
    for e in thin[:200]:
        print(f"  ({e['type']}) {e['name']!r} - evidence={e['evidence']}")
    if len(thin) > 200:
        print(f"  ... and {len(thin) - 200} more")


def main():
    parser = argparse.ArgumentParser(description="Find/merge likely-duplicate Entity nodes in Neo4j.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity ratio (0-1) to report as a candidate (default 0.85)")
    parser.add_argument("--merge", nargs=2, metavar=("KEEP", "ABSORB"), help="Merge ABSORB into KEEP (exact entity names)")
    parser.add_argument("--low-evidence", action="store_true", help="List entities with very little supporting evidence instead of finding duplicates")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    if args.merge:
        keep_name, absorb_name = args.merge
        keep_key = _normalize_key(keep_name)
        absorb_key = _normalize_key(absorb_name)
        if keep_key == absorb_key:
            print("Those two names normalize to the same entity already - nothing to merge.")
            return
        print(f"Merging {absorb_name!r} into {keep_name!r} ...")
        merge_entities(driver, keep_key, absorb_key)
        print("Done. This is not reversible - spot-check the result in Neo4j Browser if this matters.")
        return

    entities = fetch_entities(driver)
    print(f"Loaded {len(entities)} entities.\n")

    if args.low_evidence:
        report_low_evidence(entities)
        return

    candidates = find_candidates(entities, args.threshold)
    if not candidates:
        print(f"No candidate pairs found above similarity {args.threshold}.")
        return

    print(f"Found {len(candidates)} candidate pair(s) above similarity {args.threshold} - REVIEW before merging, nothing here is automatic:\n")
    for ratio, a, b in candidates:
        print(f"  [{ratio:.2f}] ({a['type']}) {a['name']!r} (evidence={a['evidence']})  <->  {b['name']!r} (evidence={b['evidence']})")

    print(f"\nThis only catches spelling/punctuation/casing variants (string similarity).")
    print(f"It will NOT catch semantic aliases with different spelling entirely (e.g. \"APT28\" vs \"Fancy Bear\")")
    print(f"- that needs a domain-specific alias list, a different approach entirely.\n")
    print(f'To merge a confirmed pair: python dedupe_entities.py --merge "<name to keep>" "<name to absorb>"')


if __name__ == "__main__":
    main()
