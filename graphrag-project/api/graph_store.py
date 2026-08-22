import logging
import re
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase

from config import settings

logger = logging.getLogger("graphrag")

# Kept in sync with graph_extraction.ALLOWED_ENTITY_TYPES by convention,
# not by import (this module has no dependency on graph_extraction) - a
# small, deliberate duplication as a defense-in-depth check. entities
# should already be validated by the time they reach here, but this
# module also uses `type` directly as a Neo4j *label* (see
# _upsert_batch_tx), so an unvalidated value flowing in from some other
# caller shouldn't be able to become an arbitrary label.
_VALID_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "THREAT_ACTOR", "MALWARE", "VULNERABILITY",
    "INDICATOR", "PRODUCT", "LOCATION", "FACILITY", "MILITARY_UNIT",
    "WEAPON", "OPERATION", "LAW", "EVENT", "DATE", "CONCEPT",
}

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            # Bound connection time explicitly — without this, a slow/
            # unreachable Neo4j can leave the driver retrying for a long
            # time with no error, which (during app startup) manifests as
            # the whole API hanging with no response and no visible error.
            connection_timeout=15,
            max_transaction_retry_time=15,
        )
    return _driver


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def _normalize_key(name: str) -> str:
    """Merge key for entities: lowercased, whitespace-collapsed, so
    "Microsoft" / "microsoft" / "Microsoft  " all resolve to one node.
    The first-seen original casing is kept as the display name."""
    return " ".join(name.strip().lower().split())


_constraints_ensured = False


def ensure_constraints():
    """
    Graph holds Document and Entity nodes only - no Chunk nodes. Entity
    text/provenance lives as a chunk_ids list property (pointing back to
    Qdrant, which is the single source of truth for chunk text) rather
    than duplicating chunk content into Neo4j.

    Called once per document during ingestion. Everything inside is
    idempotent (IF NOT EXISTS), but still costs a Neo4j round trip per
    check even when nothing changes. _constraints_ensured caches "already
    done this in this process" - see vector_store.ensure_collection for
    the same pattern and the same reasoning about why a lock isn't needed
    here despite not being strictly race-free.
    """
    global _constraints_ensured
    if _constraints_ensured:
        return
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE"
        )
        # Full-text index for fuzzy entity-name matching in
        # find_entities_in_text - a genuine Lucene-backed index, not a
        # full label scan, so this stays fast regardless of graph size.
        session.run(
            "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]"
        )
    _constraints_ensured = True


def upsert_document(doc_id: str, title: str = "", content_hash: Optional[str] = None):
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "MERGE (d:Document {id: $doc_id}) SET d.title = $title, d.content_hash = $content_hash",
            doc_id=doc_id,
            title=title,
            content_hash=content_hash,
        )


def find_document_by_hash(content_hash: str) -> Optional[str]:
    """Looks up an already-ingested document with the same content hash
    (see ingest.py) - used to warn about likely duplicate ingestion (same
    content under a different doc_id/filename) without blocking it."""
    if not content_hash:
        return None
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (d:Document {content_hash: $content_hash}) RETURN d.id AS doc_id LIMIT 1",
            content_hash=content_hash,
        )
        record = result.single()
        return record["doc_id"] if record else None


def list_documents() -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (d:Document) RETURN d.id AS doc_id, d.title AS title ORDER BY d.title"
        )
        return [{"doc_id": r["doc_id"], "title": r["title"]} for r in result]


def delete_document(doc_id: str, chunk_ids: List[str]) -> Dict[str, int]:
    """
    Removes a document and every piece of graph data solely attributable
    to it:
    - strips this doc's chunk_ids from every Entity's chunk_ids/doc_ids
      list, deleting any Entity left with none remaining (i.e. it only
      ever appeared in this document)
    - strips this doc's chunk_ids from every relationship's evidence
      chunk_ids list, deleting the relationship if no evidence remains
    - deletes the Document node itself

    `chunk_ids` must be the full set of chunk_ids that belonged to this
    document. Neo4j has no authoritative list of "which chunks belong to
    document X" on its own (that lives in Qdrant), so the caller (main.py)
    fetches it from vector_store before calling this.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE size(e.chunk_ids) > 0 AND any(cid IN e.chunk_ids WHERE cid IN $chunk_ids)
            SET e.chunk_ids = [cid IN e.chunk_ids WHERE NOT cid IN $chunk_ids],
                e.doc_ids = [d IN e.doc_ids WHERE d <> $doc_id]
            WITH e
            WHERE size(e.chunk_ids) = 0
            DETACH DELETE e
            RETURN count(e) AS n
            """,
            chunk_ids=chunk_ids,
            doc_id=doc_id,
        )
        record = result.single()
        entities_deleted = record["n"] if record else 0

        # Directed pattern (-[r]->) deliberately, not undirected (-[r]-):
        # an undirected pattern matches every relationship twice (once per
        # traversal direction), which would double-apply this SET/DELETE.
        result = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.chunk_ids IS NOT NULL AND any(cid IN r.chunk_ids WHERE cid IN $chunk_ids)
            SET r.chunk_ids = [cid IN r.chunk_ids WHERE NOT cid IN $chunk_ids]
            WITH r WHERE size(r.chunk_ids) = 0
            DELETE r
            RETURN count(r) AS n
            """,
            chunk_ids=chunk_ids,
        )
        record = result.single()
        relationships_deleted = record["n"] if record else 0

        session.run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=doc_id)

    return {"entities_deleted": entities_deleted, "relationships_deleted": relationships_deleted}


def upsert_entities_and_relationships(
    chunk_id: str,
    doc_id: str,
    entities: List[Dict[str, str]],
    relationships: List[Dict[str, str]],
):
    """Convenience wrapper for a single chunk. ingest.py itself calls
    upsert_batch directly (once per embedding batch, not once per chunk) -
    see that function's docstring for why. This wrapper exists for any
    other/future single-chunk callers."""
    upsert_batch([{"chunk_id": chunk_id, "doc_id": doc_id, "entities": entities, "relationships": relationships}])


def upsert_batch(items: List[Dict[str, Any]]):
    """
    Writes entities + relationships for a whole batch of chunks in ONE
    Neo4j session/transaction, instead of opening a fresh session per
    chunk. ingest.py processes chunks in batches of EMBED_BATCH_SIZE
    (default 32) - previously each chunk's graph write opened its own
    session, meaning ~32 separate session round trips per batch (worse
    once the relationship direction-fix below is factored in, since a
    relationship write can now take 2 queries instead of 1). One shared
    session/transaction does the same work with far fewer network round
    trips to Neo4j.

    items: [{"chunk_id": str, "doc_id": str, "entities": [...], "relationships": [...]}]

    entities: [{"name": str, "type": str}] from graph_extraction.extract_graph
    relationships: [{"source": str, "relation": str, "target": str}]

    Entities are merged on a normalized key so casing/whitespace variants
    collapse into one node. chunk_ids/doc_ids are list properties on the
    Entity itself (no separate Chunk node) so retrieval can map an entity
    back to the Qdrant point(s) it came from and the Document(s) it appears
    in.

    An entity's `type` is only ever *upgraded*, never overwritten outright:
    if it's currently the CONCEPT fallback and a later chunk's extraction
    classifies it more specifically, that specific type wins; once it has
    a non-CONCEPT type, later extractions never flip it back to CONCEPT or
    thrash between two different specific types.

    Relationships are matched in EITHER direction before creating a new
    one: apoc.merge.relationship's MERGE is directed (a)-[:TYPE]->(b), so
    if the LLM extracts the same conceptual fact with source/target
    reversed in a different chunk (common for direction-agnostic relations
    like MARRIED_TO, but also just extraction noise), a purely directed
    MERGE would silently create a second, reversed-direction edge instead
    of reinforcing the existing one - fragmenting one real relationship
    into two weaker ones and undermining expand_from_chunks' weighted
    ranking. Checking both directions first (one extra read per
    relationship) avoids that. This check-then-act isn't atomic across
    concurrent ingestion of different documents mentioning the same
    reversed relationship in the same narrow window -
    a known, narrow residual race, not fully closed here.
    """
    if not items:
        return
    driver = get_driver()
    with driver.session() as session:
        session.execute_write(_upsert_batch_tx, items)


def _upsert_batch_tx(tx, items: List[Dict[str, Any]]):
    for item in items:
        chunk_id = item["chunk_id"]
        doc_id = item["doc_id"]

        for ent in item.get("entities") or []:
            key = _normalize_key(ent["name"])
            if not key:
                continue
            entity_type = ent.get("type", "CONCEPT")
            if entity_type not in _VALID_ENTITY_TYPES:
                entity_type = "CONCEPT"
            result = tx.run(
                """
                MERGE (e:Entity {key: $key})
                ON CREATE SET e.name = $name, e.type = $type, e.chunk_ids = [$chunk_id], e.doc_ids = [$doc_id]
                ON MATCH SET
                    e.type = CASE WHEN e.type = 'CONCEPT' AND $type <> 'CONCEPT' THEN $type ELSE e.type END,
                    e.chunk_ids = CASE WHEN NOT $chunk_id IN e.chunk_ids THEN e.chunk_ids + $chunk_id ELSE e.chunk_ids END,
                    e.doc_ids = CASE WHEN NOT $doc_id IN e.doc_ids THEN e.doc_ids + $doc_id ELSE e.doc_ids END
                WITH e
                MATCH (d:Document {id: $doc_id})
                MERGE (e)-[:APPEARS_IN]->(d)
                RETURN e.type AS current_type, elementId(e) AS eid
                """,
                key=key,
                name=ent["name"],
                type=entity_type,
                chunk_id=chunk_id,
                doc_id=doc_id,
            ).single()

            if result:
                # Keeps the node's Neo4j *label* in sync with its `type`
                # property. Neo4j Browser (and Bloom) color/group nodes by
                # label, not by property value - every Entity node sharing
                # just the base :Entity label renders identically
                # regardless of what `type` says. apoc.create.setLabels
                # replaces ALL labels on the node in one call, which both
                # adds the current type label and strips any stale one
                # left over from a CONCEPT -> specific-type upgrade,
                # without needing to separately track whether type
                # actually changed this time.
                tx.run(
                    """
                    MATCH (e) WHERE elementId(e) = $eid
                    CALL apoc.create.setLabels(e, ['Entity', $type]) YIELD node
                    RETURN node
                    """,
                    eid=result["eid"],
                    type=result["current_type"],
                )

        for rel in item.get("relationships") or []:
            source_key = _normalize_key(rel["source"])
            target_key = _normalize_key(rel["target"])
            if not source_key or not target_key or source_key == target_key:
                continue
            relation = rel["relation"]

            # Look for an existing relationship of this type in EITHER
            # direction before creating a new one - see upsert_batch's
            # docstring for why this matters.
            existing = tx.run(
                """
                MATCH (a:Entity {key: $source_key})-[r]-(b:Entity {key: $target_key})
                WHERE type(r) = $relation
                RETURN elementId(r) AS rel_id
                LIMIT 1
                """,
                source_key=source_key,
                target_key=target_key,
                relation=relation,
            ).single()

            if existing:
                tx.run(
                    """
                    MATCH ()-[r]-() WHERE elementId(r) = $rel_id
                    SET r.weight = coalesce(r.weight, 0) + 1,
                        r.chunk_ids = CASE
                            WHEN r.chunk_ids IS NULL THEN [$chunk_id]
                            WHEN NOT $chunk_id IN r.chunk_ids THEN r.chunk_ids + $chunk_id
                            ELSE r.chunk_ids
                        END
                    """,
                    rel_id=existing["rel_id"],
                    chunk_id=chunk_id,
                )
            else:
                # The Neo4j relationship *type* can't be parameterized in
                # plain Cypher, so apoc.merge.relationship creates the edge
                # typed as the LLM-extracted relation (WORKS_AT, FOUNDED,
                # ...) directly - no generic wrapper type, no
                # relation-as-property.
                #
                # identProps/props/onMatchProps are left empty
                # deliberately: APOC's merge procedures take those maps as
                # literal values, not Cypher expressions, so passing a
                # string like 'coalesce(weight,0)+1' would set the
                # property to that literal string rather than evaluate it.
                # Instead apoc.merge.relationship only creates the edge,
                # and the initial weight/chunk_ids are set as an ordinary
                # SET afterward, where expressions actually evaluate.
                tx.run(
                    """
                    MATCH (a:Entity {key: $source_key}), (b:Entity {key: $target_key})
                    CALL apoc.merge.relationship(a, $relation, {}, {}, b, {})
                    YIELD rel
                    SET rel.weight = 1, rel.chunk_ids = [$chunk_id]
                    """,
                    source_key=source_key,
                    target_key=target_key,
                    relation=relation,
                    chunk_id=chunk_id,
                )


def expand_from_chunks(chunk_ids: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Graph expansion without Chunk nodes: find entities present in the seed
    chunks, walk one hop across *any* relationship type to neighboring
    entities, then surface *other* chunk_ids those neighbors appear in.

    Ranked by the sum of connecting relationships' `weight` (how many
    times that specific relationship was seen across the corpus) rather
    than a raw count of connecting entities - a chunk reached through a
    well-established, repeatedly-evidenced relationship should outrank
    one reached through a single one-off extraction.

    Entities typed as one of RAG_GRAPH_EXPAND_EXCLUDE_TYPES (DATE by
    default) are excluded from being used as an expansion hop's target -
    without this, a low-cardinality/high-frequency type like a
    year-rounded DATE becomes a hub that connects otherwise-unrelated
    chunks just because they happen to share it, drowning out genuinely
    related results.

    The relationship type is left unspecified in the MATCH ([]  not
    [:SOME_TYPE]) since edges are typed per the LLM's extracted relation
    (WORKS_AT, FOUNDED, ...) rather than a single fixed type. The
    `other:Entity` label constraint alone is enough to exclude Document
    nodes reached via APPEARS_IN, since Document isn't labeled Entity.

    Returns chunk_id + score only - Neo4j no longer stores chunk text, so
    callers (retrieval.py) fetch the actual text for these chunk_ids from
    Qdrant.
    """
    if not chunk_ids:
        return []

    exclude_types = [t.strip().upper() for t in settings.RAG_GRAPH_EXPAND_EXCLUDE_TYPES.split(",") if t.strip()]

    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (seed:Entity)
            WHERE any(cid IN seed.chunk_ids WHERE cid IN $chunk_ids)
            WITH collect(DISTINCT seed) AS seeds
            UNWIND seeds AS s
            MATCH (s)-[r]-(other:Entity)
            WHERE NOT other IN seeds AND NOT other.type IN $exclude_types
            WITH other, s, coalesce(r.weight, 1) AS rel_weight
            UNWIND other.chunk_ids AS cid
            WITH cid, s, rel_weight
            WHERE NOT cid IN $chunk_ids
            RETURN cid AS chunk_id,
                   sum(rel_weight) AS score,
                   collect(DISTINCT s.name)[0..5] AS shared_entities
            ORDER BY score DESC
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            exclude_types=exclude_types,
            limit=limit,
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "shared_entities": r["shared_entities"],
                "score": r["score"],
                "source": "graph",
            }
            for r in result
        ]


def entities_for_chunks(chunk_ids: List[str]) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE any(cid IN e.chunk_ids WHERE cid IN $chunk_ids)
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            chunk_ids=chunk_ids,
        )
        return [{"name": r["name"], "type": r["type"]} for r in result]


_LUCENE_SPECIAL_CHARS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _fuzzy_distance_for(word: str) -> int:
    """
    Shorter words get less fuzzy tolerance. An edit distance of 2 on a
    4-character token - a year like "2022", now a common entity name
    since DATE entities are rounded to just the year - lets up to half
    its characters differ, meaning a search for "2022" would also match
    "2020"/"2023"/etc. Edit distance 2 is proportionally a much smaller
    change on a longer word, where it stays useful for catching genuine
    typos without over-matching.
    """
    return 1 if len(word) <= 4 else 2


def _build_fuzzy_query(text: str, min_word_len: int = 3, max_words: int = 12) -> str:
    """
    Builds a Lucene full-text query from free text for find_entities_in_text:
    strips Lucene special characters (which would otherwise break or be
    misread by the query syntax - the same sanitization LangChain's Neo4j
    GraphRAG reference implementation does), tokenizes into words, and
    appends fuzzy tolerance to each - so a misspelled or near-miss mention
    in the query ("Log4Shel" instead of "Log4Shell") still matches instead
    of missing entirely. Tolerance scales down for short words - see
    _fuzzy_distance_for.

    Terms are OR'd together (Lucene's default) rather than AND'd: this
    runs against the user's whole query text, which usually contains much
    more than just entity names, so requiring every word to match would
    almost never fire. Lucene's own relevance scoring naturally ranks an
    entity higher the more of these fuzzy terms it actually matches.
    """
    sanitized = _LUCENE_SPECIAL_CHARS.sub(" ", text)
    words = [w for w in sanitized.split() if len(w) >= min_word_len]
    return " ".join(f"{w}~{_fuzzy_distance_for(w)}" for w in words[:max_words])


def find_entities_in_text(text: str, min_name_length: int = 3, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Finds Entity nodes likely named in `text`. This gives retrieval a
    direct path for "tell me about <specific name>" queries: look the
    entity up in the graph by name rather than relying on vector search to
    happen to rank a chunk mentioning it highly enough to surface on its
    own.

    Uses the entity_name_fulltext index (created in ensure_constraints)
    with fuzzy tolerance per word - a genuine Lucene-backed index lookup,
    not a full label scan, so this stays fast regardless of how large the
    entity graph is. The fuzzy tolerance is the same technique LangChain's
    Neo4j GraphRAG reference implementation uses, and closes a real gap an
    earlier exact-match version had: a typo or near-miss spelling of an
    entity name in the query would previously miss entirely. The tradeoff
    is the flip side of the same coin - fuzzy matching can occasionally
    surface a loosely-related entity that happens to share a similarly-
    spelled word. Ranking by Lucene's own relevance score (rather than
    just entity name length, as the exact-match version did) and the RRF
    fusion weighting in retrieval.py (RAG_WEIGHT_ENTITY) are what keep
    that in check.
    """
    if not text or not text.strip():
        return []
    fuzzy_query = _build_fuzzy_query(text)
    if not fuzzy_query:
        return []
    driver = get_driver()
    with driver.session() as session:
        try:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('entity_name_fulltext', $fuzzy_query)
                YIELD node, score
                WHERE size(node.name) >= $min_len
                RETURN node.name AS name, node.type AS type, node.chunk_ids AS chunk_ids
                ORDER BY score DESC
                LIMIT $limit
                """,
                fuzzy_query=fuzzy_query,
                min_len=min_name_length,
                limit=limit,
            )
            return [
                {"name": r["name"], "type": r["type"], "chunk_ids": r["chunk_ids"] or []}
                for r in result
            ]
        except Exception as e:
            logger.warning(f"graph_store.find_entities_in_text: fulltext query failed, returning no entity matches: {e}")
            return []


def find_relationship_paths(
    named_entities: List[Dict[str, Any]], max_hops: int = 4, max_pairs: int = 3
) -> List[Dict[str, Any]]:
    """
    For queries that name two or more known entities (e.g. "how are X and
    Y related?"), finds the shortest relationship path between them
    directly in the graph and returns it as a human-readable description -
    answering from graph structure directly rather than hoping some chunk
    happens to state the connection in prose.

    Only the first `max_pairs` consecutive pairs from `named_entities` are
    checked (paired as found[0]-found[1], found[1]-found[2], ...) to keep
    this bounded - a query naming many entities would otherwise mean a
    combinatorial number of path queries.

    Relationship type isn't specified in the path pattern ([*..N], not
    [:SOME_TYPE*..N]) since edges are typed per the LLM's extracted
    relation (WORKS_AT, FOUNDED, ...), not a single fixed type.
    """
    if len(named_entities) < 2:
        return []

    driver = get_driver()
    paths = []
    with driver.session() as session:
        pairs = list(zip(named_entities, named_entities[1:]))[:max_pairs]
        for a, b in pairs:
            key_a = _normalize_key(a["name"])
            key_b = _normalize_key(b["name"])
            if not key_a or not key_b or key_a == key_b:
                continue
            # max_hops controls Cypher's variable-length pattern bound
            # ([*..N]), which must be a literal, not a query parameter -
            # safe to interpolate here since it's an int from settings,
            # never user input.
            result = session.run(
                f"""
                MATCH (a:Entity {{key: $key_a}}), (b:Entity {{key: $key_b}})
                MATCH p = shortestPath((a)-[*..{int(max_hops)}]-(b))
                RETURN [n IN nodes(p) | n.name] AS names,
                       [r IN relationships(p) | type(r)] AS rel_types
                LIMIT 1
                """,
                key_a=key_a,
                key_b=key_b,
            )
            record = result.single()
            if not record or not record["names"]:
                continue
            names = record["names"]
            rel_types = record["rel_types"]
            segments = [names[0]]
            for i, rel in enumerate(rel_types):
                segments.append(f"-[{rel}]-> {names[i + 1]}")
            paths.append(
                {
                    "entities": [a["name"], b["name"]],
                    "description": " ".join(segments),
                }
            )
    return paths
