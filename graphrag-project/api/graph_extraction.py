import json
import logging
from typing import Dict, Any

from config import settings
from retry import with_retry, is_retryable_ollama_error
from ollama_client import get_sync_client

logger = logging.getLogger("graphrag")

ALLOWED_ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "THREAT_ACTOR", "MALWARE", "VULNERABILITY",
    "INDICATOR", "PRODUCT", "LOCATION", "FACILITY", "MILITARY_UNIT",
    "WEAPON", "OPERATION", "LAW", "EVENT", "DATE", "CONCEPT",
]

EXTRACTION_SYSTEM_PROMPT = f"""You extract a knowledge graph from a passage of text, focused on cybersecurity, threat intelligence, military, and political/geopolitical content.

Return ONLY a JSON object with this exact shape, nothing else:
{{
  "entities": [{{"name": "<canonical name>", "type": "<one of {', '.join(ALLOWED_ENTITY_TYPES)}>"}}],
  "relationships": [{{"source": "<entity name>", "relation": "<SHORT_UPPER_SNAKE_CASE specific verb phrase>", "target": "<entity name>"}}]
}}

Entity type guide:
- PERSON: a named individual (an official, analyst, commander, or an individual threat actor).
- ORGANIZATION: a legitimate company, agency, institution, coalition, or non-state group.
- THREAT_ACTOR: a named attacker group, APT, ransomware gang, or hacktivist collective (e.g. "APT28", "Lazarus Group", "Conti") - use this INSTEAD OF ORGANIZATION for these specifically, even though they're organizations in a loose sense.
- MALWARE: a named malware family, exploit kit, or offensive tool (e.g. "Emotet", "Cobalt Strike", "Stuxnet").
- VULNERABILITY: a named or CVE-identified vulnerability (e.g. "Log4Shell", "CVE-2021-44228", "EternalBlue").
- INDICATOR: a technical indicator of compromise - an IP address, domain, file hash, or URL tied to an incident. Only extract these when the passage ties them to something meaningful (an attack, an actor, a campaign) - don't extract every IP/hash mentioned in passing.
- PRODUCT: a named software, hardware, or commercial product (e.g. "Microsoft Exchange Server", "Cisco ASA") - the thing affected, used, or discussed, as distinct from MALWARE or WEAPON.
- LOCATION: a country, region, or city.
- FACILITY: a named building or fixed installation (e.g. "the Pentagon", a named military base, a data center).
- MILITARY_UNIT: a named military formation (e.g. "101st Airborne Division", "GRU Unit 26165").
- WEAPON: a named weapon system or military hardware (e.g. "Javelin missile", "F-35").
- OPERATION: a named military, intelligence, or cyber operation or campaign (e.g. "Operation Overlord", "Operation Aurora").
- LAW: a named law, treaty, sanction, or policy.
- EVENT: a named incident, conflict, election, summit, or similar occurrence not already covered by OPERATION.
- DATE: see the DATE rule below - this is the one type that's an exception to "skip dates".
- CONCEPT: last resort only - see the CONCEPT rule below.

Rules:
- Only extract entities that are meaningful named things matching one of the types above. Skip plain numbers and percentages, and generic common nouns.
- Use the most complete/canonical form of a name you can infer from the passage (e.g. "Marie Curie" not "she" or "Curie" if the full name appears anywhere in the passage).
- Every entity MUST include a "type" field - never omit it, never leave it blank.
- The "type" field MUST be exactly one of the types listed above - never invent a new type. If nothing in the list truly fits, use CONCEPT rather than making up a new category.
- CONCEPT is a last resort, not a default. Try PERSON, ORGANIZATION, THREAT_ACTOR, MALWARE, VULNERABILITY, LOCATION, or EVENT first - most named things in this domain fit one of those. Only use CONCEPT for a genuinely abstract idea that fits none of the specific types (e.g. "deterrence", "due process"). If you catch yourself using CONCEPT because you're unsure, reconsider whether a more specific type actually fits better first.
- DATE is an exception to "skip dates": extract ONLY the YEAR as a DATE entity, never a full date. "March 15, 2021", "early 2021", and "Q1 2021" all become the single entity "2021" - do not create separate entities for the month or day, and do not create a different DATE entity for every distinct full date that falls in the same year. If a passage mentions a date vaguely enough that you can't determine the year, skip it rather than guessing.
- Every entity referenced in "relationships" must also appear in "entities".
- "relation" must be a SPECIFIC verb phrase naming the actual relationship, e.g. EXPLOITED, ATTRIBUTED_TO, TARGETED, DEPLOYED, COMPROMISED, PATCHED, DISCOVERED_BY, SANCTIONED, ALLIED_WITH, INVADED, DEPLOYED_TO, MEMBER_OF, LOCATED_IN, OCCURRED_IN.
- Prefer a specific verb phrase when the passage supports one - it's more useful than a generic one. But if nothing specific fits, a more general relation (e.g. RELATED_TO, ASSOCIATED_WITH) is still fine to use - don't omit a relationship just because you can't find a precise verb for it.
- If the passage has no meaningful entities or relationships, return {{"entities": [], "relationships": []}}.
- Do not include any text, explanation, or markdown formatting outside the JSON object.

Example:
Passage: "APT28, a Russian state-sponsored threat actor linked to the GRU, exploited the Log4Shell vulnerability (CVE-2021-44228) to target Ukrainian government networks starting in early 2022."
{{
  "entities": [
    {{"name": "APT28", "type": "THREAT_ACTOR"}},
    {{"name": "GRU", "type": "ORGANIZATION"}},
    {{"name": "Log4Shell", "type": "VULNERABILITY"}},
    {{"name": "Ukraine", "type": "LOCATION"}},
    {{"name": "2022", "type": "DATE"}}
  ],
  "relationships": [
    {{"source": "APT28", "relation": "ATTRIBUTED_TO", "target": "GRU"}},
    {{"source": "APT28", "relation": "EXPLOITED", "target": "Log4Shell"}},
    {{"source": "APT28", "relation": "TARGETED", "target": "Ukraine"}},
    {{"source": "APT28", "relation": "ACTIVE_IN", "target": "2022"}}
  ]
}}
"""


def _extraction_model() -> str:
    return settings.GRAPH_EXTRACTION_MODEL or settings.OLLAMA_MODEL


def _parse_extraction(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "graph_extraction: model returned non-JSON output, skipping entities for this chunk: %r",
            raw[:300],
        )
        return {"entities": [], "relationships": []}

    entities = data.get("entities") or []
    relationships = data.get("relationships") or []

    clean_entities = []
    seen_names = set()
    missing_type_count = 0
    invalid_type_count = 0
    model_chose_concept_count = 0
    for e in entities:
        name = str(e.get("name", "")).strip()
        raw_type = str(e.get("type", "")).strip().upper()
        if not raw_type:
            # The model omitted "type" entirely (or sent an empty string) -
            # this is a code-level default, not a classification the model
            # actually made. Tracked separately from an explicit CONCEPT
            # choice below so the two failure modes don't get conflated in
            # the logs - "the model never answers this field" needs a
            # prompt/model fix; "the model always answers CONCEPT" needs a
            # different one.
            missing_type_count += 1
            etype = "CONCEPT"
        elif raw_type not in ALLOWED_ENTITY_TYPES:
            # The model invented a type outside the allowed set (e.g.
            # "DATE", "TIME") despite the prompt listing exactly what's
            # allowed. Previously this passed straight through unchanged -
            # nothing validated it - and an invented type would flow all
            # the way into the graph as a raw, untrusted string. Reset to
            # CONCEPT rather than storing whatever the model made up.
            invalid_type_count += 1
            etype = "CONCEPT"
        else:
            etype = raw_type
            if etype == "CONCEPT":
                model_chose_concept_count += 1
        if not name or len(name) < 2:
            continue
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        clean_entities.append({"name": name, "type": etype})

    if missing_type_count or invalid_type_count or model_chose_concept_count:
        logger.info(
            f"graph_extraction: of {len(entities)} entities - "
            f"{missing_type_count} had no 'type' field, "
            f"{invalid_type_count} used a type outside the allowed set (reset to CONCEPT), "
            f"{model_chose_concept_count} were explicitly classified CONCEPT by the model. "
            f"If this ratio stays high across chunks, check EXTRACTION_SYSTEM_PROMPT compliance "
            f"or try a stronger GRAPH_EXTRACTION_MODEL."
        )

    valid_names = {e["name"].lower() for e in clean_entities}
    clean_relationships = []
    for r in relationships:
        source = str(r.get("source", "")).strip()
        target = str(r.get("target", "")).strip()
        relation = str(r.get("relation", "")).strip().upper().replace(" ", "_")
        if not source or not target or not relation:
            continue
        # Drop relationships that reference an entity we filtered out above,
        # rather than creating a dangling edge to something not in the graph.
        if source.lower() not in valid_names or target.lower() not in valid_names:
            continue
        if source.lower() == target.lower():
            continue
        clean_relationships.append({"source": source, "relation": relation, "target": target})

    return {"entities": clean_entities, "relationships": clean_relationships}


def extract_graph(text: str) -> Dict[str, Any]:
    """
    Uses the configured LLM (via Ollama) to pull entities + relationships out
    of a chunk of text, replacing spaCy NER. Returns
    {"entities": [{"name","type"}], "relationships": [{"source","relation","target"}]}.

    Never raises on a bad/unparseable model response or a failed Ollama
    request - ingestion continues with no graph data for that chunk rather
    than failing the whole job, since vector search alone still works
    without it.
    """
    text = text.strip()
    if not text:
        return {"entities": [], "relationships": []}

    payload = {
        "model": _extraction_model(),
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "format": "json",  # forces Ollama to constrain output to valid JSON
        "options": {
            "temperature": 0,
            # Caps generation length so a model that starts rambling past
            # its JSON answer gets cut off rather than burning ingestion
            # time on tokens nobody needs.
            "num_predict": settings.GRAPH_EXTRACTION_MAX_TOKENS,
        },
    }
    def _do_request():
        client = get_sync_client()
        resp = client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    # Retries transient failures (connection errors, timeouts, 5xx) rather
    # than dropping this chunk's graph data on the first blip - large
    # files mean many chunks, so a single flaky call is likely over a
    # whole ingest job. A 4xx is never retried.
    try:
        data = with_retry(
            _do_request,
            attempts=settings.OLLAMA_RETRY_ATTEMPTS,
            base_delay=settings.OLLAMA_RETRY_BASE_DELAY,
            retryable=is_retryable_ollama_error,
            label="graph_extraction",
        )
        raw = data.get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("graph_extraction: Ollama request failed after retries, skipping entities for this chunk: %s", e)
        return {"entities": [], "relationships": []}

    return _parse_extraction(raw)
