"""
Rule-based entity/relationship extraction for structured JSON, used
instead of an LLM call (graph_extraction.extract_graph) for JSON-sourced
chunks. The insight: JSON's own key names and nesting already encode a
lot of what the LLM would otherwise have to infer from prose - a
`customer.company` field IS an affiliation relationship, no inference
needed. Skipping the LLM call for JSON is faster (no round-trip per
chunk), free (no extraction cost), and deterministic - no risk of the
model inventing a type, omitting a field, or picking a lazy default,
which is the whole class of bug this project spent real effort chasing
for LLM-based extraction.

The tradeoff, made explicit rather than hidden: this only sees what the
JSON's *structure* tells it, via a fixed set of key-name hints (see
_KEY_TYPE_HINTS below) - a schema whose field names don't match any of
them (a "perpetrator" field where we only know "attacker", say) won't
have those fields recognized at all, no matter how meaningful they are.
Two related safety nets exist because of this:

1. A "notes"/"description"-style free-text field containing prose-
   embedded entities ("met with the CEO of Acme Corp...") is never
   inspected at all - the LLM path would catch that, this won't. Set
   JSON_STRUCTURAL_EXTRACTION_ENABLED=false to fall back to LLM-based
   extraction for JSON entirely, the same path used for every other file
   type.
2. Per-record, this also returns a `coverage` ratio - the fraction of a
   record's scalar fields that matched *any* hint. ingest.py checks this
   against JSON_STRUCTURAL_MIN_COVERAGE and falls back to the LLM for
   that specific record when coverage is too low, rather than silently
   accepting a thin or empty result. This is what keeps the hint tables
   from needing to anticipate every possible field-naming convention up
   front - an unrecognized schema self-corrects to the LLM path instead
   of quietly losing graph data until someone notices.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

# (key-name substrings to match, guessed entity TYPE) - checked against
# each dict key, lowercased, first match wins. Order matters: more
# specific hints are listed before the generic "name"/"title" fallback,
# so a field literally called "name" only falls back to PERSON when
# nothing more specific (including the *enclosing* key - see _walk) fits.
# Kept in sync with graph_extraction.ALLOWED_ENTITY_TYPES by convention -
# this covers the cybersecurity/threat-intel/military/political domain
# taxonomy, not just the earlier generic one.
_KEY_TYPE_HINTS: List[Tuple[Tuple[str, ...], str]] = [
    (("threat_actor", "threatactor", "adversary", "attacker", "actor", "intrusion_set", "apt_group", "apt"), "THREAT_ACTOR"),
    (("malware", "trojan", "ransomware", "backdoor", "payload_name", "malware_family"), "MALWARE"),
    (("vulnerability", "cve", "exploit", "weakness"), "VULNERABILITY"),
    (
        (
            "indicator", "ioc", "ip_address", "ipaddress", "src_ip", "dst_ip",
            "domain_name", "file_hash", "hash_value", "malicious_url", "c2_server", "c2_domain",
        ),
        "INDICATOR",
    ),
    (("military_unit", "battalion", "brigade", "regiment", "squadron", "task_force"), "MILITARY_UNIT"),
    (("weapon", "missile_system", "missile", "warhead", "aircraft_type"), "WEAPON"),
    (("operation_name", "codename", "campaign_name"), "OPERATION"),
    (("military_base", "air_base", "naval_base", "installation", "data_center", "datacenter", "facility"), "FACILITY"),
    (("company", "organization", "org", "employer", "vendor", "publisher", "manufacturer", "supplier", "industry", "sector"), "ORGANIZATION"),
    (
        (
            "author", "customer", "client", "contact", "employee", "owner", "manager",
            "assignee", "user", "person", "buyer", "seller", "recipient", "sender",
            "requester", "reviewer", "approver",
        ),
        "PERSON",
    ),
    (("city", "country", "address", "location", "state", "region"), "LOCATION"),
    (("product", "item", "sku"), "PRODUCT"),
    (("attack_type", "attacktype", "tactic", "technique", "classification", "attack_method"), "CONCEPT"),
    (("event", "incident"), "EVENT"),
    (("law", "regulation", "policy", "sanction", "treaty"), "LAW"),
    (
        (
            "date", "timestamp", "occurred_at", "discovered_date", "published_date",
            "reported_date", "detected_at", "year",
        ),
        "DATE",
    ),
    (("name", "title"), "PERSON"),
]

_MAX_VALUE_LEN = 100  # a matched field's value longer than this looks like prose, not a name - skip it
_MAX_DEPTH = 6  # bounds recursion cost on deeply nested structures

# Matches a plausible 4-digit year (1900-2099) anywhere in a date-like
# string ("2021-03-15", "March 15, 2021", "Q1 2021", an ISO timestamp,
# ...), without grabbing part of a longer number. Used to round any
# DATE-hinted field down to just the year - see _extract_year.
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _guess_type(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    key_lower = key.lower()
    for substrings, etype in _KEY_TYPE_HINTS:
        if any(s in key_lower for s in substrings):
            return etype
    return None


def _extract_year(value: str) -> Optional[str]:
    match = _YEAR_RE.search(value)
    return match.group(0) if match else None


def _relation_label(key: str) -> str:
    return "HAS_" + re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").upper()


def extract_from_json_value(value: Any) -> Dict[str, Any]:
    """
    Walks a parsed JSON value (dict/list/scalar) and derives entities +
    relationships purely from its structure. Returns the same
    {"entities": [...], "relationships": [...]} shape as
    graph_extraction.extract_graph, so callers (ingest.py) can use either
    interchangeably.

    Rules (deliberately simple - see module docstring for the tradeoff):
    - A dict field whose key looks like an identity field (name, company,
      city, ...) and whose value is a short string becomes an entity,
      typed by whichever hint matched. When the matching field is a
      generic identity field ("name"/"title"), the *enclosing* key's hint
      takes priority if it has one - {"company": {"name": "Acme Corp"}}
      types "Acme Corp" as ORGANIZATION (from "company"), not PERSON
      (from "name").
    - Sibling identity fields on the same object relate to each other
      (e.g. {"buyer": "John", "seller": "Jane"} -> John -HAS_SELLER->
      Jane), using each sibling's own field key.
    - When an entity-bearing object is nested inside another, the outer
      object's entity relates to the inner one too - nesting becomes a
      relationship, labeled HAS_<KEY> and categorized by the same key
      hint table.
    - If an object has no identity field of its own (a plain wrapper,
      e.g. a top-level record whose only fields are nested objects), it
      borrows the first child's entity as its own effective
      representative - so a later sibling key in that same object (e.g.
      "orders" next to "customer") still relates to the entity found via
      an earlier one, instead of each sibling being evaluated in
      isolation.
    """
    entities: List[Dict[str, str]] = []
    relationships: List[Dict[str, str]] = []
    seen_names = set()
    # Tracks how many of this record's scalar (non-dict/list) fields we
    # actually recognized via a key hint vs. how many exist in total -
    # this becomes the `coverage` ratio returned below, which is how
    # ingest.py decides whether to trust this result or fall back to the
    # LLM for this specific record instead (see JSON_STRUCTURAL_MIN_COVERAGE).
    # A record where almost nothing matched isn't "correctly found
    # nothing" - it's a strong signal this schema's field-naming
    # convention just isn't one we've taught the hint tables yet.
    stats = {"fields_seen": 0, "fields_matched": 0}

    def _add_entity(name: str, etype: str):
        key = name.lower()
        if key not in seen_names:
            seen_names.add(key)
            entities.append({"name": name, "type": etype})

    def _walk(v: Any, depth: int, parent_entity: Optional[Dict[str, str]], parent_key: Optional[str]) -> Optional[Dict[str, str]]:
        """Returns the 'effective entity' representing this subtree, for
        the caller to relate siblings/parents to - either a directly
        found identity field, or (if none) one borrowed from a nested
        child, or (if neither) whatever parent_entity was passed in."""
        if depth > _MAX_DEPTH:
            return parent_entity

        if isinstance(v, dict):
            # If this object has a date/year-like sibling field, a bare
            # "name" field is far more likely to be naming an EVENT (an
            # incident that happened on/around that date) than a PERSON -
            # a very common shape for incident/breach/event-log JSON.
            # Checked once per object before resolving any individual
            # field, since it needs to see all of this object's keys
            # first.
            has_date_sibling = any(_guess_type(k) == "DATE" for k in v.keys())

            locals_found: List[Dict[str, str]] = []
            for key, val in v.items():
                if isinstance(val, (dict, list)):
                    continue  # not a hint-matching candidate - handled by recursion below
                stats["fields_seen"] += 1
                field_type = _guess_type(key)
                if field_type is None:
                    continue
                stats["fields_matched"] += 1

                if field_type == "DATE":
                    # Round to the year rather than keeping the full date -
                    # "March 15, 2021", "Q1 2021", and an ISO timestamp
                    # from the same year all become the single entity
                    # "2021", so they merge into one node instead of each
                    # full date creating its own never-repeating entity.
                    # Accepts a bare numeric year too ("year": 1988 as a
                    # JSON int, not a string) - dates in flat incident/
                    # event-style records are often stored as plain
                    # numbers rather than date strings, unlike every other
                    # type here which only ever appears as a string.
                    if isinstance(val, bool):
                        continue
                    elif isinstance(val, (int, float)):
                        val_str = str(int(val))
                    elif isinstance(val, str):
                        val_str = val
                    else:
                        continue
                    year = _extract_year(val_str)
                    if not year:
                        continue
                    _add_entity(year, "DATE")
                    locals_found.append({"name": year, "type": "DATE", "key": key})
                    continue

                if not isinstance(val, str):
                    continue
                val = val.strip()
                if not val or len(val) > _MAX_VALUE_LEN:
                    continue
                is_generic_field = key.lower() in ("name", "title")
                if is_generic_field:
                    enclosing_type = _guess_type(parent_key)
                    etype = enclosing_type or ("EVENT" if has_date_sibling else field_type)
                else:
                    etype = field_type
                _add_entity(val, etype)
                locals_found.append({"name": val, "type": etype, "key": key})

            direct_primary = locals_found[0] if locals_found else None

            # Siblings on the same object relate to the primary one.
            for other in locals_found[1:]:
                relationships.append(
                    {
                        "source": direct_primary["name"],
                        "relation": _relation_label(other["key"]),
                        "target": other["name"],
                    }
                )

            # This object's primary entity relates to the parent's
            # entity too - nesting itself becomes a relationship.
            if direct_primary and parent_entity and parent_key:
                relationships.append(
                    {
                        "source": parent_entity["name"],
                        "relation": _relation_label(parent_key),
                        "target": direct_primary["name"],
                    }
                )

            # effective_primary updates *during* iteration (not computed
            # once up front) so a later sibling key benefits from an
            # entity found via an earlier one, even when this object has
            # no identity field of its own.
            effective_primary = direct_primary or parent_entity
            borrowed_primary = None

            for k, v2 in v.items():
                if not isinstance(v2, (dict, list)):
                    continue
                child_primary = _walk(v2, depth + 1, effective_primary, k)
                if not child_primary or child_primary is effective_primary:
                    continue
                if direct_primary is not None:
                    continue  # already related via the direct-parent link above
                if borrowed_primary is None:
                    borrowed_primary = child_primary
                    effective_primary = child_primary
                elif borrowed_primary is not child_primary and isinstance(v2, dict):
                    # Only for dict children - a list child's items
                    # already got their relationship to effective_primary
                    # directly during their own item-level processing
                    # (each item is walked with parent_entity already set
                    # to effective_primary), so adding it again here
                    # would double-count evidence for no reason.
                    relationships.append(
                        {
                            "source": borrowed_primary["name"],
                            "relation": _relation_label(k),
                            "target": child_primary["name"],
                        }
                    )

            return direct_primary or borrowed_primary or parent_entity

        elif isinstance(v, list):
            first_primary = None
            for item in v:
                item_primary = _walk(item, depth + 1, parent_entity, parent_key)
                if first_primary is None and item_primary and item_primary is not parent_entity:
                    first_primary = item_primary
            return first_primary or parent_entity

        return parent_entity

    _walk(value, 0, None, None)
    coverage = stats["fields_matched"] / stats["fields_seen"] if stats["fields_seen"] else 0.0
    return {"entities": entities, "relationships": relationships, "coverage": coverage}
