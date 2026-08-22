import os


class Settings:
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "chunks")

    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "testpassword")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    # nomic-embed-text was trained with task prefixes and retrieval quality
    # is measurably worse without them (per Nomic's model card). Set both to
    # "" in .env if you switch to a model that doesn't use this convention.
    EMBEDDING_DOC_PREFIX: str = os.getenv("EMBEDDING_DOC_PREFIX", "search_document: ")
    EMBEDDING_QUERY_PREFIX: str = os.getenv("EMBEDDING_QUERY_PREFIX", "search_query: ")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))       # characters
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120")) # characters

    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "300"))
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "600"))
    # Retry policy for Ollama HTTP calls (embeddings, extraction, chat) -
    # guards ingestion of large files against transient blips (a model
    # still loading, a brief network hiccup) instead of failing an entire
    # job, or silently dropping graph data for one chunk, on the first
    # error. 4xx responses (unknown model, bad request) are never retried
    # - retrying can't fix a config problem, only wastes time.
    OLLAMA_RETRY_ATTEMPTS: int = int(os.getenv("OLLAMA_RETRY_ATTEMPTS", "3"))
    OLLAMA_RETRY_BASE_DELAY: float = float(os.getenv("OLLAMA_RETRY_BASE_DELAY", "1.0"))

    # Model used for LLM-based entity/relationship extraction during
    # ingestion (graph_extraction.py). Defaults to OLLAMA_MODEL if unset -
    # override this to point extraction at a smaller/faster model, since
    # it runs once per chunk and adds real latency to ingestion.
    GRAPH_EXTRACTION_MODEL: str = os.getenv("GRAPH_EXTRACTION_MODEL", "")
    # Caps how many tokens the extraction model can generate per chunk
    # (Ollama's num_predict). The JSON response is small, so this mainly
    # guards against a model rambling past its answer instead of stopping -
    # without a cap that costs real ingestion time with nothing to show
    # for it. Raise it if you see truncated/invalid JSON on chunks with a
    # lot of entities.
    GRAPH_EXTRACTION_MAX_TOKENS: int = int(os.getenv("GRAPH_EXTRACTION_MAX_TOKENS", "1024"))
    # How many chunks within one embedding batch get their entity/
    # relationship extraction requested concurrently, instead of one
    # Ollama round-trip at a time. This is the main lever for large-file
    # ingestion throughput, since per-chunk extraction is the dominant
    # cost - but the real-world speedup depends on Ollama's own
    # OLLAMA_NUM_PARALLEL being raised to match, or requests just queue
    # there anyway.
    GRAPH_EXTRACTION_CONCURRENCY: int = int(os.getenv("GRAPH_EXTRACTION_CONCURRENCY", "3"))

    # --- File parsing (file_parsers.py) ---
    # Legacy .doc -> .txt conversion via LibreOffice scales its timeout
    # with file size instead of a single fixed value, since a large legacy
    # .doc can plausibly take longer to convert than a small one.
    DOC_CONVERT_TIMEOUT_BASE: int = int(os.getenv("DOC_CONVERT_TIMEOUT_BASE", "60"))
    DOC_CONVERT_TIMEOUT_PER_MB: int = int(os.getenv("DOC_CONVERT_TIMEOUT_PER_MB", "2"))
    DOC_CONVERT_TIMEOUT_MAX: int = int(os.getenv("DOC_CONVERT_TIMEOUT_MAX", "600"))
    # Above this file size, JSON is parsed with a streaming parser (ijson)
    # instead of json.load()'ing the whole file into memory at once - only
    # matters for large JSON exports; small files use the simpler in-memory
    # path since streaming has its own per-item overhead.
    JSON_STREAMING_THRESHOLD_MB: int = int(os.getenv("JSON_STREAMING_THRESHOLD_MB", "20"))
    # When true (default), JSON files skip the LLM extraction call
    # entirely and instead derive entities/relationships directly from
    # the JSON's own key names and nesting (see structural_extraction.py)
    # - a key like "customer.company" already tells you the relationship,
    # no LLM inference needed. Faster, free, and deterministic, at the
    # cost of missing anything embedded in free-text JSON string values
    # (a "notes" field with prose-mentioned entities won't be caught this
    # way). Set false to fall back to the same LLM-based extraction used
    # for every other file type.
    JSON_STRUCTURAL_EXTRACTION_ENABLED: bool = os.getenv("JSON_STRUCTURAL_EXTRACTION_ENABLED", "true").lower() == "true"
    # Fraction (0-1) of a JSON record's scalar fields that must match a
    # known key hint for structural_extraction's result to be trusted.
    # Below this, ingest.py falls back to LLM extraction for that specific
    # record instead of accepting a thin/empty structural result - this is
    # the self-correcting safety net for schemas whose field-naming
    # convention doesn't match structural_extraction.py's hint tables, so
    # an unrecognized schema degrades to the slower-but-general LLM path
    # automatically rather than silently losing graph data. 0 disables the
    # fallback entirely (always trust structural extraction once enabled,
    # matching the original behavior); 1.0 means fall back unless every
    # scalar field matched.
    JSON_STRUCTURAL_MIN_COVERAGE: float = float(os.getenv("JSON_STRUCTURAL_MIN_COVERAGE", "0.3"))
    # OCR fallback for scanned/image-only PDF pages (via pytesseract +
    # pdf2image). Only engages for pages where native text extraction
    # returns nothing - never runs OCR on a page that already has real
    # text. No-ops with a one-time warning if the OCR packages/binaries
    # aren't available in this environment.
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_GRAPH_EXPAND: int = int(os.getenv("RAG_GRAPH_EXPAND", "5"))
    # Minimum vector similarity score (0-1, cosine) a chunk must have to be
    # used at all. Default 0 = disabled (current behavior: always return the
    # top-k closest chunks, however weak the match). There's no safe default
    # to pick blindly here — it depends on your embedding model and data.
    # See the README for how to find a good value using /retrieve.
    RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.3"))

    # --- Multi-channel retrieval fusion (retrieval.py) ---
    # Retrieval runs vector search, entity-name lookup, graph expansion, and
    # keyword search as independent channels, then merges them with
    # Reciprocal Rank Fusion (RRF): each channel contributes
    # weight / (RAG_RRF_K + rank_in_that_channel) per chunk, summed across
    # channels. This means a chunk found by multiple channels outranks one
    # found strongly by only one - a much better relevance signal than
    # "top of the vector list" alone. 60 is the standard RRF constant from
    # the literature; there's rarely a reason to change it.
    RAG_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))
    RAG_WEIGHT_VECTOR: float = float(os.getenv("RAG_WEIGHT_VECTOR", "1.0"))
    # Entity-name matches get a higher default weight than vector/graph -
    # the query literally naming something that exists in the graph is
    # about as strong a relevance signal as retrieval gets.
    RAG_WEIGHT_ENTITY: float = float(os.getenv("RAG_WEIGHT_ENTITY", "1.3"))
    RAG_WEIGHT_GRAPH: float = float(os.getenv("RAG_WEIGHT_GRAPH", "0.8"))
    RAG_WEIGHT_KEYWORD: float = float(os.getenv("RAG_WEIGHT_KEYWORD", "0.6"))
    # How many candidates each channel pulls before fusion narrows down to
    # top_k - wider than top_k itself so RRF has real material to combine
    # instead of just re-ordering an already-truncated list.
    RAG_CANDIDATE_MULTIPLIER: int = int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "3"))
    # How many chunks the keyword channel scrolls through before ranking
    # client-side by term overlap (see vector_store.keyword_search). Must
    # stay well above top_k * RAG_CANDIDATE_MULTIPLIER - it's the pool
    # ranking happens over, not the final answer size, so a value too
    # close to the final size means true best matches can fall outside
    # the pool and never get ranked at all.
    RAG_KEYWORD_SCROLL_LIMIT: int = int(os.getenv("RAG_KEYWORD_SCROLL_LIMIT", "500"))
    # Caps how many chunks from the same document can appear in one merged
    # result set, so one long/well-embedded document can't crowd out
    # everything else.
    RAG_MAX_CHUNKS_PER_DOC: int = int(os.getenv("RAG_MAX_CHUNKS_PER_DOC", "3"))
    # Max relationship hops considered when looking for a path between two
    # named entities in the query (see graph_store.find_relationship_paths).
    RAG_PATH_MAX_HOPS: int = int(os.getenv("RAG_PATH_MAX_HOPS", "4"))
    # Entity types excluded from being traversed as an expansion hop's
    # target in expand_from_chunks - comma-separated. DATE is excluded by
    # default: years are deliberately low-cardinality (rounded to just the
    # year - only ~100-200 possible values ever exist), so a year like
    # "2022" ends up directly connected to every chunk that happens to
    # mention that year, regardless of topic - a classic hub entity that
    # would otherwise pollute graph expansion with chunks related to
    # nothing but sharing a calendar year. Add other types here (e.g.
    # LOCATION) if a similarly low-cardinality/high-frequency type in your
    # corpus starts behaving the same way.
    RAG_GRAPH_EXPAND_EXCLUDE_TYPES: str = os.getenv("RAG_GRAPH_EXPAND_EXCLUDE_TYPES", "DATE")
    # Drops a candidate from the merged results if its text overlaps an
    # already-selected chunk's text by at least this fraction of words -
    # catches the common case where two overlapping/adjacent chunks
    # (CHUNK_OVERLAP means they legitimately share text) both surface,
    # wasting context-window budget on largely redundant content.
    RAG_DEDUP_OVERLAP_THRESHOLD: float = float(os.getenv("RAG_DEDUP_OVERLAP_THRESHOLD", "0.6"))
    # Neighbor-window expansion: for the top RAG_NEIGHBOR_EXPAND_TOP_N
    # fused hits, also pulls in the RAG_NEIGHBOR_WINDOW chunks immediately
    # before/after each (same document) - context a chunk boundary may
    # have cut off from a strong hit. Cheap (no extra LLM call, just a
    # Qdrant lookup by doc_id+index) so this is on by default; set
    # RAG_NEIGHBOR_WINDOW=0 to disable.
    RAG_NEIGHBOR_WINDOW: int = int(os.getenv("RAG_NEIGHBOR_WINDOW", "1"))
    RAG_NEIGHBOR_EXPAND_TOP_N: int = int(os.getenv("RAG_NEIGHBOR_EXPAND_TOP_N", "3"))
    # Reranking: after RRF fusion, asks the LLM to score each candidate's
    # relevance to the query directly, then re-sorts by that score before
    # truncating to top_k. Usually improves precision, at the cost of one
    # extra LLM call - and therefore real added latency - per retrieval.
    # Off by default so you opt into the tradeoff deliberately.
    RAG_RERANK_ENABLED: bool = os.getenv("RAG_RERANK_ENABLED", "false").lower() == "true"
    RAG_RERANK_POOL_MULTIPLIER: int = int(os.getenv("RAG_RERANK_POOL_MULTIPLIER", "2"))
    # Model override for retrieval's optional LLM-assist steps (query
    # condensation, query expansion, reranking) - falls back to
    # GRAPH_EXTRACTION_MODEL, then OLLAMA_MODEL, if unset. These are quick
    # auxiliary calls (rewrite a question, score some passages), so a
    # smaller/faster model than your main chat model is often a good fit.
    RAG_ASSIST_MODEL: str = os.getenv("RAG_ASSIST_MODEL", "")
    # Query condensation: for multi-turn chat, rewrites a follow-up like
    # "what about its founder?" into a standalone question using recent
    # history before retrieval - without this, retrieval only ever embeds
    # the follow-up alone, which often carries little meaning on its own.
    # Costs one extra LLM call per turn; off by default.
    RAG_CONDENSE_ENABLED: bool = os.getenv("RAG_CONDENSE_ENABLED", "false").lower() == "true"
    RAG_CONDENSE_HISTORY_TURNS: int = int(os.getenv("RAG_CONDENSE_HISTORY_TURNS", "4"))
    # Query expansion: generates alternate phrasings of the query via the
    # LLM and searches with each too, folding results into the vector
    # channel - improves recall on queries whose specific wording happens
    # to embed weakly. Costs N extra embed+search round-trips; off by
    # default.
    RAG_QUERY_EXPANSION_ENABLED: bool = os.getenv("RAG_QUERY_EXPANSION_ENABLED", "false").lower() == "true"
    RAG_QUERY_EXPANSION_COUNT: int = int(os.getenv("RAG_QUERY_EXPANSION_COUNT", "2"))

    # --- Agentic retrieval (off by default) --------------------------------
    # When enabled, /rag/chat uses agentic_retrieval.agentic_retrieve
    # instead of the single-shot hybrid_retrieve: decomposes complex
    # queries, grades results, and self-corrects with broadened search
    # when retrieval is weak. Adds multiple LLM round-trips per query -
    # designed for strong models (120B+) where those decisions are reliable
    # and the latency is acceptable. The existing /retrieve endpoint always
    # stays single-shot regardless of this toggle.
    RAG_AGENTIC_ENABLED: bool = os.getenv("RAG_AGENTIC_ENABLED", "false").lower() == "true"
    # Max sub-questions the decomposition step can produce.
    RAG_AGENTIC_MAX_SUBQUESTIONS: int = int(os.getenv("RAG_AGENTIC_MAX_SUBQUESTIONS", "4"))
    # How many times to reformulate and retry retrieval per sub-question
    # when grading says the results are too weak.
    RAG_AGENTIC_MAX_RETRIES: int = int(os.getenv("RAG_AGENTIC_MAX_RETRIES", "2"))
    # Minimum relevance score (0-10) a chunk must get from the grading
    # LLM to survive. Below this it's discarded as irrelevant.
    RAG_AGENTIC_RELEVANCE_THRESHOLD: float = float(os.getenv("RAG_AGENTIC_RELEVANCE_THRESHOLD", "5.0"))
    # Fraction of top_k that must survive grading before the agent
    # considers results "good enough" and stops retrying. E.g. 0.4 with
    # top_k=5 means at least 2 relevant chunks must survive.
    RAG_AGENTIC_MIN_RELEVANT_RATIO: float = float(os.getenv("RAG_AGENTIC_MIN_RELEVANT_RATIO", "0.4"))

    RAG_SYSTEM_PROMPT: str = os.getenv(
        "RAG_SYSTEM_PROMPT",
        "You are a helpful assistant. You have been given context retrieved "
        "from a knowledge base, which may or may not be relevant to the "
        "user's question. If the context is relevant, use it and mention "
        "which document a fact came from. If the context is NOT relevant to "
        "the question — including if it's empty or about a different topic "
        "— ignore it completely and answer the question normally using your "
        "own knowledge. Never force an answer to fit irrelevant context.",
    )
    # Appends a "Sources retrieved" footer to /v1/chat/completions answers so
    # you can see what was actually pulled into context from inside a normal
    # chat UI (like OpenWebUI), without needing to call /retrieve separately.
    SHOW_SOURCES_IN_CHAT: bool = os.getenv("SHOW_SOURCES_IN_CHAT", "true").lower() == "true"


settings = Settings()
