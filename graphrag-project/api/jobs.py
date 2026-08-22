import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from config import settings

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

# Persisted alongside uploads (on the api_uploads volume) so job history
# survives container restarts/recreation, not just in-process reloads.
_JOBS_FILE = os.path.join(settings.UPLOAD_DIR, ".jobs.json")

# Caps how many finished (done/error) job records are kept, so this file
# and the in-memory dict don't grow forever over the life of a
# long-running deployment - every ingest creates a job, and every batch
# within it calls update_job. Without a cap, both the disk write cost per
# update AND the in-memory footprint scale with *all* job history ever
# accumulated, not just the current job. Jobs still queued/processing are
# never pruned regardless of how many there are.
_MAX_JOB_HISTORY = 500


def _load() -> None:
    global _jobs
    if os.path.exists(_JOBS_FILE):
        try:
            with open(_JOBS_FILE, "r") as f:
                _jobs = json.load(f)
        except Exception:
            _jobs = {}


def _save() -> None:
    # Best-effort: if this fails, in-memory state for the current process
    # is still correct, it just won't survive a restart.
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        tmp_path = _JOBS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(_jobs, f)
        os.replace(tmp_path, _JOBS_FILE)
    except Exception:
        pass


def _prune_if_needed() -> None:
    """Call with _lock already held. Evicts the oldest finished jobs once
    total job count exceeds _MAX_JOB_HISTORY - never evicts a job that's
    still queued/processing, even if that means staying over the cap
    temporarily."""
    if len(_jobs) <= _MAX_JOB_HISTORY:
        return
    finished = sorted(
        (j for j in _jobs.values() if j.get("status") in ("done", "error")),
        key=lambda j: j.get("created_at", 0),
    )
    overflow = len(_jobs) - _MAX_JOB_HISTORY
    for j in finished[:overflow]:
        _jobs.pop(j["job_id"], None)


_load()


def create_job(filename: str, doc_id: str) -> str:
    job_id = str(uuid.uuid4())
    now = time.time()
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "doc_id": doc_id,
            "status": "queued",  # queued -> processing -> done | error | cancelled
            "chunks_ingested": None,
            "total_chunks": None,
            "chunks_processed": None,
            "duplicate_of": None,
            "cancel_requested": False,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        _prune_if_needed()
        _save()
    return job_id


def update_job(job_id: str, persist: bool = True, **fields) -> None:
    """
    `persist` controls whether this update is written to disk immediately
    (default) or just applied in-memory. ingest.py passes persist=False
    for pure progress updates (chunks_processed, once per batch - which
    can be hundreds of times for a large file) and lets status changes
    (queued -> processing -> done/error) always persist immediately, since
    those are the updates that matter for surviving a restart. In-memory
    state (what GET /jobs/{job_id} actually reads) is always updated
    either way - only the disk write is throttled, so a crash can only
    ever lose some progress-bar precision, never a status transition.
    """
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _jobs[job_id]["updated_at"] = time.time()
            if persist:
                _save()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def request_cancel(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Marks a queued/processing job for cancellation. This doesn't stop
    anything itself - it's cooperative: ingest.py's batch loop checks
    is_cancel_requested() between batches and halts there once it notices,
    since there's no way to forcibly interrupt work already in flight (an
    Ollama call already underway can't be aborted mid-request - the
    current batch always finishes normally before ingestion stops).

    Returns the updated job record, or None if the job doesn't exist or
    has already finished (nothing left to cancel).
    """
    with _lock:
        job = _jobs.get(job_id)
        if not job or job.get("status") not in ("queued", "processing"):
            return None
        job["cancel_requested"] = True
        job["updated_at"] = time.time()
        _save()
        return dict(job)


def is_cancel_requested(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        return bool(job and job.get("cancel_requested"))


def list_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    with _lock:
        items = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
        return [dict(j) for j in items[:limit]]

