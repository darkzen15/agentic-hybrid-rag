# README2.md — Offline Deployment & Maintenance Guide

This is a standalone companion to the main README, focused entirely on
getting this stack (Qdrant + Neo4j + Ollama + OpenWebUI + the API) onto a
machine with **no internet access**, and on what you can and can't still do
once you're there. Take this file with you — it doesn't assume you have
the rest of the conversation/context this was written from.

---

## 1. What has to cross the air gap

| Item | Why | How it travels |
|---|---|---|
| 5 container images (Qdrant, Neo4j+APOC, Ollama, OpenWebUI, your custom `api` image) | Nothing runs without them | `docker save` / `docker load` |
| Ollama model data (`llama3.1:8b`, `nomic-embed-text`, etc.) | Models are multi-GB downloads | Copy the named Docker volume |
| Neo4j's APOC plugin | Normally downloads itself at container **startup**, not build time | Bake it into the image first (see §2) |
| Project files (`docker-compose.yml`, `.env`, `api/` source) | The actual deployment config | Plain file copy |
| Docker itself | If the offline machine doesn't have it | Offline installer, prepared separately |

**Architecture must match.** A `docker save`'d image built for x86_64 will
not run on an ARM/Apple Silicon machine, or vice versa. If the connected
build machine and the offline target machine differ, build with
`docker buildx build --platform linux/amd64 ...` (or whichever the target
actually is) instead of a plain `docker compose build`.

**Expect 8–15+ GB total.** `llama3.1:8b` alone is ~4.9GB, and the `api`
image now bundles Tesseract + Poppler + LibreOffice on top of the Python
stack. Use a fast external SSD or USB 3.0+ drive — a slow USB 2.0 stick
will make this genuinely painful.

---

## 2. The sharp edge: Neo4j's APOC plugin needs internet at *startup*

`NEO4J_PLUGINS=["apoc"]` in `docker-compose.yml` doesn't bake the plugin
into the image. It makes Neo4j's entrypoint script **download** the APOC
jar from GitHub the first time the container starts. On an offline
machine that download silently fails (or hangs), and the graph half of
this stack breaks in a way that looks like everything else — the same
class of confusing failure this project hit earlier in its life.

**Fix: bake APOC in permanently with `docker commit`, while you still have
internet:**

```bash
# On the connected machine, with the stack already working normally:
docker compose up -d neo4j
# wait until it's healthy (docker compose ps), then:
docker commit graphrag-neo4j graphrag-neo4j-with-apoc:local
```

Then change the `neo4j` service's `image:` line in `docker-compose.yml`:

```yaml
neo4j:
  image: graphrag-neo4j-with-apoc:local   # was: neo4j:5-community
```

You can leave `NEO4J_PLUGINS=["apoc"]` set — with the jar already present
on disk, Neo4j's initializer should find it and skip the download — but
the `docker commit` is what actually *guarantees* this works with zero
internet as a fallback, rather than hoping the skip-if-present behavior
holds across versions.

---

## 3. Fix the Ollama volume for portability (one-time)

The Ollama service should use a **named Docker volume**, not a bind mount
to a specific user's folder — a bind mount to e.g. `C:\Users\JJ\.ollama`
won't exist on a different machine at all. Confirm this in your compose
file:

```yaml
ollama:
  volumes:
    - ollama_data:/root/.ollama   # NOT a hardcoded host path
```

`ollama_data` should already be declared under the top-level `volumes:`
section. If your `ollama` service still points at a raw filesystem path,
switch it to this before doing anything else — everything downstream
(volume export/import) depends on it being a proper named volume.

---

## 4. On the connected machine: build and gather everything

```bash
# Build your custom API image
docker compose build api

# Pull the base images
docker compose pull qdrant ollama openwebui
docker pull neo4j:5-community   # then commit APOC into it, per §2

# Pull your models into Ollama
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
# + anything set as GRAPH_EXTRACTION_MODEL / RAG_ASSIST_MODEL in your .env
```

---

## 5. Package everything for transfer

**Images** — `docker save` accepts multiple images and bundles them into
one tar file, which is much less fiddly than juggling five separate files:

```bash
docker save -o graphrag-images.tar \
  graphrag-api \
  qdrant/qdrant:latest \
  graphrag-neo4j-with-apoc:local \
  ollama/ollama:latest \
  ghcr.io/open-webui/open-webui:main
```

**Ollama model data** — copy the named volume out via a disposable helper
container (works regardless of platform/OS, and doesn't require knowing
where Docker actually stores volumes on disk):

```bash
# Confirm the real volume name first - compose prefixes it with the
# project (directory) name, e.g. "graphrag_ollama_data":
docker volume ls

docker run --rm -v graphrag_ollama_data:/data -v "$(pwd)":/backup alpine \
  tar czf /backup/ollama_data.tar.gz -C /data .
```

**Project files** — copy the directory as-is: `docker-compose.yml`,
`.env`, `api/` (source + `Dockerfile` + `requirements.txt`), and
`eval_retrieval.py` if you use it.

Copy `graphrag-images.tar`, `ollama_data.tar.gz`, and the project
directory onto your transfer drive.

---

## 6. On the offline machine: restore and verify

```bash
# Load the images
docker load -i graphrag-images.tar

# Restore the Ollama volume (create it first if compose hasn't run yet)
docker volume create graphrag_ollama_data
docker run --rm -v graphrag_ollama_data:/data -v "$(pwd)":/backup alpine \
  tar xzf /backup/ollama_data.tar.gz -C /data

# Bring the stack up - no builds, no pulls, everything is already local
docker compose up -d
```

**Don't assume it worked — check.** This is exactly what
`/health/detailed` exists for:

```bash
curl http://localhost:8000/health/detailed | python3 -m json.tool
```

This confirms Qdrant/Neo4j/Ollama are all reachable *and* that your model
tags survived the trip intact (a `"missing_models"` entry here means a
tag mismatch — compare against `docker compose exec ollama ollama list`).
Catching that here beats rediscovering it later as a mysterious 404.

---

## 7. Can you still edit the Python code once you're offline?

**Yes — editing and rebuilding both work fully offline**, as long as you
aren't adding new dependencies. Here's why, and where the limit is.

### Why plain edits + rebuild just work

Look at the layer order in `api/Dockerfile`:

```dockerfile
RUN apt-get update && apt-get install -y ...       # expensive, needs internet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # expensive, needs internet
COPY . .                                             # cheap, pure local copy
```

Docker caches each layer and only re-runs a layer (and everything after
it) if that instruction's inputs changed. The image you `docker load`ed
already has the finished `apt-get`/`pip install` layers sitting in local
storage. If you edit a `.py` file and rebuild:

```bash
docker compose build api
docker compose up -d api
```

Docker sees `requirements.txt` is byte-identical to what produced the
cached layer, reuses the `apt-get`/`pip install` layers untouched, and
only re-runs `COPY . .` — pure local disk I/O, no network call. This is
the same caching mechanism that makes "edit code, rebuild" fast in normal
development; it works identically with zero internet.

### Even faster: skip rebuilding with a bind mount + live reload

If you'll be iterating on the code a lot while offline, avoid the rebuild
step entirely. Add this to the `api` service (a separate
`docker-compose.override.yml` keeps your main file clean):

```yaml
api:
  volumes:
    - api_uploads:/app/uploads
    - ./api:/app          # bind-mount source over the baked-in copy
  command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

With `--reload`, uvicorn watches the filesystem and restarts the app the
moment you save a `.py` file. No build command, no restart command —
just edit and save.

### Where offline editing stops working

- **Adding a new line to `requirements.txt`** invalidates the `pip
  install` layer, so that rebuild needs internet — unless you pre-stage a
  local wheelhouse ahead of time: on the connected machine,
  `pip download -r requirements.txt -d wheelhouse/`, transfer the
  `wheelhouse/` folder over, then on the offline machine change the
  Dockerfile's install line to
  `pip install --no-index --find-links=wheelhouse -r requirements.txt`.
- **Adding a new `apt-get install` package** to the Dockerfile hits the
  same wall, and is more annoying to work around (manually downloading
  `.deb` files and their dependency trees, or standing up a local apt
  mirror) — only worth doing if you expect to need new system packages
  regularly.

So: code edits, prompt/config tweaks, Cypher query fixes — all fine
offline, indefinitely. It's specifically *new dependencies* that need
planning ahead, in either of the two ways above.

---

## 8. Quick pre-flight checklist

Before you disconnect from the internet for good, confirm:

- [ ] `docker compose build api` succeeded with no errors
- [ ] `graphrag-neo4j-with-apoc:local` exists (`docker images`) and is
      referenced in `docker-compose.yml`
- [ ] `ollama` service uses the named `ollama_data` volume, not a bind
      mount to a specific host path
- [ ] All required models are pulled (`docker compose exec ollama ollama
      list` shows the exact tags your `.env` references)
- [ ] `docker save` bundle + Ollama volume tarball + project directory
      are all copied to your transfer drive
- [ ] You've test-loaded the images and run `GET /health/detailed`
      **before** disconnecting, ideally on a machine that mimics the
      offline target, so any surprise happens while you can still fix it
      with internet on hand
