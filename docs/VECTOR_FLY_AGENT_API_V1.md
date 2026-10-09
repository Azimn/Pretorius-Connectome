# Vector Fly Agent Retrieval API v1

**Scope:** a working read-only local HTTP/JSON interface to the already-verified 450-memory Vector Fly SQLite index. This is a retrieval boundary for Pretorius, Kiki, MUD agents and local tools; it does not replace their neural architectures or provide an autonomous agent runtime.

## Architecture and data integrity

- Canonical v12 reconstructed autobiography is still the only source of biographical facts. The immutable L1 source and split-fitted TF-IDF L2 v2 stay unchanged.
- At startup, the existing `VectorFlyStore` verifies original source provenance, every indexed record, the L2 source/encoder fingerprints and every SQLite vector posting. If the database or source is altered, the service refuses to start.
- The server performs no memory mutation. HTTP PUT, PATCH and DELETE receive 405. There is no endpoint that edits a biography, trains weights, creates synapses, or leaks validation/test records from a training-only index.
- HTTP is standard-library Python. A single-threaded server shares one verified SQLite connection. No paid AI API, hosted vector database, web access, embedding download or API credential is needed for localhost operation.
- Lexical similarity is **not** semantic entailment. Each result contains the true authored event ID, original source excerpt, source provenance and score. These memories are fictional reconstructions, not lived experiences.

## Launch on Windows, Linux or macOS

From the repository root with Python 3.11+, build a source-pinned cache and database **once** (or reuse the databases from the previous Vector Fly v1 workflow artifact):

```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'scikit-learn>=1.5,<2' 'rank-bm25>=0.2,<1'
python scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/vector-db-seed31
python scripts/vector_fly_db.py build --cache-dir data/derived/vector-db-seed31 --database data/derived/vector-fly-450.sqlite --scope all
python scripts/serve_vector_fly.py --database data/derived/vector-fly-450.sqlite --cache-dir data/derived/vector-db-seed31 --host 127.0.0.1 --port 8765
```

Leave the last command running while your agent needs retrieval. For repeated sessions, skip the earlier build commands and use the existing verified database. Both the database and the fitted L2 cache directory must remain present.

For scientific experiments, build with `--scope train` into a **different** SQLite filename and serve that instead. The API always reports which scope is active; it must not treat all-450 browsing as a valid holdout experiment.

## Stable HTTP/JSON endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/v1/health` | Check service status, read-only flag and active scope |
| GET | `/v1/info` | Full index metadata and scientific limitations |
| POST | `/v1/search` | Exact lexical vector retrieval with optional filters |
| GET | `/v1/memories/{event_id}` | Original complete canonical L1 record, if indexed |

### Search

```sh
curl -sS http://127.0.0.1:8765/v1/search -H 'Content-Type: application/json' -d '{"query":"camphor beetle brass key","top_k":5}'
```

Optional JSON keys are `episode_id` and `provenance`. Unknown parameters are rejected rather than ignored. `top_k` is an integer from 1 through 20; `query` must be nonempty and at most 2,000 characters. The maximum JSON request body is 8,192 bytes.

Example response structure (abbreviated; the actual event results are measured in the original database CI):

```json
{
  "api_version": "vector-fly-retrieval/1",
  "scope": "all",
  "source_git_blob": "718dcc2d5ba4feccdef1690d447edfcebaa9bfb5",
  "vector_kind": "lexical-tfidf-not-semantic",
  "query": "camphor beetle brass key",
  "count": 1,
  "results": [
    {
      "event_id": "E01-001",
      "episode_id": "E01",
      "title": "The Empty Specimen Drawer",
      "provenance": "reconstructed",
      "source_excerpt": "On my third day at the university...",
      "similarity": 0.246430545522,
      "evidence_status": "source text similarity, not claim entailment"
    }
  ],
  "evidence_status": "retrieval only; no factual entailment check"
}
```

The `count: 1` above is a shortened structural illustration, not the full five-result query. For measured source-linked results and exact full excerpts, read [the original database execution](../results/vector_fly/VECTOR_DATABASE_V1_RESULTS.md).

### Original event

```sh
curl -sS http://127.0.0.1:8765/v1/memories/E01-001
```

The response contains the original L1 projected structured memory, plus source status. A nonexistent or held-out ID returns 404, not a hallucinated object.

### Health and metadata

```sh
curl -sS http://127.0.0.1:8765/v1/health
curl -sS http://127.0.0.1:8765/v1/info
```

## Agent integration contract

An AI/MUD agent equipped with a local HTTP client should:

1. Search first for relevant memory cues using `POST /v1/search`.
2. Treat results as candidates, not as proof of a claim. Never claim the top match logically entails a user assertion.
3. Retrieve original records by event ID when more context is necessary.
4. Preserve the source `event_id`, episode, provenance and search scope with any downstream recollection or continuity ledger update.
5. Keep separate the agent's post-instantiation lived memories from the fictional pre-instantiation reconstructed autobiography, and do not write into this read-only source.

This API makes memory **retrievable**; it is not a ChatGPT connector automatically installed in conversations and it cannot itself write to an Evennia character or another agent's database. A consuming program must call its endpoints explicitly.

## Security

- Default listening host is `127.0.0.1`. The implementation does not add permissive browser CORS headers and suppresses access logging to avoid saving source excerpts or queries in logs.
- Binding to a non-loopback host (`--host 0.0.0.0`, for example) **requires a bearer token**. Set a secret at least 24 characters long in an environment variable and pass `--token-env NAME`. Clients send `Authorization: Bearer <token>`.
- Plain HTTP does not encrypt bearer tokens. Do not expose this server to the public Internet or transmit credentials across untrusted networks. Use a secure tunnel or TLS-terminating authenticated proxy outside this simple local server when remote access is necessary.
- Endpoints accept bounded JSON request sizes, reject invalid types, and return JSON errors. No user-supplied strings are interpolated into database SQL; original query parameterization remains in force.

## Validation and next step

Run `python -m unittest discover -s tests -p test_vector_api.py -v`. The tests exercise the real corpus, exact source-linked HTTP searches, filters, record lookup, held-out isolation, no writes, malformed payloads, size limits, wrong tokens and secure binding behavior. The [API workflow](../.github/workflows/vector-fly-agent-api.yml) also runs the original vector-store tests and independently builds/verifies the source-backed SQLite database.

The next worthwhile feature is optional **two-stage retrieval**: use this stable API to identify source candidates, then compare real FlyWire topological reranking against a rewired control over the **same candidate IDs**. A subsequent separately versioned dense semantic vector model may improve paraphrase matching, but requires an independently tested query encoder and model/checksum provenance. Neither is included in this v1 HTTP service.
