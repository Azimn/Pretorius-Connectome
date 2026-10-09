# Vector Fly Agent Retrieval API v1: verified execution

**Recorded:** October 8, 2026 (America/Chicago). **Status:** complete real-memory HTTP API execution passed in GitHub Actions. This is an engineering retrieval deliverable, not a new cognitive result.

## Verification

[Executed GitHub Actions run 37867892494](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37867892494) passed on branch commit `da1e73b8ded62eaceb922cd7e8fc84557be336ad`.

**Fourteen tests passed:** seven HTTP contract tests and seven SQLite vector-store tests, all against the frozen 450 reconstructed events. Original-v12 source validation passed before indexing. The CI also independently rebuilt the deterministic train-only v2 TF-IDF cache, built and completely verified a fresh 450-memory SQLite database, exercised its command-line retrieval, and checked the server command-line entrypoint.

Tested HTTP behavior:

- `GET /v1/health` and `GET /v1/info` expose service status and scope without creating memories.
- `POST /v1/search` returns exactly the same source IDs and scores as the existing SQLite search for a real source-relevant query. Unknown lexical input returns zero candidates.
- `GET /v1/memories/E01-001` returns the unchanged source record. Missing IDs return 404, not fabricated records.
- Episode and provenance filters stay effective; fictional reconstructed memory provenance is retained.
- A training-only database does not reveal held-out event IDs, including through direct record lookup.
- Invalid JSON parameter types, oversized queries, unexpected fields, write HTTP verbs and malformed paths fail closed.
- Non-loopback exposure without credentials is refused. Incorrect bearer tokens fail; an explicitly supplied adequate token works on the local test server.
- SQLite read-only access was verified from the HTTP service thread after correcting its thread-affinity configuration. It uses a serial HTTPServer, so this is a cross-thread handoff, **not** a claim of safe concurrent writes or scalable multiworker hosting.
- The underlying seven database tests also verify exact sparse retrieval, original source records, tampering rejection, and no-replace publication under competing builds.

[Machine-readable build/verification/query evidence](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37867892494/artifacts/11589141008) remains accessible as a standard GitHub Actions artifact. Source and this human-readable report are committed in Git for durable reference.

## Interoperability boundary

The API offers a stable read-only JSON interface for a separate local agent or Evennia client to call. It does **not** automatically register as a ChatGPT plugin, deploy a process to a server, establish a persistent internet-facing endpoint, change BioCircuit's incompatible 256-dimensional feature space, or update the definitive Pretorius's lived memory.

Source: [HTTP handler](../../src/pretorius_connectome/vector_api.py), [service entrypoint](../../scripts/serve_vector_fly.py), [contract tests](../../tests/test_vector_api.py), [agent runbook](../../docs/VECTOR_FLY_AGENT_API_V1.md), and [workflow](../../.github/workflows/vector-fly-agent-api.yml).

## Next accountable milestone

The next scientific integration is optional fixed-candidate graph reranking: obtain lexical candidates from this database, compare real FlyWire versus equal-capacity rewired topology over the **same event IDs**, preserve per-case source evidence, and report independent evaluation rather than conflating model ranking with factual entailment. A separately versioned semantic embedding index remains another optional line. Neither was constructed or validated in the present API experiment.
