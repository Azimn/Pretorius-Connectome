# Vector Fly persistent SQLite vector database v1: executed results

**Date:** October 8, 2026 (America/Chicago). **Status:** source-pinned database code and actual 450-memory database generated and verified on CI. This is a working lexical search engine, NOT evidence of biological or semantic memory.

**Successful build/test run:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863611161

**Downloadable built SQLite files and machine-readable results:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863611161/artifacts/11587412346

## Measured build outputs

| Database | Scope | Indexed events | Sparse postings | SQLite size | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| vector-fly-450.sqlite | all original events | 450 | 31,159 | 1,986,560 bytes | 87e24687cc74e2fdb241496a5d32a49d01658abcd993bcd762ccc8fa56b462ee |
| vector-fly-train31.sqlite | seed 31 training episodes only | 317 | 24,261 | 1,519,616 bytes | 6afe747c8af0a0c7bdc2dfd4be99f3b224183f43dd7c7649b3af4d5d121403fa |

Both indexes use the canonical source-verified stable-vocabulary L2 v2 encoder, 8,192 dimensions and SQLite inverted postings, not a new token fitter, learned semantic embedding or approximate graph index. The complete browsing mode includes previously held-out source events, so it must **never** be used as a held-out research retrieval condition.

## Actual executed checks

- Six newly added database unit tests passed in GitHub Actions (6/6).
- The complete original 450-event autobiography passed source validation before any database construction.
- The stored complete database was reopened and all **450 original records and 31,159 postings** compared with their canonical source and L2 vectors.
- The restricted research database was reopened and all **317 allowed records and 24,261 postings** compared with pinned source and training membership.
- Queries returned exact source event IDs and scored results matching the source sparse matrix; sorting breaks ties by stable event ID.
- A real example query for 'camphor beetle brass key' returned first-person canonical text about the camphor-smelling university cabinet, dead beetle and misplaced key.
- The out-of-vocabulary query 'zzzzqzxzy unrecoverableword12345' produced **no invented memory candidates**.
- Tampered SQLite feature values and canonical source records are rejected, as are caches fitted under a different declared seed.

**Limitations:** Similarity does not verify that a quotation entails a query. The source is authored fictional reconstruction, not lived experience. No biologically constrained synaptic training, new learning, approximate nearest-neighbor, independent behavioral holdout, semantic embedding or cross-architecture feature compatibility is demonstrated by this database.

## Reproduction and retention

Implementation: [vector_store.py](../../src/pretorius_connectome/vector_store.py), [command-line runner](../../scripts/vector_fly_db.py), [tests](../../tests/test_vector_store.py), [workflow](../../.github/workflows/vector-fly-database.yml), [runbook](../../docs/VECTOR_FLY_DATABASE_V1.md).

The small results report and entire source implementation are permanently committed in Git. The actual SQLite binaries are attached to the cited 90-day GitHub Actions artifact rather than permanently versioned as large generated binaries. They can be regenerated from unchanged pinned inputs with the documented commands. Avoid declaring a production vector service until a durable hosting strategy, API boundary and operator access controls are chosen.

**Next gate:** if desired, provide a read-only HTTP or MUD-agent retrieval interface, then test optional FlyWire reranking over exactly the same candidate IDs. Dense learned semantic vectors should be introduced as a separate versioned feature space, never mislabeled as the lexical v2 cache.

## Final concurrency-safety check

[CI run 37864046037](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37864046037) **passed all seven database tests**, adding a deliberately competing build to verify that the final publication never overwrites a database created concurrently. It also regenerated both persistent databases, verified the complete source/vector index, and passed real and out-of-vocabulary retrieval. [Final built SQLite artifact](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37864046037/artifacts/11587801011). The original six-test benchmark remains archived above as historical evidence; this seven-test run is the final release gate. The atomic no-replace publication uses same-directory hard-link creation and fails safely if the destination already exists.
