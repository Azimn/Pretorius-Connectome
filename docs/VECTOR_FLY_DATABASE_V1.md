# Vector Fly v1: persistent autobiographical vector database

Status: executable SQLite exact sparse vector database over the canonical 450 reconstructed Pretorius memories. **Vector type is deterministic word/bigram TF-IDF v2, not learned semantic embeddings or a neural fly memory.**

## What this adds

This component reuses the canonical source-pinned L1 autobiography and the existing split-fitted v2 L2 feature cache instead of refitting or rewriting them. The new SQLite file stores original event IDs, episode IDs, provenance and full source-linked structured records plus indexed (feature coordinate, event, TF-IDF weight) postings.

Search encodes incoming text with the original verified train-fitted encoder, calculates exact sparse cosine/dot-product scores from the indexed postings, and ranks by score followed by stable event ID. It supports episode and provenance filters and retrieval of the full original L1 record by event ID. Unknown vocabulary returns an empty result; source excerpts are references, not evidence that any asserted statement is entailed.

The index is built atomically, treated as immutable, and verified on opening. The loader validates the canonical L1/L2 pin, fit split and source bytes, SQLite integrity, complete stored records and every indexed vector weight against the source cache. No paid API, GPU or hosted vector database is needed.

## Two intentionally different database scopes

- **all**: all 450 records are searchable for interactive browsing. TF-IDF vocabulary remains fit only on the declared training episodes, but no held-out retrieval claim can be made in this mode.
- **train**: only the episode-disjoint training records are retrievable. Use this mode for any holdout evaluation. Records in validation and test are not candidates.

The original biography and FlyWire v783 anatomical connectivity remain unchanged. BioCircuit's separate 256-dimensional signed lexical vectors are not interchangeable with this TF-IDF database.

## Reproduce on Python 3.11+

    python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'scikit-learn>=1.5,<2' 'rank-bm25>=0.2,<1'
    python scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/vector-db-seed31
    python scripts/vector_fly_db.py build --cache-dir data/derived/vector-db-seed31 --database data/derived/vector-fly-450.sqlite --scope all
    python scripts/vector_fly_db.py query --cache-dir data/derived/vector-db-seed31 --database data/derived/vector-fly-450.sqlite --text 'camphor beetle brass key' --top-k 5
    python scripts/vector_fly_db.py verify --cache-dir data/derived/vector-db-seed31 --database data/derived/vector-fly-450.sqlite
    python scripts/vector_fly_db.py build --cache-dir data/derived/vector-db-seed31 --database data/derived/vector-fly-train31.sqlite --scope train
    python scripts/vector_fly_db.py get --cache-dir data/derived/vector-db-seed31 --database data/derived/vector-fly-450.sqlite --event-id E01-001

Existing database files cannot be silently overwritten; use a new filename for an updated snapshot. Query requires the verified L2 cache directory, because a query must use exactly the same vocabulary and IDF that built the index.

## Tests and evidence

    python -m unittest discover -s tests -p test_vector_store.py -v

The tests build both scopes from the real immutable 450-event autobiography, check exact scores against the canonical sparse matrix, compare stored original records byte-semantically, enforce held-out episode exclusions, reject source and posting tampering, reject a mismatched L2 seed, and check reproducible ranking.

The [CI workflow](../.github/workflows/vector-fly-database.yml) builds both SQLite files, verifies them and uploads their binaries plus machine-readable build/query results. The versioned code and pinned inputs are permanent on GitHub; workflow binaries are downloadable but subject to artifact retention. Add tested outcomes to the permanent research handoff only after CI is green.

## Scope of claims and next work

This is a fully functional lexical **vector database**, not an ANN/HNSW engine and not a demonstration of biological synaptic storage. For 450 memories, exact search is cheap, deterministic and auditable. It does not prove semantic understanding, contextual entailment or continuity of Pretorius's character.

The natural next integration is optional graph reranking of identical database candidate lists, with explicit real-vs-rewired FlyWire conditions and latency tracking. Do not silently substitute graph scores for the lexical source baseline. Any semantic embedding backend must have a separate dimension, model identity/checksum, query encoder and held-out protocol.


## Release validation

The initial end-to-end database build passed [run 37863611161](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863611161). After a PR review identified a possible concurrent-build overwrite, publication was changed to an atomic create-if-absent hard link. A new regression explicitly simulated that race, and all **seven** tests plus the complete 450-memory and training-only database build and full vector verification passed [run 37864046037](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37864046037). The [final SQLite artifact bundle](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37864046037/artifacts/11587801011) includes both measured indices and JSON queries. This remains lexical, not semantic.
