# Shared memory v1: operational portable L1 and L2 cache

**Status:** implemented and verified on CI. The historical corpus and biological FlyWire CSR arrays are unchanged.

The canonical source is the frozen v12 autobiography with exactly 450 records and 27 episodes, Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`. The candidate sidecars are pinned to Git blob `ad32025166c382caf13e07c7e3b0863eb89e1adb`. Both are reconstructed literary history, not observed lived memories or independently approved sensory facts.

The single source-owned implementation is `src/pretorius_connectome/shared_memory.py`. It uses `load_v12` and `episode_split` from the existing project rather than replacing them. L1 records preserve original first-person text, dates, people, cues, link IDs, relationship changes, provenance and explicitly unreviewed candidate annotation. Metadata stays separate from vector features. L2 is the existing FlyWire associative word/unigram-bigram, English-stopword, sublinear TF-IDF baseline capped at 8,192 terms. Its vocabulary and IDF fit ONLY on episode-disjoint training memories, then transform all 450 fixed narratives for indexing/evaluation. No event ID, action label or source metadata is encoded in the TF-IDF vector.

Build and validate a complete cache locally (Python 3.11+, NumPy/SciPy/scikit-learn):

```sh
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2" "scikit-learn>=1.5,<2"
python scripts/build_shared_memory.py --seed 31 --output-dir data/derived/shared-memory-v1 --report results/shared-memory/seed31.json
python -m unittest discover -s tests -p test_shared_memory.py -v
```

The directory contains `manifest.json`, `records.jsonl`, `vocabulary.json`, `idf.npy`, `docs.npz`. The manifest records source/annotation Git hashes, schema version, ordered IDs, encoder code checksum, fit and held-out IDs, episode split, vector dimensions, dtype, normalization and SHA-256 for every shard. `SharedCache` verifies this metadata, shard bytes, source ID coverage and order, episode disjointness and a deterministic encoder-to-cache roundtrip before allowing use. Any incompatible source or encoder version fails closed. No unsafe pickle is loaded.

For a FlyWire/synthetic-graph retrieval benchmark, supply the cache as an optional flag while preserving the prior fit-on-demand behavior as the default:

```sh
python scripts/run_associative_memory.py --synthetic-test --benchmark --seeds 31 --shared-cache data/derived/shared-memory-v1 --output results/associative/shared-seed31.json
```

The cached FlyWire consumer requires its training ID set to exactly equal the cache's IDF fit set; it cannot silently use another seed. Cache vectors are a lexical retrieval baseline, not a semantic representation. The graph projection, actual v783 CSR roots, contacts and connectivity are not rewritten.

## First measured engineering result

Successful source CI: [run 37839340670](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37839340670). All 6 cache/consumer regression tests passed. For seeds 31 and 37, the source-verified full cache built and reloaded successfully, yielding 8,192 dimensions each. The pinned training episode splits had 317 and 316 events respectively. Tracemalloc reported 11,380,917 bytes and 11,381,903 bytes of peak tracked allocations during the two build+reload procedures. Build wall times in that CI runner were 0.917 and 0.902 seconds; build-plus-verified-reload times were 1.214 and 1.207 seconds. These measurements exclude package installation, OS allocations untracked by tracemalloc, any FlyWire graph and downstream neuronal inference.

Synthetic topology retrieval with the same training split produced matching cached-versus-legacy lexical, graph and hybrid scores to numerical tolerance. This is parity evidence, not proof of a neural or accuracy improvement. Both cached and uncached approaches still require the same graph calculation. There is no claimed universal wall-time speedup from shared vector reuse.

CI stores generated cache files, manifests and reports as a regenerable workflow artifact named `pretorius-shared-memory-v1`. The code and this measured report remain in Git even when the artifact expires. To use this feature from BioCircuit, import this source-owned package from the checked-out Connectome repository and apply a distinct, versioned L3 projection from 8,192 TF-IDF features into BioCircuit's 256 sensory channels. Never pretend the legacy 256 signed token-hash baseline and the new 256 TF-IDF folding are identical feature spaces.

**Limitations:** present pilot challenges remain development-exposed; no semantic embeddings, entailment proof, autobiographical neural recollection or identity continuity are implied.
