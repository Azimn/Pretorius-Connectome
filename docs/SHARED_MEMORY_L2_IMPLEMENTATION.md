# Shared memory L2: reusable, split-safe lexical feature artifact

**Status:** implemented and CI-verified after reconciling with the existing published L1 archive. This is an engineering cache, **not** a learned semantic representation or biological memory.

**Immutable L1 owner:** `artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz` and `artifacts/shared_memory/v1/manifest.json`. These already belong to Pretorius-Connectome and are not regenerated, overwritten, or duplicated by this implementation. L1 holds original source-anchored text and structured biographical metadata from the frozen v12 450-event corpus. L2 is added by `src/pretorius_connectome/shared_memory_l2.py`, using the existing `read_l1`, `load_v12` and `episode_split` routines.

**L2 build:**

```sh
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2" "scikit-learn>=1.5,<2"
python scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/shared-memory-l2-seed31 --report results/shared-memory/seed31-l2.json
```

Output comprises `manifest.json`, `records.jsonl`, `vocabulary.json`, `idf.npy` and `docs.npz`. The L2 manifest pins L1 archive and manifest SHA-256, original Git source/blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`, candidate sidecar blob `ad32025166c382caf13e07c7e3b0863eb89e1adb`, the encoder implementation's SHA-256, ordered event IDs, all three episode-disjoint partitions, exact fit IDs, vocabulary dimension/dtype and every output-shard digest.

The encoder deliberately reuses the existing FlyWire associative retrieval word/unigram-bigram `TfidfVectorizer` parameters (English stopwords, sublinear TF-IDF, up to 8,192 vocabulary terms, L2-normalized float64 values). Only training-episode records are used to fit vocabulary and IDF. Withheld records are *transformed* using that frozen fit for independent episode evaluation, never included in fitting statistics. `SharedCache` verifies the fit partition, original L1 bytes, shard checksums, source event order and query-to-document consistency without pickle/deserialization of arbitrary Python objects.

**Optional FlyWire retrieval consumer** (keeps historical no-cache behavior unchanged):

```sh
python scripts/run_associative_memory.py --synthetic-test --benchmark --seeds 31 --shared-cache data/derived/shared-memory-l2-seed31 --output results/associative/shared-l2.json
```

An incompatible seed, corpus event identity, altered narrative or mismatched encoding configuration is a hard error. The cache is consumed before the topology-specific feature-to-neuron projection. L3 FlyWire synapse root IDs, CSR connectivity and contact counts are not modified.

**BioCircuit consumer:** the separate BioCircuit adapter loads the same `SharedCache` through a version-pinned Connectome checkout, then applies a separately labeled 256-sensory-channel signed feature bucket projection. It is therefore using the same L2 fitted lexical representation, **not** claiming its legacy 256-dimensional signed token hash has become equivalent to FlyWire's full 8,192-feature space. Original BC01 demonstrations remain runnable without the shared dependency.

## Measured acceptance checks, October 8, 2026

[CI run 37839758698](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37839758698) successfully ran the existing two L1 tests, six new L2 tests, both source-pinned full-corpus builds and cached-vs-uncached associative retrieval parity. At seed 31 the fitted training partition contains 317 events; at seed 37, 316 events. Each produces 8,192 feature dimensions. Build-only wall time: 0.846 s and 0.822 s. Build plus verification/reload: 1.132 s and 1.103 s. Python-tracked peak allocations: 9,986,708 and 10,007,486 bytes respectively. These timings are environment-specific, exclude dependency installation and external FlyWire graph work, and **do not establish an overall end-to-end speedup**.

The full cache manifests and per-query retrieval results are saved as regenerable CI artifacts. The source implementation, tests, workflow and this report are kept permanently in the repository. Future cache schema or encoder updates require new manifests and regeneration; do not silently accept stale output. Previously examined Pilot 04-07 probes are development-only and cannot supply independent semantic validation.

**Remaining research gate:** shared preprocessing and training-data parity are distinct from proving that biological connectivity, neural plasticity or a personalized recurrent substrate benefits autobiographical decisions. No learned synaptic weights or speculative claims are transferred to The Doctor Lives.
