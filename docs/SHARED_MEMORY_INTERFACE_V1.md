# Shared Pretorius memory interface v1 — operational artifact

**Canonical owner:** Azimn/Pretorius-Connectome. **Consumers:** its FlyWire associative retrieval and Azimn/Pretorius-Neural-Network BioCircuit BC01. Tracker: https://github.com/Azimn/Pretorius-Connectome/issues/10

## Portable layers

L0 is the unchanged 450-event, 27-episode v12 fictional autobiography (event Git blob 718dcc2d5ba4feccdef1690d447edfcebaa9bfb5; sidecar ad32025166c382caf13e07c7e3b0863eb89e1adb). L1 is 450 ordered exact full source JSON dictionaries plus original candidate cue IDs and status in l1_records.jsonl; preserves all original IDs, dates, relationships and source fields. L2-BC is bc01_lexical_256.npy: 450x256 float32 arrays from the **existing BC01 signed BLAKE2b unigram/bigram hashing**, not a semantic encoder. L2-BC has **no training fit** (fit_event_ids=[]); it can be safely reused on each episode split. FlyWire TF-IDF fits an entirely different 8,192-feature (up to) word-bigram vocabulary and IDF strictly on *train episodes per seed*, so cannot share that L2 without an explicitly split-pinned cache. Existing FlyWire topology, synapse counts and neural weights are unaltered.

manifest.json includes schema version, immutable original blobs, canonical source commit, source field mapping version, encoder algorithm hash, source-ordered IDs, SHA256 of both artifacts, encoder name and dim, model revision null, fit scope, feature normalization, dtype and random seed. The loader verifies checksums, ordering, annotations, IDs, dimensions, finite values and recomputes cached lexical features; it fails closed rather than silently using a mismatched cache. Consumer implementations separately compare original pinned source or original encoder to the loaded artifact. Hashes detect corruption, but not a malicious party replacing an entire file plus unauthenticated manifest; pin repository revisions for untrusted distribution.

## Reproduce

    python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'scikit-learn>=1.5,<2'
    python scripts/export_shared_memory.py --output results/shared_memory/v1
    python scripts/export_shared_memory.py --output results/shared_memory/v1 --validate-only
    python scripts/run_associative_memory.py --synthetic-test --query "millstream map" --shared-dir results/shared_memory/v1
    python scripts/run_associative_memory.py --synthetic-test --benchmark --seeds 31 --shared-dir results/shared_memory/v1 --output results/shared_memory/associative.json
    python -m unittest discover -s tests -p 'test_shared_memory.py' -v

The optional --shared-dir path changes only the input-record loader: all original BM25/TF-IDF, graph and held-out episode operations remain the same. Demonstration topology is **synthetic**, not FlyWire. BioCircuit companion CLI also supports --shared-dir and replaces source-side hashed training exposures with exact cached vectors while leaving query encoding and recurrent circuit architecture unchanged.

Shared processing alone cannot establish semantic recall, entailment or superiority of a neural architecture. Existing test prompts and BC decision cards are exploratory and not an independent gold benchmark. Keep per-architecture weights, checkpoint and biological synapse arrays separate. GitHub Actions retains actual exported L1/L2 and measurement artifacts for reproducibility.
