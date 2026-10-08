# Shared TF-IDF L2 v2: cross-process deterministic feature selection

**Date:** 2026-10-08. **Status:** new versioned cache introduced after discovering a reproducibility failure in v1. The immutable original L0 autobiography and published L1 archive are unchanged.

## Observed failure of v1

BioCircuit BC01-D3 ran the exact same source checkout (v12 Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`), fixed seeds 31/37/43, program and Python package versions in independent Actions runs [37841883161](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841883161) and [37842237684](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37842237684). Nevertheless each run constructed **different SHA-256 fingerprints for `vocabulary.json`, `idf.npy` and `docs.npz`** at each of the three seeds. Example seed 31: first vocabulary fingerprint begins `77105afc`, second `ecbe2464`; first IDF fingerprint begins `720ecbd7`, second `6f69a1a9`. The different fitted feature coordinates caused different BC01 action predictions from otherwise identical numeric experiments.

This invalidates an earlier broad reproducibility inference from tests performed within only one fitted process. A cache being *internally consistent* and checksum validated is not sufficient if two independent builders create different feature spaces under the same declared version. The experimental D3 v1 accuracies must not be pooled, presented as replicable, or promoted into the canonical Pretorius.

## Intervention

The old `sklearn-tfidf-word12-v1` cache schema is replaced by the explicitly **different** `pretorius.shared-features.v2` / `sklearn-tfidf-word12-rankstable-v2`. This new vocabulary construction uses **existing scikit-learn CountVectorizer**, not a custom token parser. Fit only on episode-disjoint training narratives, count raw token and bigram frequency, then select up to 8,192 features by **frequency descending with lexical ascending tie-break**. Assign columns in lexical order and fit TF-IDF using that explicit fixed vocabulary on the *same training narratives*. Transform any validation or test narratives using the already fitted IDF, never fit on held-out data.

The source-owned L2 manifest contains the new schema name, versioned encoder, feature-selection rule, encoder implementation checksum, source/L1 and candidate annotation checksums, fit-event IDs, episode split and every output shard SHA-256. A v1 cache intentionally fails closed when loaded using v2. No old v1 output is relabeled or silently upgraded.

The historical FlyWire uncached `TfidfVectorizer(max_features=8192)` remains unmodified as an **independent baseline**. Because stable v2 selection can break ties differently from the old library default, its scores are **not asserted bit-identical** to the historical baseline. The previous direct equality test was inappropriate across differently selected vocabularies. The runner reports both as separate conditions, with their own input provenance. Biological FlyWire v783 neuron IDs and original CSR contact counts are unchanged.

## Measured validation

[Source-owned CI run 37842710890](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37842710890) passed the 2 frozen L1 tests, **10 L2 tests** including two independently launched Python processes with different `PYTHONHASHSEED` values, and 4 separate stateless BC01 lexical cache tests. The trained vocabulary, saved IDF array and sparse document matrix were exactly equal across both independent processes, including output shard hashes. The two source-verified seeds still fit training-only vocabularies of dimension 8,192. CPU cache build and load, source checks, and synthetic graph retrieval comparator also completed.

This establishes deterministic feature construction in the tested environment and guards future process-hash drift. It does not independently validate Pretorius identity, semantic entailment or any recurrent neural behavior. A future sklearn-version change remains a new environment variable: a deployed cache must pin the library versions and compare the manifest/checksums, never assume arbitrary wheel upgrades preserve identical floats.

## Consequence for BioCircuit

All v1-based D3 behavioral results remain valuable **negative diagnostic artifacts**, but cannot serve as a valid same-input multi-run comparison until regenerated against this v2 cache. Update the pinned source checkout in BioCircuit CI, regenerate every split-specific cache, verify cross-process equality and checkpoint restarts and preserve the old v1 raw JSON separately before publishing the new measurements.

The production The Doctor Lives repository is untouched.
