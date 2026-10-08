# Pretorius cross-project memory reuse contract (v1)

**Status: documented integration contract, not a claim that a shared embedding cache already exists.**  
**Canonical source owner:** `Azimn/Pretorius-Connectome`. **Consumers:** `Azimn/Pretorius-Neural-Network` (BioCircuit), FlyWire-derived retrieval/imprinting experiments in this repository, and, only after an explicit migration gate, `Azimn/The-Doctor-Lives`.

## Purpose and ownership

Both BioCircuit BC01 and FlyWire experiments ingest the same reconstructed first-person biography. They MUST NOT independently regenerate events, deduplicate differently, silently rewrite narratives, or each invent a second general-purpose data-preparation pipeline. This repo owns the immutable record and its editorial annotation sidecars. Consumer experiments own their distinct neural substrate, topology, projection and plasticity. The canonical fiction is NOT proof of observed lived experience.

| Layer | Single source / reusable output | Status and responsible implementation |
| --- | --- | --- |
| L0, source | `memories/current/Pretorius_v12_450_Events_Complete.jsonl`, 450 events / 27 episodes, Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`; `memories/annotations/v12_450_sidecars.jsonl` is an editorial **candidate** sidecar, not verified perception | Frozen in Pretorius-Connectome. BioCircuit `biocircuit/bc01.py` already validates the source blob and 450 count, and provides a small smoke fixture |
| L1, normalization / provenance | Stable event IDs, episode IDs, first-person text, source fields, recall cues, dates, relationships and candidate sensory annotations; deterministic normalization and train/validation/test identifiers | **Reuse existing archive/validators; common portable normalized-record export not yet built** |
| L2, encoder-dependent feature representations | Optional content-derived tokenized/hashed/TF-IDF/locally computed semantic vectors with encoder version, training split and fit-state pin | **Not currently a common shared cache.** BC01 uses `persona_net.encoding.ExperienceEncoder` and `biocircuit.bc01.represent` (signed hashed lexical, sensory 256); FlyWire retrieval uses `src/pretorius_connectome/associative.py` (word/bigram TF-IDF, max 8,192 and stable feature-to-neuron hash), while Pilot 03 has train-only cue encoders |
| L3, topology-specific transformation | BioCircuit compartments + recurrent donor vs fixed FlyWire v783 CSR and feature-to-neuron mapping, synthetic masks and learned weight overlays | Remain **experiment-specific**. Never directly transplant learned recurrent weights or alter original biological synapse counts |
| L4, evaluations | Frozen test queries, matched comparison budgets, human-interpreted source decision cards, observed output and limitations | Reuse as explicitly tagged development-only assets, with source attribution. The 16 BC01 human-interpreted cards are **not** historical ground truth or an independent holdout |

## Reuse rule for future coding agents

Before writing any corpus importer, normalization, encoding or retrieval module, inspect the source repo and the current BioCircuit files:
- `memories/README.md`, `scripts/validate_autobiographical_corpus_v12.py`, `memories/annotations/v12_450_sidecars.jsonl`
- `src/pretorius_connectome/associative.py`, `src/pretorius_connectome/pilot03.py`, `scripts/run_associative_memory.py`
- `Azimn/Pretorius-Neural-Network/biocircuit/bc01.py`, `biocircuit/bc01_decisions.py`, `persona_net/encoding.py`, `resources/biocircuit/bc01_decision_cards_v1.json`
- [BioCircuit BC01 issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26), [FlyWire research handoff](RESEARCH_HANDOFF.md), [local-NLI continuation issue #9](https://github.com/Azimn/Pretorius-Connectome/issues/9)

Prefer importing, adapting or materializing a *version-pinned* existing intermediate representation over rebuilding. Explicitly name the missing capability if a new module is required. Keep all original L0 byte content unchanged. Never use episode/event IDs, decision labels, truth labels, review verdicts or holdout probes as features at inference time.

## Planned portable handoff artifact

If a new shared representation is warranted, implement **once** in this source repository with an offline CPU generator and tests. A manifest MUST record `schema_version`, `source_repo`, `source_commit`, `source_git_blob`, `annotation_git_blob` if used, `record_ids_ordered`, `normalization_version`, `encoder_name`, `encoder_code_hash`, `model_revision` and its checksum if any, `fit_event_ids`/episode split, `vector_dim`, `dtype`, `feature_normalization`, `random_seed`, and checksums for output shards. Store stable source metadata beside rather than inside neural vector dimensions. Make all consumers verify hashes and fail closed on mismatches, and provide a smoke fixture.

**Do not fit a corpus-wide IDF/feature-statistics model and then call episode-held-out records independent.** Create distinct fit caches keyed by the training split. Query and source must use the same corresponding fitted encoder; keep held-out probes and withheld texts out of fitting. Model-specific features may be shared when encoder and split match, but a new model or a new split legitimately requires a distinct derived representation. The BioCircuit hashed 256-dimensional vectors and FlyWire 8,192-feature TF-IDF vectors are not interchangeable merely because they originate from the same text.

The planned FlyWire adapter should consume L1/L2 records, apply a separately versioned feature-to-neuron projection, and preserve original FlyWire connectivity and biologically measured synapse counts. Compare identical input feature caches and evidence sources for real graph, rewired graph, synthetic controls, and BioCircuit where an apples-to-apples evaluation is actually possible. Store training weights, graph diffusion outputs and checkpoints separately. A neural candidate score is not an entailment verdict and not proof of a character's remembered life.

## Practical next integration gate

1. Inventory existing normalized fields and reusable encoding code, starting with the already-pinned v12 corpus and BC01 decoder; do not scrape or author biography again.
2. Add a minimal portable L1 export plus hash-and-split-aware L2 feature cache **only if** existing code cannot be directly imported or adapted. Prefer existing `scikit-learn` TF-IDF for that baseline; do not invent another tokenizer for equivalent work.
3. Refactor BC01 and the FlyWire associative runner to accept the exact same external representation through small adapters while retaining their original methods as frozen baselines.
4. Test equal event order, deterministic hashes, cache/reload equality, disjoint held-out splits, invalid-cache rejection, identical query preprocessing, and unchanged FlyWire CSR synapse counts.
5. Commit measurements and failures, including CPU time and memory cost, and mark any source-dependent evaluations as development-only.

**No measured speedup or improved neural accuracy is claimed by this document.** Creating a contract does not yet make the repository share code or vectors. Future work must implement the adapter and verify it. Review this contract in both projects before adding overlapping pipelines.

## Version pins already used by BC01

BioCircuit `biocircuit/bc01.py` currently pins the full corpus at Pretorius-Connectome commit `597fb23473a60eecf2e1b50f79c22bfbea816be5` with the L0 Git blob listed above and also supports a bundled 3-record smoke fixture with blob `6e9404d8568b5e3431cdecce0ebd94a73248d6c8`. These pins describe the BC01 source as implemented, not a new FlyWire topology experiment.


## Operational L1 milestone, verified 2026-10-08

The previously planned **portable L1 artifact is now implemented**, exported and committed. The exact [source-pinned compressed archive](../artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz) and [manifest](../artifacts/shared_memory/v1/manifest.json) are the cross-project data interface. Full original corpus coverage, deterministic byte-identical export, strict corruption rejection, and unchanged source-derived FlyWire loader behavior were verified by [canonical workflow 37839291647](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37839291647). [BioCircuit workflow 37839516626](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37839516626) verified 450-record source parity, equal signed-hash input vectors and equal BC01 result arrays. The measured details and archive checksums are in [the permanent L1 report](../results/shared_memory/L1_INTERFACE_RESULTS.md).

This revises only the earlier **L1-not-yet-built** status; the previously planned **L2 shared vector cache is still not built**. The two architectures use incompatible feature spaces and cannot share their trained weights directly. The previous prospective text is preserved as planning history. Keep that distinction explicit in Issue #10 and subsequent development.

## Operational shared interface update (2026-10-08)

The immutable L1 exporter and compressed artifact now exist on main at `artifacts/shared_memory/v1/`. The next L2 feature layer is also operational, merged from [PR #13](https://github.com/Azimn/Pretorius-Connectome/pull/13) at `06bece269459a43d9e4ed09e5baabbacbd2082f7`. It intentionally consumes `shared_memory.read_l1` rather than re-exporting the original biography. The single TF-IDF cache builder and loader are `src/pretorius_connectome/shared_memory_l2.py`, used by FlyWire's optional `--shared-cache` path and imported directly by BioCircuit's `biocircuit/shared_features.py`. Full reproducibility, source and split checks, six L2 tests and cached-vs-legacy retrieval parity were confirmed by [CI 37839758698](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37839758698). BioCircuit's separate [integration CI 37840112970](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37840112970) passed 18 tests including full 450-memory import, identical vectorization of original source/query, and checkpoint restart. This shared cache uses lexical TF-IDF, **not a semantic embedding**. BioCircuit's 256-channel feature projection is an explicitly different L3 transformation, not a replacement for FlyWire L2 or for its original lexical-hash BC01 baseline. Reuse has not demonstrated useful neural action learning. Source code, [measured result](SHARED_MEMORY_L2_IMPLEMENTATION.md), and ongoing limitations now survive outside ChatGPT.
