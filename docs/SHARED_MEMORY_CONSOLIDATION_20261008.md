# Shared memory L1/L2 consolidation checkpoint, 2026-10-08

**Purpose:** preserve the observed competing-PR resolution, the actual tested compatibility boundary, and outstanding scientific/engineering limitations. Continue the FlyWire associative experiment from here rather than independently recreating encoders.

## Resolved GitHub branches

- [PR #11](https://github.com/Azimn/Pretorius-Connectome/pull/11): **closed, not merged**. Its initial shared_memory.py implementation conflicted with the published canonical L1. The working separate TF-IDF logic was repackaged rather than force-merged.
- [PR #13](https://github.com/Azimn/Pretorius-Connectome/pull/13): **merged**. Provides `shared_memory_l2.py`, a separately versioned train-episode-fit TF-IDF feature cache, opt-in FlyWire consumer, and original no-cache baselines.
- [PR #12](https://github.com/Azimn/Pretorius-Connectome/pull/12): **merged**. Provides separate `shared_features_bc01.py` for the stateless exact 256-sensory signed lexical feature representation. It does not replace canonical L1.
- [BioCircuit PR #31](https://github.com/Azimn/Pretorius-Neural-Network/pull/31): **merged**. Provides BioCircuit's separate opt-in TF-IDF-to-256 L3 projection using the Connectome-owned fitted cache, while retaining its original lexical neural inputs unchanged as the default experiment.

**Important:** the original BioCircuit signed-hash vectors, train-fitted FlyWire TF-IDF vectors, and the optional BioCircuit projection of TF-IDF are three **different** feature configurations. None is a common learned synaptic representation or evidence of autobiographical understanding.

## Canonical artifact and provenance

The 450-source-record / 27-episode L0 is authoritative: `memories/current/Pretorius_v12_450_Events_Complete.jsonl`, Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`.

The [published L1 manifest](../artifacts/shared_memory/v1/manifest.json) and [published L1 gzip](../artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz) retain immutable Git blobs `7a8d05b46b025b1bfe44339b3edbee040df22394` and `5381c3c22ab224bb6ebf9f037489c1b9be564888`. L1 v1 contains a 17-field **projection**, not a complete replication of every source attribute. In particular, original L0 may contain other causal and interpretive fields such as `formation_mechanisms`, `causal_inference_status`, and `alternative_interpretation`. **Do not claim lossless preservation of every L0 field in this frozen L1 v1 or overwrite published L1 bytes.** Preserve the original L0 and introduce any future full-metadata export under a different schema and pin.

## Corrected L2 validation

The code review of PR #13 identified a risk of L2-manifest/feature-shard self-consistency masking bad provenance. `src/pretorius_connectome/shared_memory_l2.py` now:

1. Checks the published L1 archive and manifest against **independent code-pinned Git blob identities**, not values from the L2 cache being inspected.
2. Compares all projected cache records with those in canonical L1, not just IDs, order and source-label text.
3. Reconstructs episode-disjoint seeded splits from the original fixed corpus and checks the declared train/validation/test event lists.
4. Independently **refits vocabulary and IDF using only declared training narratives**, checking the restored vocabulary/IDF exactly before using any cached feature matrix.
5. Still checks shard hashes, query-vs-source transformations, and that the FlyWire graph/synapse counts are not modified.

New regression tests reject deliberately modified narratives with rehashed shards, substituted IDF arrays with rehashed manifests, and declared-seed drift. [Corrected CI workflow 37840675033](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37840675033) passed its **2 canonical L1, 9 TF-IDF L2, and 4 BioCircuit sensory cache tests**, including synthetic-graph cached-vs-original lexical/graph/hybrid comparison. [Foundation suite 37840675023](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37840675023) passed. This validates cache integrity, not retrieval quality.

## Parallel BioCircuit adapter

The BioCircuit consumer from merged PR #31 uses the shared source-fitted TF-IDF encoder only as an **optional** experimental input followed by its own labeled L3 sensory projection. Its existing 256-dimensional signed hashed lexical encoder is still the default. Initial [postmerge consumer CI 37840771885](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37840771885) passed.

A subsequent code review raised the risk of Python module caching allowing a **wrong Connectome checkout** to supply the feature code in long-lived processes. The BioCircuit adapter now rejects any preloaded `pretorius_connectome` module from outside the requested source checkout, and its regression test simulates wrong-checkout contamination. Inspect the new [BioCircuit shared-memory workflow](https://github.com/Azimn/Pretorius-Neural-Network/actions/workflows/biocircuit-shared-memory.yml) before marking the additional guard verified in CI.

## Scientific disposition

The finished full FlyWire and mushroom-body topology experiments remain separately recorded in [permanent case-level results](../results/associative/README.md). Neither proved better correct autobiographical recall specifically due to real biological wiring. The shared-memory work is an **interoperability and reproducibility intervention**. It does not change their original measured results, replace biological topology, produce semantic embeddings, or justify transferring BioCircuit recurrent weights.

**Next focused research decision:** resume the original FlyWire neural hypotheses using the fixed L1 plus separate L2 controls. An anatomically informed memory cue population and explicit plasticity would be a new preregistered experiment, not a reason to rebuild the prior source pipeline. Future claimed advantages require an independent test set. [Cross-project tracker #10](https://github.com/Azimn/Pretorius-Connectome/issues/10) remains open for time/memory efficiency and independently reviewed comparative validation.
