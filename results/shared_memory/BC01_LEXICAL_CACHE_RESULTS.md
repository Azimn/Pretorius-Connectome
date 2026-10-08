# BC01 exact lexical L2 cache: engineering measurements

**Execution:** October 8, 2026. **Status:** canonical provider and BioCircuit consumer implemented and tested in both repositories. Final fresh post-merge CI may run separately. This is a **stateless lexical cache, not a semantic embedding**.

## Provenance and durable code
- Existing immutable L0 archive: 450 reconstructed fictional events in 27 episodes, original Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`.
- Existing committed compressed, checksum-pinned L1 remains at `artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz` and its existing `manifest.json`.
- Existing source-owned split-fitted shared TF-IDF L2 (`shared_memory_l2.py`) remains unchanged, vocabulary/IDF fitted **train episodes only** and FlyWire graph/CSR unchanged.
- Additional optional exact BC01 source sensory L2: `src/pretorius_connectome/shared_features_bc01.py`, exporter `scripts/export_shared_memory.py`. Produces `bc01_sensory_256.npy` (450×256 float32) and separate `bc01_l2_manifest.json` containing source, ordered IDs, SHA-256, encoder spec, dtype, L2 norm and explicitly empty fit IDs. Does not edit the published L1 files.
- Provider merged by [PR #12](https://github.com/Azimn/Pretorius-Connectome/pull/12), main squash commit `81146c8b8b3cc0d538c5055bdabf1f767e0edd2f`.
- Consumer merged from BioCircuit [PR #32](https://github.com/Azimn/Pretorius-Neural-Network/pull/32) at commit `2191156597b7d7cff694d7c0178cc154e081a149`. [BC01 implementation CI 37841321751](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841321751) passed full 450-event baseline-vs-cached test-result equality, matching recurrent synapse delta counts and checkpoint checks, and original L1 plus fitted shared TF-IDF input regression. The permanent companion report is [BC01 legacy cache results](https://github.com/Azimn/Pretorius-Neural-Network/blob/main/results/biocircuit/BC01_LEGACY_CACHE_RESULTS.md).

## Actual CPU and storage observations
From successful GitHub Actions [shared-memory L2 workflow 37840409964](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37840409964). Linux runner, one measured iteration, **synthetic 128-node graph only**, training 317 / validation 62 / test 71 original events for seed 31. No FlyWire anatomical topology was loaded in this timing assay. Numbers are seconds except storage.

| Operation | Measured |
| --- | ---: |
| Read original 450-event archive | 0.011961 s |
| Read shared compressed L1 with integrity validation | 0.015065 s |
| Re-encode 450 BC lexical sensory vectors from scratch | 0.154920 s |
| Load **and fully revalidate** 450 cached BC vectors against source | 0.165228 s |
| Split-fitted TF-IDF + synthetic graph setup from original | 0.115098 s |
| Same TF-IDF + graph setup through shared L1 | 0.087941 s |
| Combined published-L1 + BC-L2 files, manifests included | **710,405 bytes** |
| Measured process peak RSS | 132,300 KiB |

**Both exact lexical feature replay and identical original-vs-shared FlyWire graph/retrieval scores passed.** No synapse arrays were altered. The times are environment-dependent single measurements. In particular, strict full verification of the cache **took longer than uncached hashing on this run** (0.165 vs 0.155 s): no unconditional preprocessing speedup has been demonstrated. The major benefit is compatibility, provenance and deterministic input equivalence, not runtime performance or greater semantic power.

## Acceptance boundaries
- BC01 lexical L2 and FlyWire train-fit TF-IDF L2 remain **distinct** spaces. Sharing source L1 and a stateless lexical sensory precompute does not make neural weights transferable.
- An event ID and source metadata are evaluator/provenance fields, not query features. Unknown or contradiction benchmarks remain unreviewed development materials.
- Source tests, L1 replay, baseline FlyWire controls and fixed-source checks are green. BioCircuit companion PR #32 was merged; run final post-merge CI in addition to the passing original branch comparison before declaring all green.
- Real v783 synapse counts are unmodified by this pipeline; timing parity was measured on a synthetic graph, not the full FlyWire dataset.

**Reproduce:** `python scripts/export_shared_memory.py --output artifacts/shared_memory/v1`; `python scripts/benchmark_shared_memory.py --bundle artifacts/shared_memory/v1 --output results/shared-memory/bc01-cache-measured.json`. GitHub Actions uploads complete generated cache and benchmark JSON as artifacts.
