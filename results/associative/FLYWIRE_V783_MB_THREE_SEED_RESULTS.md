# FlyWire v783 mushroom-body neuropil experiment: executed three-seed findings

**Recorded:** October 8, 2026. **Run:** [37836674947](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836674947). **Status:** successful biological acquisition, extraction, tests and three-seed evaluation. This is a descriptive, **post-hoc**, previously examined and unreviewed semantic challenge. Results do not constitute a new blind study.

## Source and anatomy

The source is the publisher's checksum-verified FlyWire v783 `proofread_connections_783.feather` and `proofread_root_ids_783.npy`. The updated converter selects `MB_` neuropil-tagged rows, validates original IDs, sums original positive synapse counts when aggregating directed neuron-pair contacts, and keeps incident nodes only.

* Original publisher connection rows: **16,847,997**
* Mushroom-body-tagged source rows: **638,841**
* MB subgraph directed connectivity pairs: **574,660**
* Incident neurons: **14,025**
* Preserved synaptic contacts in selected regions: **1,535,281**

This is a neuropil-restricted directed graph, not a complete Kenyon-cell subsystem. It excludes contacts in other neuropils involving the same neurons. All input language features are still assigned to neurons through arbitrary, deterministic text hashing. The graph uses positive-only log1p count weights, source normalization and bounded sparse diffusion, **not biological synaptic plasticity or neurotransmitter signs**.

## Measured evaluation

Three episode-disjoint seeds (`31,37,43`) use the same 450 frozen source events and the already-studied 68-case assistant-authored challenge. Across seed splits: 67 positive-paraphrase evaluations, 67 paired contradictory probes and 20 absent-episode probes. These observations are not all statistically independent.

| Method | Correct positive top-1 | Correct and accepted | Contradictions falsely accepted | Absent episodes falsely accepted |
| :--- | ---: | ---: | ---: | ---: |
| Narrative TF-IDF | 15/67 (22.4%) | 5/67 (7.5%) | 16/67 (23.9%) | 1/20 (5.0%) |
| Actual MB connectivity, graph-only | 13/67 (19.4%) | 7/67 (10.4%) | 10/67 (14.9%) | 1/20 (5.0%) |
| Actual MB connectivity, hybrid | 15/67 (22.4%) | 5/67 (7.5%) | 17/67 (25.4%) | 1/20 (5.0%) |
| Rewired MB topology, graph-only | 15/67 (22.4%) | 7/67 (10.4%) | 10/67 (14.9%) | 1/20 (5.0%) |
| Rewired MB topology, hybrid | 15/67 (22.4%) | 5/67 (7.5%) | 16/67 (23.9%) | 1/20 (5.0%) |

Per-case paired comparisons show the real MB graph versus rewired graph changed the predicted event on 5/23, 3/20 and 4/24 positive probes by seed. Yet there were **zero correct-and-accepted gains and zero losses** across all three seeds in that pairing. The positive top-1 accuracy for the real anatomical graph was lower than the rewired control overall, despite matching its accepted-correct count. Its apparent ability to reject more contradictions than lexical retrieval was **fully reproduced by the rewired control** under this evaluation, and is therefore not evidence of a special advantage from biological wiring.

## Interpretation and limitations

**No positive topology-specific result.** The measured MB subgraph alters some rankings, but offers no improvement in correctly accepted autobiographical recall over its degree-stub randomized comparator. The positive graph-only accuracy is lower than the rewired baseline. Hybrid retrieval is mostly the original text retriever in these measurements.

No anatomical class-to-feature mapping was used. A region label is not an independently validated neural circuit model. Scores do not check whether a retrieved passage logically entails or contradicts the prompt, and the entire benchmark uses previously examined, assistant-authored, unreviewed cases. Results should not be promoted into publishable conclusions of general semantic memory.

The published original FlyWire v783 data has been **derived** into two CSR representations; the biological source has not been altered. For the computational transformations and rerun commands, see [the ROI protocol](../../docs/ASSOCIATIVE_MEMORY_ROI_EXPERIMENT.md). The extraction and benchmark runner's GitHub tests passed on this run.

**Reproducibility:** [Successful workflow](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836674947), [full JSON artifact](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836674947/artifacts/11574924963), [per-seed compact numerical evidence](FLYWIRE_V783_MB_THREE_SEED_COMPACT.json). The full original raw JSON and conversion report are also targeted for archival into `results/associative/runs/` by `.github/workflows/associative-archive-historical.yml`; verify that archival run before claiming permanent raw preservation.

## Next experiment decision

Stop repeating unstructured topology swaps. If continuing this research line, first test an **explicit neuron-class-aligned encoding** with pinned FlyWire annotations (Kenyon cells, projection neurons, output neurons, modulatory inputs). Compare against cell-class-permuted and matched-degree nulls with equal numbers of representational units and the same learnable capacity. Then run a newly authored, independent, human-reviewed evaluation before any claims of generalizable biological advantage. Do not change the frozen baseline results.
