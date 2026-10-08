# FlyWire MB plasticity Pilot 01: superseded v1 diagnostic

**Run:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37842622884

**Original full case-level artifact:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37842622884/artifacts/11578398053

**Original source-run JSON:** 337,207 bytes, SHA-256 `653681284667c77a3abd9cf7286ffaf06c81644a82d3115329a247aba6acaa44`.

**Scientific status:** this real FlyWire v783 mushroom-body experiment executed successfully on 2026-10-08 against the historical, cross-process-nondeterministic v1 lexical feature selector. Its exact within-run, case-level evidence is preserved by the linked artifact while available. This result is **superseded as the canonical experiment** by deterministic TF-IDF L2 v2, which must be executed and archived separately. Do not treat this as an independent blind confirmation or reproducible multi-run neural-performance estimate.

The original anatomical graph contained 14,025 neurons and 574,660 directed aggregate connectivity entries. The pinned fictional autobiography comprised 450 reconstructed Pretorius memories; the reused 68 challenge prompts were previously examined, assistant authored and not human reviewed. The same three episode-disjoint seeds (31,37,43) were used as earlier associative studies.

## Descriptive pooled results, 67 positive prompts / 67 contradiction prompts / 20 absent episodes

| Condition | Positive top-1 | Correct and accepted | Contradiction false acceptance | Absent false acceptance |
| --- | ---: | ---: | ---: | ---: |
| Narrative TF-IDF, historical v1 | 15/67 | 5/67 | 16/67 | 1/20 |
| Real MB, frozen | 13/67 | 7/67 | 10/67 | 1/20 |
| Real MB, learned | 13/67 | 6/67 | 10/67 | 1/20 |
| Rewired MB, frozen | 15/67 | 7/67 | 10/67 | 1/20 |
| Rewired MB, learned | 15/67 | 7/67 | 10/67 | 1/20 |
| Real MB, learned hybrid | 15/67 | 5/67 | 16/67 | 1/20 |
| Rewired MB, learned hybrid | 15/67 | 5/67 | 18/67 | 1/20 |

The pre-specified graph-only contrast was **negative**: the real plastic topology correctly identified and accepted one fewer positive question than its equally trained rewired control (6/67 versus 7/67), and one fewer than its own untrained baseline (6/67 versus 7/67). The false-contradiction rate was the same 10/67 in all three. The single real-minus-rewired lost acceptance came from seed 31; seeds 37 and 43 each had zero net gains or losses.

Training applied an identical coactivity rule constrained to the existing directed graph and did not modify source integer synapse counts or raw CSR edge endpoints. The number of coactivity-touched edges varied with the topology: real versus rewired, 77,211 versus 83,809 on seed 31; 77,141 versus 84,209 on seed 37; 77,573 versus 84,064 on seed 43.

**Additional null-model confound:** the old target-stub permutation retained the raw 574,660 connections but introduced duplicate target pairs in some rows. When CSR duplicates were merged, its effective edge count fell to 556,627, 556,675 and 556,732 in seeds 31, 37 and 43, respectively, versus 574,660 effective edges in the original. The two networks therefore had different numbers of trainable edge parameters. The canonical v2 pilot corrects this with source/destination-degree and unique-support-preserving directed edge swaps, and requires an exact capacity match before archiving.

**Limitations:** all results are observational comparisons of lexical evidence retrieval against a fixed model-specific challenge, not a new independent evaluation. More critically, the original `sklearn-tfidf-word12-v1` feature selector had a documented cross-process feature-coordinate reproducibility failure. Reproducing the old run could change its vectors and predictions. The corrected rank-stable v2 encoder has **different feature-selection semantics**; a new v2 result must be recorded separately, never silently described as a strict replication of this v1 result. The full artifact is subject to GitHub retention and does not itself satisfy permanent raw-data archiving.

The definitive new experiment and its permanent provenance gate are tracked by [PR #14](https://github.com/Azimn/Pretorius-Connectome/pull/14), [the v2 feature incident](../../docs/SHARED_MEMORY_L2_V2_REPRODUCIBILITY.md), and [the Pilot 01 protocol](../../docs/FLYWIRE_PLASTICITY_PILOT_01.md).
