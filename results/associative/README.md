# Associative Memory Research: Permanent Results Index

**Status:** all original experiment output JSON files listed here are committed to this repository. The originating GitHub Actions artifacts remain secondary copies with finite retention. Frozen narrative sources and original biological data are not duplicated or rewritten here.

## Experimental reports

| Experiment | Interpretable report | Full per-case JSON | Original successful run |
| --- | --- | --- | --- |
| Synthetic 512-node directed test graph, one seed | [Synthetic findings](SYNTHETIC_V1_RESULTS.md) | [Original JSON](runs/synthetic-seed31-run37833271857.json) | [37833271857](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37833271857) |
| Real FlyWire v783 whole brain, one seed | [Initial biological result](FLYWIRE_V783_V1_RESULTS.md) | [Original JSON](runs/flywire-full-seed31-run37833431615.json) | [37833431615](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37833431615) |
| Real FlyWire v783 whole brain, three seeds | [Whole-brain three-seed report](FLYWIRE_V783_THREE_SEED_RESULTS.md) | [Original JSON](runs/flywire-full-three-seed-run37836107318.json) | [37836107318](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836107318) |
| Anatomically selected mushroom-body neuropils, three seeds | [MB three-seed report](FLYWIRE_V783_MB_THREE_SEED_RESULTS.md) | [Original JSON](runs/flywire-mb-three-seed-run37836674947.json) | [37836674947](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836674947) |
| MB selected-connectivity conversion metadata | [MB report](FLYWIRE_V783_MB_THREE_SEED_RESULTS.md) | [Exact conversion JSON](runs/flywire-mb-conversion-run37836674947.json) | [37836674947](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836674947) |

Compact, extracted stdout summaries for three-seed experiments are also versioned separately in [whole-brain compact statistics](FLYWIRE_V783_FULL_THREE_SEED_COMPACT.json) and [mushroom-body compact statistics](FLYWIRE_V783_MB_THREE_SEED_COMPACT.json). These are not substitutes for the full per-case files.

## Scientific findings and scope

The full FlyWire graph did not outperform its rewired control on correct positive top-1 identification in the frozen post-hoc test. The mushroom-body-restricted graph changed individual predictions but had 13/67 correct positive top-1 versus 15/67 for its rewired control, and 7/67 accepted correct positives in both. Its lower contradiction false-acceptance than the lexical baseline was also reproduced by the randomized control. Neither experiment demonstrates a topology-specific autobiographical recall advantage.

Every result here is from the previously examined, assistant-authored and unreviewed Pilot 04 challenge, using episode-disjoint seeded splits. These are exploratory findings, **not** independent blinded evidence of improved recall, biological engrams, or character identity. The source narratives and neural encoder choices, not just brain wiring, strongly condition outcomes.

## Long-term data preservation

[Archival workflow](../../.github/workflows/associative-archive-historical.yml) and [archiver script](../../scripts/archive_associative_artifacts.sh) downloaded the original completed Actions artifacts, validated JSON and committed each full output to an immutable Git-tracked filename. [Successful archival execution 37838615987](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37838615987). These data survive routine GitHub artifact expiration.

For future experiments, preserve new full per-case JSON as a tracked file under `results/associative/runs/` before declaring the milestone complete. Do not rely solely on short-lived Actions artifacts. The existing archiver is an idempotent historical recovery tool, not an automatic archiver for arbitrary future run IDs.

## Resumption

Read [repository research handoff](../../docs/RESEARCH_HANDOFF.md), [anatomical follow-up protocol](../../docs/ASSOCIATIVE_MEMORY_ROI_EXPERIMENT.md), and the latest commit / workflow status before additional model development. Continue with precommitted hypotheses and matched controls rather than post-hoc parameter optimization.

## Active next experiment: train-only FlyWire trace plasticity (pending real results)

[Plasticity Pilot 01 protocol](../../docs/FLYWIRE_PLASTICITY_PILOT_01.md) and [PR #14](https://github.com/Azimn/Pretorius-Connectome/pull/14) add a distinct, explicit edge-weight learning experiment after the frozen whole-brain and MB studies. It uses immutable canonical L1 and **cross-process deterministic train-only shared TF-IDF L2 v2** (PR #15) without replacing BioCircuit's separate 256-dimensional lexical features. The new weight overlay is trained on source narratives only and does not modify original FlyWire synapse counts. Its planned real three-seed MB comparisons include original frozen and rewired graphs, both with matched plasticity, plus lexical baseline and hybrid controls. **No new biological result has been established until the real CI workflow succeeds.** When it does, archive full source-linked per-case results under runs/, interpret the negative or positive effect honestly, and update this index with exact measured numbers and archival run IDs.

## Historical plasticity v1 diagnostic (superseded)

[The completed real MB v1 plasticity run](FLYWIRE_MB_PLASTICITY_V1_PROVISIONAL.md) reported 6/67 correct-and-accepted positive prompts using learned original MB connectivity versus 7/67 learned rewired and 7/67 frozen original, with 10/67 contradictions falsely accepted in each. The model used the subsequently superseded cross-process-unstable lexical feature selector. These numbers are archived as **provisional negative diagnostics**, not as independently repeatable v2 evidence. A fresh v2 run is required before a definitive comparison.

## Archived real MB plasticity Pilot 01 result

[Measured result report](FLYWIRE_MB_PLASTICITY_PILOT01_RESULTS.md) and [permanent original case-level JSON](runs/flywire-mb-plasticity-pilot01-run37847801158.json) are archived from [successful verified main-branch run 37847801158](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37847801158). All observed outcomes remain exploratory on previously inspected prompts.
