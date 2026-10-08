# FlyWire v783 whole-brain associative retrieval: executed three-seed result

**Date:** 2026-10-08. **Status:** successfully executed, exploratory and post-hoc. **Research finding:** no top-1 retrieval advantage of real biological connectivity over its rewired target-stub control.

**Execution:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836107318

**Machine-readable case artifact:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836107318/artifacts/11574989379

## Frozen data and conditions

The source is the 450-event Pretorius v12 corpus and the previously reviewed by this same project, assistant-authored 68-case Pilot 04 challenge. Cases are **not independently authored, blinded, or human-validated**. Episode-disjoint splits are evaluated on seeds 31, 37, 43. No new test prompts were authored for this experiment, and model settings were not adjusted using these three outputs. Source corpus checks and official verified FlyWire v783 acquisition completed in CI.

The imported directed CSR used 139,255 neuron IDs and 15,091,983 aggregated connectivity entries; it is not a single giant dense matrix. Anatomical synapse counts became positive-only row-normalized log1p diffusion weights. The feature-to-neuron map remains a generic deterministic lexical hash, **not a biologically interpreted map**. The shuffled control is a target-stub permutation preserving unweighted input/output stub counts, not exact biological motifs or weighted in-degrees.

## Primary outcomes, pooled by case count across three seed-specific partitions

There are **67 evaluated true paraphrases**, **67 matched contradictions**, and **20 absent-episode probes** summed across seeds. Since source memories can appear in different seed splits, pooled observations are not all independent. The percentages below are descriptive, not confidence intervals.

| Retrieval configuration | True top-1 | True correct and accepted | False acceptance of contradictions | False acceptance of absent memories |
| --- | ---: | ---: | ---: | ---: |
| Narrative TF-IDF | 22.4% | 7.5% | 23.9% | 5.0% |
| Real FlyWire graph | 22.4% | 9.0% | 34.3% | 15.0% |
| Real FlyWire hybrid | 22.4% | 7.5% | 26.9% | 10.0% |
| Rewired null graph | 22.4% | 9.0% | 26.9% | 15.0% |
| Rewired null hybrid | 22.4% | 7.5% | 26.9% | 5.0% |

### Top-1 accuracy by seed

| Seed | Tested positive prompts | Lexical | Real FlyWire graph | Rewired graph |
| --- | ---: | ---: | ---: | ---: |
| 31 | 23 | 17.4% | 17.4% | 17.4% |
| 37 | 20 | 20.0% | 20.0% | 20.0% |
| 43 | 24 | 29.2% | 29.2% | 29.2% |

### Paired retrieval evidence

The full FlyWire graph and rewired graph differ in predictions on **1 of 67 positive probes**, and neither produces net gains in *correct and accepted* positive memories versus the other across these three seeds. Against lexical retrieval, graph-only changes several predicted IDs, so the propagation step is active, but these shifts do not improve the gross top-1 ranking accuracy.

A separate graph-only calibration difference yields one more accepted correct positive in seed 37, **also present in the rewired-null graph**. This is not an advantage attributable to biological topology. The real full graph generates more false confirmations of contradictory prompts than both the lexical baseline and the null under this calibration.

## Conclusions and limitations

**Supported conclusion:** at the frozen design and evaluation settings, the real FlyWire v783 graph offers **no measured positive top-1 recall advantage** over a simple narrative TF-IDF index or a target-stub-rewired graph. This specific topology is not useful as a simple arbitrary-neuron hashed representation for the current prompts.

**Not supported:** claims that FlyWire is useless for all memory architectures, that anatomical connectivity reproduces autobiographical identity, or that these post-hoc results are independent replication. Positive-only graph diffusion is not neurotransmitter-sensitive learning, plasticity, or recalled language.

The next defensible experimental change is **anatomically constrained inputs and/or learning**, tested against capacity-matched randomized controls. A lightweight first ablation can use the publisher's neuropil tags to isolate mushroom-body synapses; the tracked [ROI protocol](../../docs/ASSOCIATIVE_MEMORY_ROI_EXPERIMENT.md) specifies this. It must not be presented as a functional mushroom-body simulation. Only after explicit ROI evaluation should we add neuron-class-specific encoding.

**Decision gate:** preserve the lexical baseline as a usable system while treating the biological layer as optional and unproven. Avoid repeatedly expanding a whole-brain simulation when no topology-specific benefit is observed.
