# Pilot12: synthetic-only semantic cue × degree-matched rewiring control

**Nonbiological preliminary evidence, source-linked and explicitly exploratory.** [Successful synthetic run 37871790784](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37871790784), original artifact pilot12-SYNTHETIC-only-six-conditions, complete case-level JSON and locally trained neural overlay checkpoints. This is a generated 8192-neuron degree-16 aggregated directed graph **not** the actual original FlyWire v783 topology. The full publisher-verified biological experiment is a separately gated CI job.

Across all six methods, the source corpus, train/test episode split, 317 source autobiographical events, source cue exposures, BC01 256D **content** targets, fixed neuron-to-feature readout, learning rule and external oracle-candidate scorer were held identical. The only experimental factors were original BC01 vs pinned **external** pretrained MiniLM sentence cue encoder, original vs degree-preserving rewired eligible directed connections, and matched versus shuffled source-content association.

| Condition | Familiar explicitly trained last literal cue, correct top1 (N=159) | Genuinely untrained fourth original source cue, correct top1 (N=31) | Absent test episodes falsely accepted under validation-only gate (N=71) | Nonzero learned synaptic edges |
|---|---:|---:|---:|---:|
| Original synthetic graph + original BC01 | **8/159 (5.03%)** | 0/31 | 8/71 (11.27%) | 4,737 |
| Degree-switched synthetic + BC01 | **9/159 (5.66%)** | 0/31 | 8/71 (11.27%) | 4,740 |
| Original synthetic + pinned MiniLM cue input | **2/159 (1.26%)** | 0/31 | 8/71 (11.27%) | 6,538 |
| Degree-switched synthetic + MiniLM | **1/159 (0.63%)** | 0/31 | 8/71 (11.27%) | 6,558 |
| Original synthetic + MiniLM shuffled content | **1/159 (0.63%)** | 0/31 | 8/71 (11.27%) | 6,622 |
| Degree-switched synthetic + MiniLM shuffled content | **1/159 (0.63%)** | 0/31 | 6/71 (8.45%) | 6,615 |

**Preliminary observation:** Pinned MiniLM was *worse* than the original lexical cue encoder on the synthetic substrate for familiar source cue event identification, and neither encoding retrieved any of the 31 untrained fourth-source-cue events. Rewiring original synthetic eligible connections did not produce any clear benefit for content identification. This is a negative result, not a claim of pretrained semantic failure generally: the model uses a fixed random 384→256 projection and selects only 8 signed input coordinates, which may suppress semantic geometry.

**Important limitations:** This result does NOT use the actual biological FlyWire v783 connectivity and does not show anything specific about fly neuroanatomy. A success on the real-v783 verification must be reported separately. The external pretrained transformer provides a language prior not learned in the fly synapses. The neural model returns 256D signed input-to-content associative features only; memory event identity and gate calibration are EXTERNAL oracle diagnostics. The cues were preexisting, short recall surfaces and not independently human-reviewed semantic paraphrases.

Follow [Pilot12 preregistered protocol](../../docs/FLYWIRE_DIRECT_IMPRINT_PILOT12.md) and [cumulative research Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17). Never change Pilot12 parameters in response to the synthetic results without creating a separately versioned Pilot13 intervention.
