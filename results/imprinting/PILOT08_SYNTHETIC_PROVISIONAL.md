# Direct FlyWire imprint Pilot 08: synthetic baseline, measured negative result

**2026-10-08. Biological status:** **not the original FlyWire result**. These are the preliminary nonbiological synthetic-control measurements from [CI run 37863818653](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863818653). Exact synthetic per-case JSON and the numerical checkpoint were attached to that run as expiring artifact direct-imprint08-synthetic-only. The original complete FlyWire-v783 study executes in a separate workflow job.

The source was the canonical pinned 450-event and 27-episode autobiography. For seed 31, the test trained on **317** events and evaluated **159** held-out last-cue probes from that training group. The separate absent-event test contained **71** source episodes' untrained records, under source episode splitting. Candidate comparison is external oracle-only BC01 lexical-coordinate identification, not an autonomous model memory lookup.

| Synthetic condition | Nonzero learned edges | Correct top-1 | Correct and accepted | Absent false acceptance |
| --- | ---: | ---: | ---: | ---: |
| Learned synthetic source-coupled content | 4,235 | 1/159 (0.006289) | 0/159 | 41/71 (0.577465) |
| Shuffled source-content pairing, identical synthetic graph | 4,320 | 1/159 (0.006289) | 1/159 (0.006289) | 41/71 (0.577465) |
| No learned synaptic changes | 0 | 0/159 | 0/159 | 0/71 |

The learned model's mean target-content cosine was **0.074475**, versus **0.073324** for shuffled-content training. Calibration threshold was **0.215507**, fitted only on training-cue and validation-absent controls, and applied to the test. The two nonzero models produced a numerical response for each of the 159 positive probes.

**Interpretation:** No detectable meaningful advantage over shuffled-content pairing on the synthetic directed-graph fixture. All three conditions are far from competent event-level associative recall. The little observed correlation may reflect lexical overlap and codebook effects. The two nonzero overlays have similar absent-event false acceptance. None of these values establish a real fruit-fly biological connection result, semantic understanding, language reconstruction or subjective autobiographical recall.

**Next:** Preserve this negative synthetic baseline, examine the real verified 139,255-neuron / 15,091,983-directed-edge biological run as a separate condition, and distinguish storage capacity from generalization across withheld lexical recall cues. Do not alter the Pilot 08 primary evaluation after seeing the synthetic results without explicitly versioning any changed experiment.
