# Direct synaptic imprinting Pilot 08: real whole-brain FlyWire result

**Date:** 2026-10-08. **Measured negative outcome on original full FlyWire v783.** Exact [successful actual 139,255-neuron experiment run 37863818653](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863818653) and its [original case JSON plus learned-overlay checkpoint artifact](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863818653/artifacts/11587462757). The original case-level JSON and checkpoint must also be permanently archived after merging the implementation: the matching post-merge main-branch archive workflow is part of this PR.

All SHA-256 values for the four original biological connectivity arrays matched the previously independently verified full-v783 CSR: 139,255 neurons; 15,091,983 aggregate directed pairs; 54,492,922 integer anatomical synaptic contacts; none modified. The only learned state was a separate sparse signed overlay on a subset of existing directed synapses.

Canonical source: 450 first-person reconstructed memories across 27 episodes. Episode-disjoint seeded split 31 imprinted **317** train events, compared external oracle-only event identity for **159** separately held-out last literal cues from that training group; **71** truly withheld-episode test absences. The decision threshold (0.0082783) was fit using separate calibration positives and withheld-episode validation absences, then held fixed across compared models.

| Method on same original whole-brain FlyWire topology | Nonzero learned directed-edge deltas | Correct event identity, top 1 (n=159) | Correct and accepted | False acceptance on absent events (n=71) | Mean target-content cosine |
|---|---:|---:|---:|---:|---:|
| Learned matched cue-content overlay | 25,780 | 0/159 (0.000000) | 0/159 | 55/71 (0.774648) | 0.168620 |
| Shuffled-content overlay, matched procedure | 25,836 | 0/159 (0.000000) | 0/159 | 56/71 (0.788732) | 0.160361 |
| Untrained zero overlay | 0 | 0/159 | 0/159 | 0/71 | 0 |

The learned overlay produced a nonzero 256D numerical output for all 159 positive probes and survived exact source-CSR-bound saved-state reload, yet **did not recover a single correct memory among 317 candidates**. Although its mean cosine to expected lexical content was about 0.0083 higher than shuffled, this small exploratory effect cannot establish real associative memory without meaningful target discrimination. The negative absent-event rejection rates show that a threshold based on output magnitude and oracle ranking is inadequate. The zero-output untrained baseline trivially rejects absences and is not a useful success benchmark.

The model has no source document retriever, narrative table, event ID lookup, or codebook at inference; it produces a signed hashed **lexical** output, not language. The candidate event names are selected **by an external diagnostic codebook only**. This experiment **does imprint content-dependent numerical updates onto existing real biological connectome edges**, but **does not demonstrate recoverable autobiographical identity, semantics, biological STDP, a fly remembering Pretorius, or an autonomous Pretorius**.

## Consequences

Keep this exact configuration as a frozen negative baseline: no tuning its learning rate, random feature allocation, probe set, evaluation threshold or selected neuron count retroactively. The next differently named pilot should diagnose the randomly chosen disjoint pre/post populations and extreme low overlap between eligible synapses, quantify cue/content activity fidelity, and compare source-matched vs shuffled with a more effective topology-aware routing and capacity-matched rewired control. Retain original anatomy, parser, and tested real-hash invariants. New evaluation probes need independent review before confirmatory claims.

The live research tracker is [Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17). See [protocol](../../docs/FLYWIRE_DIRECT_IMPRINT_PILOT08.md) and the independent [real v783 invariance report](../shared_memory/flywire-v783-real-csr-invariance-run37847525950.json). The accompanying archive workflow, after merge, will commit every original real case and source SHA into the permanent Git history.
