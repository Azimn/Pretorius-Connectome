# Pretorius direct synaptic memory research: cumulative FlyWire scorecard

**Research state: October 8, 2026.** This document reconciles the frozen negative and positive experimental observations across direct-imprint pilots, keeping each study's actual inference and source-event meaning explicit. The **original 450-event/27-episode reconstructed autobiography is held constant**; reconstructed narrative text is not independently verified history or biological memory.

## Fixed source and biological substrate

The original whole-brain FlyWire release v783: 139,255 neural root IDs; 15,091,983 aggregate directed connection pairs; 54,492,922 integer anatomical synaptic contacts. All four CSR array SHA-256 values and original publisher MD5/size are pinned by the completed original whole-brain [shared-input anatomical immutability assay](../results/shared_memory/flywire-v783-real-csr-invariance-run37847525950.json). Learned state is stored in a **separate versioned numerical synaptic overlay on existing directed edges**: the fly's source anatomical arrays are never edited. Pilot08's original trained weights and Pilot10's original weights are both permanently committed in [artifacts/imprinting/checkpoints](../artifacts/imprinting/checkpoints/).

Canonical source data are explicitly versioned L1 and the original stateless BC01 hashed **256-dimensional lexical** feature encoder. These are *not learned semantic embeddings* or weights ported from BioCircuit. Neural inference is cue-only and returns a 256D numerical vector. The event identity, accepted event and text descriptions are produced only by an **external oracle candidate scorer** using source-linked training-memory targets; this cannot be described as an autonomous natural-language/autobiographical recall ability.

## Original measured evidence by test design

| Study | Learned synaptic state | Positive cue condition | Source-linked original real top-1 result | Shuffled-control result | Original absent-event false acceptance |
|---|---|---|---:|---:|---:|
| [Pilot08](../results/imprinting/PILOT08_REAL_V783_RESULTS.md) | 317 original train source events; 25,780 changed original edge weights | Entirely distinct, **untrained** last literal recall cue (159 probes) | **0/159** | **0/159** | **55/71** |
| [Pilot09](../results/imprinting/PILOT09_REAL_CUE_TRANSFER_RESULTS.md) | Exact frozen original Pilot08 checkpoint | Original familiar concatenated training cue (same 159 cases) | **46/159** | **1/159** | **55/71** |
| [Pilot09](../results/imprinting/PILOT09_REAL_CUE_TRANSFER_RESULTS.md) | Exact same frozen checkpoint | Distinct **untrained** last literal recall cue (same 159 cases) | **0/159** | **0/159** | **55/71** |
| [Pilot10](../results/imprinting/PILOT10_REAL_MULTICUE_RESULTS.md) | 317 train events, **three explicitly source-authored cues per event**, 28,938 changed original directed edges | First original literal cue, **trained** (159 cases) | **25/159** | **1/159** | **56/71** |
| [Pilot10](../results/imprinting/PILOT10_REAL_MULTICUE_RESULTS.md) | Same trained overlay | Last original literal cue, **trained** (159 cases) | **20/159** | **0/159** | **56/71** |
| [Pilot10](../results/imprinting/PILOT10_REAL_MULTICUE_RESULTS.md) | Same trained overlay | Last trained cue with mechanically deleted last word / character (159 cases) | **10/159** | **0/159** | **56/71** |
| [Pilot11](FLYWIRE_DIRECT_IMPRINT_PILOT11.md) | The exact frozen original real Pilot10 checkpoint (SHA-256 fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a) | **Genuinely untrained fourth original source literal cue**, only for events having 4+ literal cues | **Real source run pending** | Not retrained | **New calibration and heldout test pending** |

**Comparability caution:** Pilot08/09 uses the original combined training cue, one learned event presentation and rate 0.7; Pilot10 uses three distinct cues with three presentations at rate 0.7/3 each. The positive cue conditions and learned synaptic states therefore differ. A higher score for familiar cues does not indicate successful transfer to novel cues, and numbers must not be treated as one continuous progress curve. The archive contains case-level data and original event IDs; this table provides a protocol-aware index, not a pooled statistical analysis.

## Causal insights, bounded by the evidence

The original one-hop imprint changes synapses only for presynaptic BC01 coordinates activated by the training cue. A distinct cue activating none of those original memory's presynaptic coordinates cannot recover that event's *own* synaptic contribution; any nonzero response from it originates from learned deltas associated with other events. Pilot09 measured **143 of 159** paired first-versus-last source cues with *zero* overlap in selected active lexical input coordinates. Explicit multiple-cue imprint in Pilot10 creates some identity-specific readout under **previously trained** cues. It does not overcome absent-event false acceptance and does not show learned semantic equivalence.

The original frozen Pilot10 checkpoint and case-level output were archived after successful main-branch biological CI; an extra real v783 experiment cannot be substituted with the synthetic test. [Pilot11 PR #25](https://github.com/Azimn/Pretorius-Connectome/pull/25) now tests truly **untrained fourth original source cues** and separately calibrates an external similarity threshold using the 158 non-test train events and 62 validation-episode absences, then freezes it for the original 159 test positives and 71 absent test episodes.

**Synthetic preliminary Pilot11 only:** [rehearsal run 37870200081](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37870200081) used a generated topology, not FlyWire: zero of 31 truly unseen fourth-cue source prompts were correctly identified, 30/31 with zero own-event feature-coordinate overlap. A threshold selected using only validation reduced absent false acceptance **62/71 to 8/71**, but correctly accepted trained cues **8/159 to 1/159**. These numbers are *not biological findings*. [Full synthetic report](../results/imprinting/PILOT11_SYNTHETIC_PROVISIONAL.md). Only the separately source-verified original full-brain job can supply the biological Pilot11 numbers.

## Research program gates, not just prototype count

First, **unseen-cue transfer** must beat appropriate source-cue and shuffled pairing controls on an *untrained* representation, not just a repeated trained cue. Second, **true absence rejection** must be measured against the 71 heldout test-episode absences without threshold tuning on that set and paired with correctly accepted known memories. Third, a separately versioned **support/degree/capacity-matched real rewired graph and nonbiological control** are required before attributing any difference to fly-specific connectivity rather than an arbitrary sparse mask. Fourth, cue validity must be checked by independent reviewers, with exact source provenance, before semantic or confirmatory claims. Finally, the existence of a source-linked numerical association does not establish identity continuity, self-model maintenance or the definitive Pretorius persona in The Doctor Lives.

### Method to continue

Keep all previous learned weights, source and event partitions immutable, and keep full source-identified case data plus actual synaptic checkpoints in permanent Git evidence. Update this scorecard with real Pilot11 measured results only after its actual whole-v783 job passes, and link the exact source workflow, full JSON, source SHA and calibrated gate. Never promote a synthetic result to a biological result or an external oracle-assisted label to an autonomous neural recollection.

Research tracker: [Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17).
