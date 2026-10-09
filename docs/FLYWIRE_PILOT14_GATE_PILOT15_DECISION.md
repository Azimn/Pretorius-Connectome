# Pilot14 result gate and prospective Pilot15 decision

**Research decision recorded after actual verified real full-v783 Pilot14 run 37882541412.** Owning repository: [Pretorius-Connectome](https://github.com/Azimn/Pretorius-Connectome). Canonical cross-project chronology: [Artificial Life Research Journal](https://github.com/Azimn/Artificial-Life-Research-Journal/blob/main/journal/2026/2026-10-08-flywire-pilot14-synaptic-stability.md). This is not a source retuning memo; Pilots08–14 remain immutable.

## What actually happened

The original FlyWire-derived real synaptic model was re-trained with the **same source v12 450-memory corpus**, identical 317 original 3-cue training events, 159 familiar source positives, 31 source-original truly unseen fourth cues, 71 never-trained episode-absent queries and the original BC01 lexical input/target content encoder. Pilot14 compared β0 additive source learning, β1 and β4 local per-synapse usage protection, two binary-degree-preserving rewiring controls, deliberately wrong source targets and a **non-neural parameter-count-matched** original-source cue→content linear mapping. Full actual [original biological run 37882541412](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37882541412) passed and reproduced **exactly** the historical Pilot10 original original-edge delta array and all 159/71 historical case decisions, without changing any original FlyWire anatomical contact.

| Core preregistered metric | Original additive | Original β=1 | Original β=4 | Rewired β=1 | Nonneural masked linear |
|---|---:|---:|---:|---:|---:|
| Same earliest 16 source memory IDs correct after all 317 | 4 | 8 | 10 | 11 | 7 |
| Newest 16 original source memory IDs correct at final training | 2 | 1 | 0 | 0 | **6** |
| Familiar source 159 test-positive correct ID, trained literal cue | 20 | 36 | 36 | 39 | **46** |
| Truly untrained fourth literal source cues correct, 31 | 0 | 0 | 0 | 0 | 0 |
| Original historical threshold false-accepted 71 episode-absent | 56 | 60 | 57 | 59 | **52** |

These numbers come from an **external source codeword matching oracle**, not a trained internal natural-language readout or autobiographical narrative retrieval. The fly-derived numerical synapse model and the non-neural linear comparator both use only the original cue string when called for inference, and return a 256D signed content vector. The source identities/target candidate matrix never enter model inference.

## Decisions supported by data

**Synapse-stability mechanism isolated in this model:** scaling later directed-edge writing by 1/(1+β·prior_uses) protects some earlier source-correct associations at the expense of source-new learning. The stronger β4 arm is therefore NOT the recommended new default memory system despite outperforming β0 on old-source ID retention: it fails at newest-16 acquisition (0/16). The mechanism works similarly in rewired original-degree-controlled connections; there is **no isolated causal advantage of original Drosophila anatomical wiring** over binary-degree-matched rewiring. Original v783 source biology remains immutable; no biological learning law has been demonstrated.

**Non-neural comparator remains stronger for new material:** the 50,920-trainable-slot masked linear map achieved 6/16 newest and 46/159 familiar source ID matches. This is a stronger result than any original FlyWire model on both new source learning and full familiar recall, but with different contact-weight and signed update-magnitude budgets. Match controls more strictly before an academic causal claim; do not claim that FlyWire is innately inferior universally.

**Unseen-cue and absent-episode bottlenecks remain unsolved:** 0/31 novel original source fourth cues in all tested arms and at least 52/71 absent wrong accepts. Changes that improve near-verbatim familiar associations cannot be represented as semantic persona transfer, autonomous narrative memory, or effective source trust/rejection.

## Prospective Pilot15: two causal axes, not a post-hoc β sweep

Do **not** tune β against the already-examined source 159/71/31 cases. The next independently versioned experiment should specifically test:

1. **Balanced stability/plasticity:** local homeostatic synaptic write rules or multiple routing compartments that regulate interference while preserving new-event acquisition, with simultaneous evaluation of the identical earliest source 16 and newest source 16, and both correct and shuffled source content controls. An increase in first-16 retention is not an improvement unless the youngest 16 also learn.
2. **Content readout/internal decoding:** compare the original source-edge readout with a learned low-rank non-ID-specific vector readout, and a non-neural scalar-parameter and gradient/update-budget comparator. Model inference may use its own learned numerical readout weights but MUST NOT accept a source event ID, candidate record set, text database, frozen narratives or source-codebook nearest-neighbor lookup as part of the model. Do not smuggle the external oracle into neural inference.
3. **Generalization and novelty:** commit to new **independently human-authored, source-episode-unseen cue prompts and independent negative evidence** before training, with appropriate blinding. Source-original fourth cues from Pilots11–14 have been reused and should be labeled exploratory, not new validation.
4. **Mechanistic measures:** per-source-neuron and per-readout-neuron synaptic slot occupancy; per-feature source cue collision; direct signed source-target margins and preservation of first-16 original case IDs across 0–317 load; confidence/rejection precision and recall jointly; source graph versus binary-/contact-weight-matched null; a CPU/cost budget matched sparse non-neural baseline. Report all models' actual nonzero trained parameter counts, per-event signed update operations, total norm changes, and exact/within-tolerance original source replay.

**Clear next-go/no-go gate:** only promote a stable neural memory approach into The Doctor Lives if, on independently reviewed source and temporal splits, it preserves *both* earlier and new-event autobiographical content under load, generalizes meaningfully to untrained cues, and does not hallucinate absent events at unacceptable rates. Pilot14 by itself satisfies none of those three product-level requirements.

## Documentation and preservation checklist

- [x] Protocol committed before original result: [Pilot14](FLYWIRE_DIRECT_IMPRINT_PILOT14.md).
- [x] Original publisher-verified real biological computation passed: [Actions 37882541412](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37882541412).
- [x] [Source real outcome and all controls](../results/imprinting/PILOT14_REAL_V783_PROVISIONAL.md) committed to original project main.
- [x] [Cross-project canonical journal result](https://github.com/Azimn/Artificial-Life-Research-Journal/blob/main/journal/2026/2026-10-08-flywire-pilot14-synaptic-stability.md) updated.
- [ ] Final main-branch publisher-verified rerun successful; confirm source-run ID.
- [ ] Seven original learned NPZ files, including exposure counts and masked non-neural matrix, plus original full raw case-level JSON **permanently committed to Git**, checksum verified by result-archival workflow. Do not infer archival success merely from initial Actions downloadable artifact.
- [ ] Update archived raw result and manuscript-level evidence register with actual permanent source link, exact JSON SHA-256 and documented numerical replay.

Original journal and issue tracking: [Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17). Separate [Pilot13 bounded float32 reproducibility archival correction PR #29](https://github.com/Azimn/Pretorius-Connectome/pull/29) remains independently gated; a claimed Pilot14 source result does not erase that prior anomaly.
