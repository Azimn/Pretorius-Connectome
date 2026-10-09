# Frozen Pilot09 cue-transfer diagnostic: synthetic preliminary result

**2026-10-08. Evidence status: NONBIOLOGICAL synthetic structural-mask fixture only.** [Successful test and original per-case artifact](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37867669141), artifact name direct-imprint09-synthetic-replay. Full real FlyWire v783 original-checkpoint diagnostic is a separate job and must not be conflated with these results.

This diagnostic leaves Pilot08's trained numerical overlay, 317 original source events, model parameters, 159 source-linked train-event probes, 71 episode-heldout absent-event probes, external oracle codebook, and original fixed decision threshold **unchanged**. No parameter search or retraining occurs between train-cue and different-last-cue conditions.

| Condition | Correct using exact cue used at imprinting | Correct using heldout literal cue | Correct and accepted using heldout cue | Absent-episode false acceptance |
|---|---:|---:|---:|---:|
| Original content-linked synthetic synaptic overlay | **16/159 (10.0629%)** | **1/159 (0.6289%)** | 0/159 | 41/71 (57.7465%) |
| Same topology, mismatched content associations | **0/159** | **1/159 (0.6289%)** | 1/159 | 41/71 (57.7465%) |
| Frozen zero-delta overlay | 0/159 | 0/159 | 0/159 | 0/71 |

The same 159 events' **original training input versus heldout last literal cue** have a mean 256D BC01 selected-sparse-feature cosine of **0.007124**. **143/159 (89.9371%)** cue pairs have *zero* overlapping active hashed lexical feature coordinates. This result alone predicts weak transfer even if the network retains some pattern under familiar cues. Original trained-cue mean cosine with evaluator target is **0.303263**, but falls to **0.074475** for the heldout last cue. Same shuffled-control target cosines are 0.194618 and 0.073324.

**Interpretation:** The exact synthetic training cue achieves some identity discrimination above shuffled content (16 vs 0), though absolute retrieval remains poor. Almost entirely nonoverlapping lexical feature inputs account for a plausible **fixed cue-encoding bottleneck** in trying to recall from an unrelated literal source cue. This is not a demonstration of semantic paraphrase transfer or of successful real FlyWire imprinting. The source recall cues are not independently validated semantic variants.

**Next:** Await the separately pinned, publisher-verified **actual whole-brain v783 checkpoint** replay before deciding whether the same storage/transfer distinction applies to the biological topology. Preserve this synthetic baseline as the diagnostic control. Subsequent feature encoder or synaptic routing revisions require a NEW labeled experiment with matched shuffled, zero and eventually degree/capacity-matched rewired biological controls.
