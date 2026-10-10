# Pilot16 MNEMOSYNE: measured SYNTHETIC-generated connectivity result

**NOT biological whole-FlyWire evidence.** Original complete source synthetic workflow [GitHub Actions 38025454741](https://github.com/Azimn/Pretorius-Connectome/actions/runs/38025454741), synthetic original artifact ID **11659028456**, full source **7,649,102-byte case JSON SHA-256 `20ab356ad62b8d85053e54b7809f2d53552adf62aeb15dc2d42a99fc549bb123`**, plus all ten actually learned source model state checkpoints. The generated graph contains **8,192 ARTIFICIAL neurons** and **8,332 artificially eligible directed source/readout slots**, which become **7,828 distinct 256×256 feature-pair trainable matrix positions** upon structural collapse. This artifact is **not the actual original publisher FlyWire v783** anatomical source.

The source fictional Pretorius 450 events/27 episodes, original 317 source train event IDs, original first/middle/last source literal cues (3 × 317 = 951), original signed BC01 narrative CONTENT targets, 159 older familiar source probes, original 31 genuinely never presented fourth source literal cues, 62 validation-episode absent cases and 71 heldout source episode-absent test cases remain unchanged. Original 317 CONTENT event-codeword oracle is external to model inference. The frozen local MiniLM 384→256 semantic input is exactly source pinned from Pilot12, pretrained ONNX SHA **`6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452`**, not a test-fitted language model.

## Measured longitudinal retained-source identities

| Model arm | First 16 source IDs correct after 16 learned events | First 16 correct after 317 | Newest 16 correct after 317 | Original 159 familiar source positives correct | 31 never presented fourth cues correct | Accepted correctly familiar after native validation-only gate | False accepted test episode-absent after native gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original BC01 source/whole-synthetic-graph direct Hebbian | 5/16 | 1/16 | 0/16 | 8/159 | 0/31 | No native gate | No native gate |
| Original MiniLM source top-8 whole-synthetic-graph additive | 5/16 | 0/16 | 0/16 | 2/159 | 0/31 | No native gate | No native gate |
| Mnemosyne **dense MiniLM masked approximate RLS**, original artificial feature support | 8/16 | 1/16 | 0/16 | 3/159 | **1/31** | 2/159 | 12/71 |
| Same dense MiniLM masked RLS, degree-rewired artificial graph | 6/16 | 0/16 | 0/16 | 2/159 | 0/31 | 0/159 | 12/71 |
| Same dense masked RLS, RANDOM equal-count feature pairs | 7/16 | 1/16 | 0/16 | 1/159 | 0/31 | 0/159 | 12/71 |
| **Unmasked, dense 65,536-scalar W non-neural online ridge/RLS** | **16/16** | **2/16** | **5/16** | **26/159** | **1/31** | **3/159** | **12/71** |
| Original semantic fixed orthogonal rotation/top-64 sparse masked approximate RLS | 7/16 | 0/16 | 0/16 | 4/159 | 0/31 | 3/159 | 15/71 |
| Original dense semantic input, masked DELTA error correction | 1/16 | 0/16 | 0/16 | 2/159 | **1/31** | 1/159 | 12/71 |
| Original dense semantic masked approximate RLS, WRONG source-CONTENT pairing | 0/16 | 0/16 | 0/16 | 2/159 | 0/31 | 0/159 | 12/71 |
| Original lexical BC01 DENSE cue, masked approximate RLS | 5/16 | 1/16 | 0/16 | 5/159 | 0/31 | 2/159 | 9/71 |

### Main interpretation and parameter-accounting caveat

The **unmasked linear/RLS** model outperformed the original additive/old-top8 semantic baselines on familiar and newest source-cue top1 in the synthetic experiment. This is a learned, cue-only 256D content predictor, not a hidden source-event ID or biography nearest-neighbor lookup. It needs **65,536 independent learnable W feature-pair positions plus 65,536 additional inverse-input-covariance P numerical scalars**. The source artificial-fly-masked RLS has only **7,828 W pairs** plus the *same* 65,536 P scalars. Unequal learnable representation capacity, not fly-specific quality, is a material confound. At the actual original full-FlyWire source this allowed-pair count will be measured afresh and will not be assumed to equal 50,920 original anatomical directed neuron-level slots.

The original **1/31** truly untrained source-cue correctness for semantic error-correction conditions is a *single possible improvement* over prior 0/31 source results but **far too weak to claim meaningful semantic generalization**: these probe items have been seen in exploratory analysis across earlier pilots, and even a random guess over 317 source memories has nonzero probability of producing one accidental match. There was no independently authored new heldout semantic cue set.

**Open-world failure remains:** each model-native input familiarity gate was calibrated on only **62 validation-absent source episodes** (empirical threshold at most 6/62 validation false accepted), without using the original 159/71/31 source test sets. The strongest familiar-cue **26/159** original unmasked RLS result collapses to only **3 correctly accepted of 159**, while falsely accepting **12/71** truly absent test episodes. This is NOT a useful internal novelty detector yet. No source target codebook enters the gate, so the failure is a genuine model-internal confidence/feature-coverage limitation rather than a lower original external scorer threshold.

This synthetic result suggests that preserving full semantic cue vectors can make error-correcting learning more effective than aggressively sparse learned source input on a non-neural substrate, but **does not show that the original fly biological wiring helped** or that the chosen representation/learning rule can preserve autobiographical memories at scale. Direct true fly-v783 source verification and original-mask comparisons are a completely separate computational/experimental gate. Even if those yield similar gains, this experiment has no narrative decoder, physical fly plasticity, typed Kenyon-cell physiology or authenticated Pretorius consciousness.

Original protocol and source lead references: [FLYWIRE_PILOT16_MNEMOSYNE.md](../../docs/FLYWIRE_PILOT16_MNEMOSYNE.md). Canonical program: [Artificial Life Research Journal](https://github.com/Azimn/Artificial-Life-Research-Journal/blob/main/journal/2026/2026-10-09-mnemosyne-associative-memory-review.md).
