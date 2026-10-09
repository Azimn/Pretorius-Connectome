# Pilot10: real original FlyWire v783 multiple-cue imprint measured results

Real publisher-verified workflow: [run 37869353939](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37869353939).
Original full per-case JSON: [source evidence](runs/direct-flywire-imprint-pilot10-run37869353939.json).
SHA-256 of unmodified original result: ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8
Permanently committed [learned synaptic checkpoint](../../artifacts/imprinting/checkpoints/pilot10-original-v783-run37869353939.npz).
Learned synaptic checkpoint SHA-256: fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a

Original graph unchanged: 139,255 neurons, 15,091,983 directed pairs, 54,492,922 anatomical integer contacts. Original canonical 450 memories, 317 training events, 159 positive and 71 absent-episode evaluation cases.

| Condition | Imprints | Changed source synaptic edges | First cue correct | Last cue correct | Deletion correct | Absent false acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Explicit three source-literal cue bindings | 951 | 28938 | 0.157233 | 0.125786 | 0.062893 | 0.788732 |
| Repeated concatenated-cue equal exposure | 951 | 25782 | 0.119497 | 0.000000 | 0.000000 | 0.774648 |
| Multiple source cues with deranged content | 951 | 29043 | 0.006289 | 0.000000 | 0.000000 | 0.802817 |
| Unmodified zero-delta baseline | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Interpretation:** The last source literal cue was explicitly trained in the multi-cue condition and is NOT novel semantic paraphrase recall. Deletion-cue probes are mechanical lexical variations, not independent human semantic evaluation. Inference returned a 256D lexical vector; event identity was assigned using EXTERNAL evaluator-only content codebooks. The training exposures and nominal rate budgets match but effective edge counts and neural capacity do not. No rewired biological control, fly physiology, autonomous narrative recollection or cognitive subject is demonstrated.
