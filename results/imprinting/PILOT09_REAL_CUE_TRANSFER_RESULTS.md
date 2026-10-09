# Pilot 09: exact-cue vs heldout-cue retrieval on frozen real FlyWire

Original saved learned synapses from Pilot08 run 37863818653.
Real diagnostic workflow: [run 37868913430](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37868913430).
Complete exact original per-case JSON: [permanent result](runs/direct-flywire-imprint-pilot09-run37868913430.json).
Raw SHA256: a89e4bcb282015374d88282ea9898e90036ef00f31da86f3fc246e6e7a54940d
Frozen learned checkpoint SHA256: e6edf8dd68e84540140612dbcbc42f827e00ce707d9f2947adf3342e215fcb6d

Original real 139,255-neuron, 15,091,983-edge biological synapse arrays were unchanged.
Seed 31 uses same 317 imprinted memories, 159 paired trained-memory probes, and 71 absent-episode test probes. External codebook used only after inference.

| Synaptic condition | Correct on trained cue | Correct on different literal cue | Correct and accepted on different cue | False acceptance of absent episode | Mean cross-cue feature cosine |
|---|---:|---:|---:|---:|---:|
| Original learned overlay | 0.289308 | 0.000000 | 0.000000 | 0.774648 | 0.007124 |
| Same-graph shuffled content | 0.006289 | 0.000000 | 0.000000 | 0.788732 | 0.007124 |
| Original zero overlay | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.007124 |

Fraction of trained-memory cues whose training versus distinct literal test input had no shared hashed feature coordinates: 0.899371.

**Boundary:** This is retrospective diagnostic evaluation of a frozen source-bound biological-connectome weight overlay, not new learning or independent confirmation of semantic generalization. The system cannot generate autobiographical prose, and event identities require an external oracle-only candidate-content dictionary. Original source recall cues are not independently human-reviewed semantic paraphrases.
