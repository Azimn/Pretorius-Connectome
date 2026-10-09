# Pilot12 actual original FlyWire: frozen semantic cue × rewired null

Original [publisher-verified real whole-v783 run 37872284939](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37872284939).
Complete unchanged [source case JSON](runs/flywire-pilot12-real-v783-run37872284939.json).
Original raw JSON SHA-256: aef42672a4cff9127fe03640f5287f06bb5d2fc44c510e6de0ef22e7bbe363d1
Pinned external pretrained MiniLM revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ONNX weight file SHA-256: 6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
Projected 384→256 feature matrix SHA-256: 4bf51db082d616fffa8a12d26540b92566a1c4ba4983238d89eda0e8571b32c2
Control switch seed: 73; changed target destinations: 49962

Same 317 train source events, 951 first/middle/last cue presentations, 159 familiar test positives, 31 matched truly untrained source fourth cue positives and 71 episode-absent negatives across SIX source-paired conditions. Every original v783 root ID, CSR index and integer contact was unchanged. Null edges were constructed in an isolated in-memory graph.

| Condition | Changed learned edges | Familiar-cue top1 | Genuinely unseen fourth-cue top1 | Heldout absent false accept (validated gate) |
|---|---:|---:|---:|---:|
| real_bc01 | 28939 | 0.125786 | 0.000000 | 0.070423 |
| degree_rewired_bc01 | 28995 | 0.132075 | 0.000000 | 0.084507 |
| real_minilm | 40387 | 0.000000 | 0.000000 | 0.084507 |
| degree_rewired_minilm | 40378 | 0.006289 | 0.000000 | 0.183099 |
| real_minilm_shuffled | 40461 | 0.000000 | 0.000000 | 0.070423 |
| degree_rewired_minilm_shuffled | 40445 | 0.000000 | 0.000000 | 0.140845 |

**Interpretation boundary:** The MiniLM input is a pretrained external language-model prior; the fly model trains ONLY sparse signed source-edge synaptic deltas while preserving original BC01 narrative CONTENT coordinates. All memory/event rank decisions and accept/reject gates are external oracle diagnostics. The null is matched in binary degree and number of eligible writable directed edges, not contact-weighted target input strength or actual nonzero post-training weights. Original 31 fourth cues are unreviewed lexical source fields, not independent human-written semantic paraphrases. No biological plasticity, native-language autobiographical retrieval, independent subject or topology-specific cognitive function is claimed.
