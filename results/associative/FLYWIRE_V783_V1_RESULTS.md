# FlyWire v783 associative memory: executed one-seed diagnostic

**Recorded:** 2026-10-08. **Status:** biological connectivity experiment executed successfully, no observed top-1 retrieval improvement at the tested settings. This is a post-hoc, non-blind diagnostic, not a confirmatory experiment.

**Successful full FlyWire workflow:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37833431615

## Data and control conditions

The verified FlyWire v783 CSR import contains 139,255 neuron root IDs and 15,091,983 entries in the converted directed connectivity representation. Synapse counts are retained as anatomical evidence. In the retrieval system, contact counts are transformed with log1p and source-normalized, followed by two rounds of positive-only diffusion. These are chosen numerical rules and are not physiological synaptic efficacies.

The corpus contains 450 immutable reconstructed Pretorius autobiographical events. For seed 31, the episode split yields 23 previously examined, author-written positive paraphrases, 23 paired contradictions, and 6 absent-episode paraphrases in the evaluated test subset. The benchmark is small, post-hoc, assistant-authored, and unreviewed. Retrieval may find a related passage without verifying its truth.

| Condition | Positive top-1 | Positive correct and accepted | Contradiction false acceptance | Absent false acceptance |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF narrative lexical baseline | 0.173913 | 0.130435 | 0.608696 | 0.166667 |
| Real FlyWire directed graph only | 0.173913 | 0.130435 | 0.695652 | 0.166667 |
| Real FlyWire hybrid | 0.173913 | 0.130435 | 0.695652 | 0.166667 |
| Target-stub-permuted null graph | 0.173913 | 0.130435 | 0.652174 | 0.166667 |
| Target-stub-permuted null hybrid | 0.173913 | 0.130435 | 0.695652 | 0.166667 |

The real topology did not improve positive top-1 retrieval or accepted correct recall over the lexical baseline in this one seed. Its contradiction false acceptance was worse. The matched null had the same positive accuracy. No benefit of actual biological wiring has been demonstrated.

## Why this does not settle the hypothesis

Current textual features are hashed uniformly onto 139,255 biological nodes. Their mapping has no known relationship to a fly sensory or mnemonic population, so this may suppress the useful organization of biological connectivity. Two diffusion steps and a 256-node activity cap could further restrict network-level mixing. These are hypotheses, not established explanations of the negative result.

The benchmark does not learn synaptic weights or preserve physiological signs. Its shuffled-stub null conserves basic stub degrees but not all biological motifs or weighted in-degree. Most crucially, the prompts are not independent holdout data. Future work should compare more seeds and case-paired predictions using the same frozen settings, then evaluate any new anatomically constrained encoding using a new preregistered challenge.

**Decision:** do not describe the first real FlyWire execution as improving persona memory. Preserve its negative finding and execute the paired three-seed diagnostic before selecting a new architecture.
