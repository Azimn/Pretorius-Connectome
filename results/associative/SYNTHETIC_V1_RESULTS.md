# Connectome associative retrieval v1: executed synthetic diagnostic

**Date:** 2026-10-08. **Status:** reproducible demonstration executed successfully. This is NOT a FlyWire result or a blinded study.

**Run:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37833271857
**Artifact:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37833271857/artifacts/11574053559

The experiment used the frozen 450-memory autobiography, seed 31 episode partition, previously examined Pilot 04 authored questions, and an explicit 512-node random directed graph fixture with eight directed outgoing contacts per node. Positive and contradictory probes each have 23 cases; the withheld-episode group has six cases. Training and calibration use episode-separated data. The original benchmark is assistant-authored and unreviewed, so these figures are exploratory post-hoc diagnostics only.

| Configuration | Positive top-1 | Correct and accepted | Contradiction false acceptance | Absent false acceptance |
| :--- | ---: | ---: | ---: | ---: |
| Narrative TF-IDF (no graph) | 0.173913 | 0.130435 | 0.608696 | 0.166667 |
| Synthetic graph only | 0.043478 | 0.043478 | 1.000000 | 1.000000 |
| Synthetic hybrid | 0.130435 | 0.130435 | 0.782609 | 0.500000 |
| Permuted-target synthetic null graph | 0.043478 | 0.043478 | 1.000000 | 1.000000 |
| Permuted-target synthetic null hybrid | 0.130435 | 0.043478 | 0.608696 | 0.166667 |

**Interpretation:** This synthetic configuration offers no observed recall advantage over the plain lexical control. Graph-only performance is lower on positive paraphrases and accepts all contradictory and unseen-episode prompts in this specific run. The full per-case machine-readable artifact should be inspected before changing any assumptions.

The real FlyWire graph was not part of this trial. These results cannot support a claim that the real biological topology helps or harms retrieval. They also do not establish that retrieved passages logically entail queries. Next inference requires actual verified FlyWire graph execution, matched nulls, and a new pre-registered, independently reviewed challenge.

All six newly added unit tests passed, and the repository-wide 56-test foundation suite passed on code commit f4aa7c6f69872e039bdc08bb349413ade38d87ad. No persona identity or biological synaptic memory conclusions follow from these tests.
