# FlyWire MB activity-trace plasticity Pilot 01: actual three-seed result

**Exploratory post-hoc diagnostic**, not an independent confirmatory evaluation.
**GitHub Actions run:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/37847801158
**Permanent complete original case-level JSON:** [original bytes](runs/flywire-mb-plasticity-pilot01-run37847801158.json)
**SHA-256 of original JSON:** 6d5accc57a7f9f22b61a468e23a0bc865f903c6dd7e33db818ac9e19763d99c7

Dataset: 450 frozen reconstructed Pretorius memories across 27 episodes.
Anatomy: verified v783 MB-selected topology, 14025 nodes, 574660 aggregate adjacency entries.
Training: cross-process deterministic source-pinned shared TF-IDF L2 v2, gains 2.0, 256 cap, seeds 31, 37, 43. Historical uncached v1 is a separate representation, not an equivalent encoder.
Biological synapse counts and original CSR edge targets: unchanged in both trained conditions.

## Pooled descriptive observations (cases reused between seeds)

| Condition | True top-1 | True correct and accepted | Contradictions falsely accepted | Absent episodes falsely accepted |
| --- | ---: | ---: | ---: | ---: |
| Narrative TF-IDF | 15/67 (22.4%) | 7/67 (10.4%) | 19/67 (28.4%) | 2/20 (10.0%) |
| Real frozen MB graph | 14/67 (20.9%) | 7/67 (10.4%) | 9/67 (13.4%) | 0/20 (0.0%) |
| Rewired frozen graph | 11/67 (16.4%) | 3/67 (4.5%) | 8/67 (11.9%) | 0/20 (0.0%) |
| Real learned MB graph | 14/67 (20.9%) | 7/67 (10.4%) | 11/67 (16.4%) | 0/20 (0.0%) |
| Rewired learned graph | 11/67 (16.4%) | 3/67 (4.5%) | 8/67 (11.9%) | 0/20 (0.0%) |
| Real learned hybrid | 15/67 (22.4%) | 7/67 (10.4%) | 19/67 (28.4%) | 1/20 (5.0%) |
| Rewired learned hybrid | 13/67 (19.4%) | 5/67 (7.5%) | 20/67 (29.9%) | 1/20 (5.0%) |

## Prespecified plastic-vs-rewired comparison

Real MB learned graph had 7 correct-and-accepted positive cases versus 3 for the rewired learned control (difference +4 across pooled seed observations).
Contradiction false acceptances: real 11/67, rewired 8/67; absent episode acceptances: real 0/20, rewired 0/20.
Isolated plasticity effect: original MB trained 7 versus original MB frozen 7 correct-and-accepted (change +0); matched rewired trained 3 versus rewired frozen 3 (change +0).
**No benefit of training the real MB graph has been demonstrated** on the correct-and-accepted metric; any advantage over the rewired graph must not be misattributed to plasticity when also present before training.

Per-seed paired improvements/losses (source IDs and predictions in permanent JSON):

| Seed | Real minus rewired accepted-correct gains | Losses | Real updated edges | Rewired updated edges |
| --- | ---: | ---: | ---: | ---: |
| 31 | 1 | 0 | 77344 | 90157 |
| 37 | 2 | 0 | 77050 | 90433 |
| 43 | 1 | 0 | 77533 | 89582 |

## Interpretation limitations

The prior 68 prompt cases were assistant-authored, human-unreviewed and repeatedly examined. Pooled observations overlap across seeds; percentages are descriptive. The graph uses lexical input hashing, a train-only coactivity heuristic and a degree/support-matched directed edge-swap null, not experimentally established anatomical cell-class assignments. Accepted lexical retrieval is not entailment, and learned numerical weights are not evidence of experienced memories, consciousness or a stable Pretorius identity. The new matched null preserves unique edge support and stub degrees, but does not preserve weighted in-degree or biological motifs.

A gain here would require independent human-reviewed queries, anatomical cell-type aligned input and stronger degree/class-preserving controls before claiming biological specificity. A null or negative gain is an equally valid experimental finding and must not be hidden.
