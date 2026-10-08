# Pilot 01: executed result record

**Recorded:** 2026-10-08  
**Head commit:** `10ab7b29f3471d1166555f0de49fe1efe1285813`  
**Frozen input baseline:** `60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f`  
**CI:** [Imprinting pilot 01, successful run 37822625023](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37822625023); [Foundation tests, successful run 37822624944](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37822624944). Full seed-level output: the downloadable `pilot01-imprinting-results` artifact on the pilot workflow run.

## Exact protocol

450 autobiographical event narratives across 27 episodes; 1,463 literal surface cues and 1,526 additional candidate associative cue IDs, **not** pairwise aligned. Fixed synthetic bipartite mask, 512 cue units, 256 fingerprint units, mask density 0.55, additive masked Hebbian overlay. Seeds 0, 1, 2. Common episode-balanced ordering. Loads 50, 100, 200, 450. One original source cue per trained memory as a partial-cue probe, and a separate oracle codebook decoding to narrative fingerprints. All v12 autobiographical input files are unchanged.

| Load | Learned top-1 mean (range, 3 seeds) | Shuffled top-1 mean | No imprint | Exact cue-ID lookup | Learned mean target cosine | Fixed-50 top-1 | Fixed-50 target cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 1.000 (1.000 to 1.000) | 0.020 | 0.000 | 1.000 | 0.721 | 1.000 | 0.721 |
| 100 | 0.990 (0.980 to 1.000) | 0.010 | 0.000 | 0.970 | 0.618 | 1.000 | 0.619 |
| 200 | 0.978 (0.975 to 0.985) | 0.000 | 0.000 | 0.965 | 0.488 | 1.000 | 0.489 |
| 450 | 0.852 (0.847 to 0.858) | 0.000 | 0.000 | 0.896 | 0.346 | 1.000 | 0.351 |

Unrounded values and per-seed observations are in the immutable workflow artifact; rounded display here is not the underlying measurement.

## Interpretation and confounds

The learned overlay strongly separates from shuffled assignments and an unmodified mask in this *trained-cue fingerprint identification assay*. The learned network's identification accuracy declines with the growing candidate pool, to about 85.2% at 450. A conventional exact cue-ID retrieval reference is about 89.6% at that load. This comparison is illustrative, not matched for resources or information access.

Early memories remain identifiable at 100% with their decoder's original 50 competitors. Nevertheless, their mean target cosine declines from roughly 0.721 to 0.351. Therefore, **do not claim zero synaptic interference or no forgetting** from the unchanged top-1 alone. The code's additive updates visibly dilute the early signal in latent space even when identities remain separable.

All queries use a cue exposed during imprinting; no held-out semantic generalization, unfamiliar cue recovery, free-response reconstruction, abstention calibration, or learned behavioral choice has been tested. SHA-256 narrative fingerprints are opaque random-looking labels, not semantic representations. The evaluator holds the answer codebook outside the imprinted overlay. The current mask is synthetic and feedforward; it is neither a recurrent whole-brain simulator nor the FlyWire v783 connectome. This is a valid software-learning result under the declared assay, **not proof of neural autobiographical memory or persona identity**.

## Next gates

Pilot 02 should disaggregate identification degradation from representation interference by reporting cosine and margin trajectories for fixed target sets, include absent-memory rejection and grouped leave-episode-out probes, measure surface-only versus supplemental-ID ablation, and test paraphrases held out before tuning. For a behavioral claim, train and evaluate an explicit, independently sourced decision/choice task without exposing the target answers at inference. Only then compare the plasticity mechanism against verified FlyWire sampled topology and the separate 4,096-unit baseline. Continue expanding to 500/1,000 records only as optional scaling conditions.

Reproduce with `python scripts/run_imprinting_pilot.py --seeds 0,1,2 --loads 50,100,200,450 --output results/imprinting/pilot01.json`.
