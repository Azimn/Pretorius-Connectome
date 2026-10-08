# Pilot 03 executed results: representation challenge

Date: 2026-10-08. **Status: passed engineering execution; lexical performance improved, semantic memory not demonstrated.**

Prespecified implementation: [Pilot 03 protocol](../../docs/IMPRINTING_PILOT_03.md). Exact source: v12 450-event autobiographical JSONL and candidate sidecars, locked to base commit `60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f`. Experimental PR: #3. Original files and prior pilots are unchanged.

**Evidence:** [Successful dedicated CI run 37826379600](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37826379600), [successful foundation suite 37826379781](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37826379781). Full machine-readable output is the `pilot03-imprinting-results` artifact on the dedicated CI run (contains `pilot03.json`, 63,929 uncompressed bytes, SHA-256 `b7ed817c4e95841e913af61ee717c6f740a3d165258416ae38d4fdf893e52bcc`). The artifact can be regenerated using the exact runner and frozen input file hashes; GitHub may expire old artifacts.

## Dataset and protocol

All 450 reconstructed autobiographical events form the frozen source pool. Each trial **imprints only training episodes** and retains held-out calibration and test episodes for rejection. Seed 31 imprinted 317 events, seed 37 imprinted 316, and seed 43 imprinted 310. Each trial used different train, calibration-negative, and test-negative episode groups, with original test-positive record identities separate from calibration-positive records. These splits are episode-disjoint, **not independently deduplicated across narrative clusters**.

The main challenge is a novel query ID with the first two words of a source recall cue swapped. The deletion and deterministic typo challenges reuse the threshold calibrated on swaps. These are constrained lexical manipulations and should **not** be equated with genuine paraphrasing or semantic transfer.

All learned conditions use the identical 512 by 256 fixed synthetic mask, 55% connection density, same random seed, additive Hebbian rule, and opaque SHA-256 narrative fingerprints. Learned frequency weights for `idf_char3` are fitted only on train records, but its external IDF dictionary does add storage beyond the weight matrix. There is no biological FlyWire graph or language generation.

## Measured performance (mean across seeds 31, 37, 43)

Rates are fractions, not percentages. All accuracy uses an oracle codebook outside the learned weights. Thresholds were calibrated without using the test episodes.

| Condition | Swapped cue top-1 | Correct AND accepted | Known acceptance (TPR) | Held-out false acceptance (FPR) | Balanced accuracy | Known-vs-unknown AUROC | Deleted top-1 | Typo top-1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy full cues | 0.270916 | 0.221699 | 0.639411 | 0.601790 | 0.518811 | 0.558272 | 0.1207 | 0.0763 |
| Legacy surface | 0.332591 | 0.262424 | 0.613992 | 0.524176 | 0.544908 | 0.582917 | 0.1314 | 0.0869 |
| Word-token hash | 0.777341 | 0.554747 | 0.603477 | 0.259238 | 0.672119 | 0.731119 | 0.3919 | 0.2499 |
| Character trigrams | 0.677758 | 0.522557 | 0.674926 | 0.272483 | 0.701222 | 0.755262 | 0.3557 | 0.4490 |
| Train-only IDF trigrams | 0.779491 | 0.675484 | 0.804807 | 0.306262 | 0.749273 | 0.817247 | 0.4235 | 0.4803 |
| Shuffled narrative targets (word tokens) | 0.000000 | 0.000000 | 0.669897 | 0.292799 | 0.688549 | 0.731364 | 0.0000 | 0.0021 |
| Unmodified mask | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.500000 | 0.0000 | 0.0000 |

**Non-neural lexical word-overlap reference:** swapped-cue top-1 `0.883389`, deleted-cue top-1 `0.535963`, typo-cue top-1 `0.434161`. This system retains stored cue strings, does no abstention, and is not a matched-cost neural comparator.

**Per-seed swapped-cue top-1:** full legacy: 0.301887, 0.272152, 0.238710. Token: 0.823899, 0.746835, 0.761290. IDF trigrams: 0.805031, 0.772152, 0.761290. **IDF unknown false-acceptance range:** 0.250000 to 0.387097. **Full legacy false-acceptance range:** 0.264706 to 0.830986.

## What improved

Across these *paired Pilot 03 seeds*, IDF-weighted character trigrams identified 77.9% of the trained events from swapped-word cues, compared with 27.1% using original full-cue hashes. It also improved typo identification to 48.0% from 7.6% on the original full-cue condition. Straight word-token hashes identified 77.7% of word-swapped queries; this is **expected by construction** because token bag encoding deliberately ignores order. Neither is proof of learned meaning.

Calibration also improved: IDF trigrams reached 0.749 mean balanced accuracy and 0.817 AUROC against withheld episodes, versus 0.519 and 0.558 with legacy full cues. However, IDF still falsely accepted about **30.6%** of absent memories, and only about **67.5%** of known swapped cues were both accepted and identified correctly. Strong top-1 without calibration cannot be treated as reliable agent recollection.

## Critical counterexample: absent-memory discrimination is NOT sufficient evidence of correct imprinting

**The shuffled-target word-token control got zero correct IDs but still produced AUROC 0.731, virtually identical to the correctly paired token model's AUROC 0.731.** Similarly, the shuffled control's calibrated balanced accuracy (0.689) exceeded the correctly paired token condition (0.672). This means the rejection head is substantially sensitive to **feature familiarity/occupancy** even when cue-to-event associations have been deliberately corrupted. A high AUROC or acceptance TPR therefore cannot be interpreted as memory identity learning.

To claim useful memory association, require **correct-and-accepted event identification above shuffled association** while jointly reporting false acceptance, not just known-versus-unknown discrimination. The new encoders satisfy that narrower condition on these lexical challenges, but an unconstrained semantic reconstruction test remains missing.

## Limits and next gate

The representation improvement is an exploratory synthetic result, not a preregistered clinical-style validation or independent experiment. No new independent human-written paraphrases were evaluated. Only deterministic surface perturbations were tested. Because model-selection now has access to results on these episode/test seeds, **do not retune on these same test partitions and continue to call them unseen**.

The next legitimate step is a frozen external challenge set, created and held out *before* further development, with editorially reviewed paraphrases, negated or ambiguous near-cues, confusion across related events, and independently authored decision-context labels. Record the absence-rejection trade-off alongside lexical retrieval and shuffled-target controls. Revisit synaptic learning dynamics and bounded interference separately; the FlyWire connectome is still an untested topology, not a substitute for evaluation.
