# E3 Counterfactual Chamber: source-necessity testing and validity caveats

**Research date:** October 9, 2026 (America/Chicago); Actions runs logged October 10 UTC.  
**Experimental owner:** [Attractomancy E3](https://github.com/Azimn/Attractomancy/tree/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY).  
**Source owner:** Pretorius-Connectome. **This note does not modify the source L1, associative retrieval, model weights, or the acting character.**

## What was tested, and what was not

E2 multi-memory reasoning tests showed that a model can produce the preferred cautious/prosocial option with **no Pretorius memories at all**. A new E3-C synthetic calibration therefore isolates two independent bits in fictional **PRETORIUS_SANDBOX** records. The correct decision explicitly depends on combining the two record values, not on a universally ethical answer. Three policy families are used: packet-seal equality, destination revision, and versioned permission reversal. The test is mechanically deterministic and does not involve the original 450 L1 events.

The first E3-C fixture v1 accidentally encoded both inputs in source event IDs, leaking the withheld fact to first-record-only prompts. It is preserved as a **failed validity control**, not as interpretable two-source causality. The [corrected v2 runner](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/counterfactual_calibration_v2.py) uses event IDs opaque to the nonlocal input and asserts that missing-second-record prompts are identical across hidden-bit flips.

## Verified corrected Qwen v2 outcomes

Both small local Qwen sizes completed 84 distinct execution contexts each with original prompts, generated strings, token counts, deterministic model revisions, subject/source checks and strict output scoring. In the twelve balanced complete-record forced cases:

| Condition | Qwen2.5-0.5B | Qwen2.5-1.5B |
| --- | ---: | ---: |
| Both synthetic records with symbolic-looking cue | 5/12 | 6/12 |
| Same records via neutral key | 5/12 | 6/12 |
| No synthetic records | 6/12 | 6/12 |
| Paired counterfactual questions both correct after a one-bit flip | **0/6** | **0/6** |
| Identical editorial versus neutral outputs | **12/12** | **12/12** |

The output distributions were often fixed by task type and did not consistently reverse with the change of source fact. One-size reasoning or generic output-class selection could score roughly 50% on this balanced toy fixture. The strongest negative metric is that neither Qwen model correctly answered a pair in which *only one required fact had been flipped*. A prompt requiring UNKNOWN with only one source record also failed in **12/12** cases in each model.

The upstream D1 identity check blocked all twelve deliberately foreign synthetic envelopes **before inference**. The renderer then nevertheless often guessed a decision when records were absent. This illustrates the difference between **record admission**, **evidence sufficiency**, and **actual policy action**. The guard is a SHA-256 consistency check relative to source-trusted bytes, not an authorship signature.

These observations are **not** grounds to change the real Pretorius decision engine or conclude that a 1.5B model lacks all multihop inference ability. The questions explicitly contain a toy rule, all inputs are synthetic and the number of distinct source-complete cases is twelve per arm. An independent SmolLM2 model-family calibration has also been launched; its verified result belongs in the experimental owner's report, not in an assumed value here.

## Blinded E3 remains unexecuted

The [main Tribunal of Memory protocol](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/PROTOCOL.md) calls for independently authored novel dilemmas, at least three blinded evidence-relevance raters per case, multiple valid evidence sets, actual source-version conflicts, adversarial prior controls and token/cost normalized model-family comparisons. The [candidate reviewer packet](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/packets/e2-calibration/packet_manifest.json) was validated on six old E2 questions with 127 source excerpts, but those questions were source-informed and **do not** count as independent E3. No human judgments have been collected. The label validation tests are software tests, not substituted human reviews.

**Architectural recommendation:** Preserve the current source-owned L1 and memory adapters. Continue researching evidence-set retrieval, typed source and temporal-state checks, and whether a model can reconstruct values from records. Do not promote synthetic v2 correctness or an externally programmed action gate into evidence of persona memory continuity, subjective self, connectome transfer or symbolic efficacy.
