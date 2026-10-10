# E3 real L1: direct source witnesses versus narrative-grounded relationship continuity

**Date:** October 10, 2026. **Experimental owner:** [Attractomancy E3](https://github.com/Azimn/Attractomancy/tree/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY).  
**Canonical source owner:** Pretorius-Connectome; original 450-event reconstructed-fiction L1 at pinned commit \`6d2768211f5c2184c8bbdb833c06e169b5137197\`, manifest SHA-256 \`ce23717b3950a772cc25e4bf60107eaddf4a0375122b6893c15417669ae033bb\`.  
**Change scope:** research note only. Original L1, sidecar fields, associative/semantic/BioCircuit/FlyWire caches, model weights, relationship state and production runtime unchanged.

## E3-T2 literal field-access findings

Each original L1 event carries separate source-authored \`belief_changes\` and \`relationship_changes\` metadata strings. Twelve existing episodes were sampled, one event per episode, and both fields were selected to produce 24 literal reference targets. Each target was presented to Qwen2.5-0.5B and 1.5B under target-only, full-event and withheld-target presentations (72 prompts/model). Two prompt versions were executed and preserved as four independent run/model datasets.

The original v1 included the literal example \`{"answer":"source text"}\`, leading the smaller model to echo the dummy example in **70/72** responses. This contaminates an interpretation of v1 as an ability test. Without the example, v2 also scored poorly on strict JSON partly due to wrong JSON keys, Markdown code fencing and string \`"null"\` in place of actual \`null\`. These output-format failures should not be conflated with content retrieval: exploratory post hoc v2 response inspection found that the full original target text appeared under some JSON key in **19/24 target-only and 16/24 full-event** cases on Qwen0.5B, and **24/24 and 23/24** respectively on Qwen1.5B. The primary original strict JSON scorer was **not rewritten** to accept these answers.

The [tested direct L1 witness interface](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/results/l1-witnesses-38058594338/summary.json) recovered 24/24 original field strings by ordinary software without an LLM and blocked both tampered witness and falsely claimed source identity in the test. Its exactness is **true by construction** relative to the original source data; SHA-256 content consistency is not third-party historical authorship authentication. For any event where the structured field is already available, direct source access is preferable to model-driven text regeneration.

A separately versioned [plain-text ablation](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/L1_PLAINTEXT_PROTOCOL.md) replaced JSON with a one-line exact field response and FIELD_ABSENT missing-evidence marker. Its Qwen0.5B results show **12/24 exact isolated**, **2/24 exact full context**, and **0/24 correct withholding**. Larger-model and independent-family plain-text results belong only in their individually verified run artifacts, and the current source-owner note makes no unsupported claims about them.

## E3-T3: The Book of Debts

The next cognitive requirement is **temporally coherent relationship evidence**: can a model reconcile old and new source experiences without treating chronological recency as automatic revocation?

A [source-derived, currently unlabeled packet](https://github.com/Azimn/Attractomancy/tree/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/packets/l1-relationship-t3) contains 24 paired chronological excerpts from eight recurring characters in canonical L1: Clara Weiss, Marta Voss, Jakob Lenz, Mathilde Rosen, Anna Lenz, Emil Reuter, Anton Kappel and Henry Frankenstein. The cases are correlated (three overlapping pair comparisons per character). Reviewer-facing excerpts omit original event IDs and author-written relationship-change sidecars. This masking is procedural because public source lookup could reconstruct them.

[Review validity software](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/l1_relationship_adjudicate.py) rejects unquoted affirmative claims, recency-only supersession interpretations, incomplete verdicts and fewer than three independent disclosed reviewers; its unit tests are synthetic fixtures, **not human annotations**. Zero genuine independent relationship judgments are recorded. Some original relationship changes may remain open to competing interpretations even after review, so multiple perspectives and explicit uncertainty should be preserved.

## Implications for future memory access

1. Maintain source-owning typed fields and immutable event IDs as an engineer-visible archive; keep renderer output and episodic lived interaction records in a separate track.
2. For an event field already explicitly present, prefer direct, checksum-verified source witness access over regeneration by a language model.
3. For *unstructured* interpretations, use independent evidence reviews, narrow direct source quotations, conflict-aware chronology, and multiple acceptable interpretations. Never treat original author sidecars as blind semantic ground truth.
4. A later relationship event should update or supersede specific commitments **only with evidence for change**, not merely because its source timestamp is more recent.
5. Proceed to renderer inference only after real source-dependent relational test cases and judgments are frozen. Require earlier-only, later-only, both, none and foreign-source blocked controls, distinct model families and stable source costs. Any deterministic action gate is a tested safety property, not model learning.

**Status:** The full independently authored and multi-source-adjudicated Tribunal of Memory (E3) has not been performed. These read-only source-access studies do not validate a lasting first-person subject, persistent AI autobiography or biological connectome transfer.
