# SCH E2: Multi-memory retrieval and action integration, source-owner note

**Date:** October 9, 2026, America/Chicago (GitHub CI UTC October 10).  
**Canonical archive source:** this repository's pinned \`6d2768211f5c2184c8bbdb833c06e169b5137197\` L1 export, 450 fictional reconstructed memories in 27 episodes.  
**Experimental code and evidence:** [Azimn/Attractomancy E2](https://github.com/Azimn/Attractomancy/tree/main/experiments/SCH_E2_MULTIMEMORY_DECISIONS) and its [consolidated results](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E2_MULTIMEMORY_DECISIONS/RESULTS.md).  
**Status:** Research documentation only; **not** a code or memory mutation authorization.

## Motivation

E1 established that source-verified, durable external memory can be accessible without yielding reliable characteristic decisions. E2 asks two separable questions: (a) whether a retriever can assemble multiple useful autobiographical records from a new dilemma, and (b) whether a language renderer receiving the correct pair can combine them to choose an appropriate response.

## Retrieval findings: small, author-selected six-case fixture

Six investigator-authored dilemmas were associated with two hand-picked L1 event IDs each. The research program's existing source-owned lexical retrieval implementation (\`pretorius_connectome.associative.AssociativeMemory\`) was used as the baseline. Every retrieval condition searched the same 450 canonical source narratives with the same top-2, top-5 and top-10 budgets.

| Search method | Both hand-picked events at top 10 | Individual hand-picked events at top 10 |
| --- | ---: | ---: |
| Lexical TF-IDF | 0/6 | 2/12 |
| Source-authored event-link diffusion with 75% lexical restart | 1/6 | 3/12 |
| Shuffled association links, seeds 11, 19, 37 | 0/6 each | 2/12 each |
| Local MiniLM sentence embeddings | 0/6 | 3/12 |
| Unweighted reciprocal-rank fusion of lexical and MiniLM | 0/6 | 4/12 |
| Permuted dense-to-event assignments | 0/6 | 1/12 and 0/12 |

The source graph contains 912 authored \`links_to_prior_events\` references. These are **narrative annotations, not biological connectivity**. Three-step diffusion moved the second required source record for exactly one question, \`E02-013\`, from rank 21 to 10. It did not recover both sources in the other five dilemmas. The shuffled-link negative controls did not achieve the one positive retrieval, but a six-query fixture provides no robust statistical evidence of graph superiority.

The pretrained local MiniLM variant \`sentence-transformers/all-MiniLM-L6-v2\` was encoded with mean pooling, 384 output dimensions and no weight updates. No source record exceeded its 256-wordpiece cap. Its stronger individual-event recall and fusion's additional single hit did not produce a complete two-memory evidence set. These small gains do not justify replacing the verified lexical cache or changing production retrieval.

**Evaluation limitation:** The twelve "required" source events were chosen by a single investigator reading this same source archive. Alternative events may be equally or more relevant; no independent multi-evidence relevance annotations exist yet. All scores are exact source-ID recall, not semantic completeness or subject-level identity accuracy.

## Renderer findings

Two Qwen2.5 sizes each completed **72** guarded, independent-context conditions: two original source memories under editorial cues or neutral keys; either source alone; no source; and a wrong-subject record blocked upstream. A/B answer positions were counterbalanced.

Qwen2.5-0.5B matched the unblinded author-chosen interpretation in **3/12** complete-editorial and **2/12** complete-neutral cases; it usually abstained with missing information. Qwen2.5-1.5B output \`UNKNOWN\` on **72/72** cases, including the full-evidence arms. This perfect abstention in insufficient-source controls does not indicate successful action integration: the model refused to decide even when both sources were supplied. Both models showed no reliable cue-specific benefit over information-identical neutral lookup keys.

The question prompts themselves contain normative information, and the author choices favor cautious/prosocial options. A [separate forced-choice calibration](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E2_MULTIMEMORY_DECISIONS/FORCED_PROTOCOL.md) compares the same dilemmas with original two memories against no and unrelated memories. Its **completed** results resolve the over-abstention interpretability question but **do not** show a source-memory benefit. On the same six dilemmas with A/B order counterbalanced, Qwen2.5-0.5B matched the unblinded author preference in **8/12** original-full, **8/12** single-record, **8/12** unrelated-pair, and **7/12** no-memory cases. Qwen2.5-1.5B scored **11/12** original-full and unrelated-pair but **12/12** without *any* source record. Editorial cues versus neutral lookup keys generated **identical 12/12 paired choices** for each model. The correct source records added no demonstrated causal decision benefit on these normatively loaded author-selected vignettes, while consuming nearly four times the no-source input/output tokens (approximately 705 versus 179). Do not use these author-label rates as evidence of actual Pretorius identity continuity.

The original 1.5B all-UNKNOWN behavior under a conservative source-eligibility instruction was a prompt-induced total-abstention pattern rather than a general inability to produce a choice. Both extremes must be controlled.

## Reproducibility caution

The lexical ranking order in the MiniLM-run baseline differed for several queries from the E2R and E2G source-only runners, although canonical manifest and question-fixture hashes matched. Aggregate lexical recall@10 did not change. The [completed five-stage parity audit](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E2_MULTIMEMORY_DECISIONS/results/parity-run-38024737940/parity_audit.json) found that lexical score vectors are *exactly stable within each audited process* before and after torch import, thread adjustments, model load, and full MiniLM embedding inference. But the audit's rank order matched E2S rather than earlier E2R/E2G, even with identical source and fixture hashes. Thus the rank discrepancy survives and its cross-run root cause is not established. Numerical environment pinning and deterministic score/tie auditing are needed before reporting exact source-rank reproducibility.

## Architecture recommendations

1. Keep **retrieval candidate discovery**, **verified provenance and source class**, **evidence-set assembly**, and **action eligibility** separate. A 450-event archive is not useful for a novel decision if the relevant memories never reach the renderer.
2. Preserve immutable source event IDs, L1 hashes and original source-owned training/test split boundaries. Never mix researcher-authored synthetic overlay experiences with the reconstructed fictional Pretorius canonical autobiography.
3. Treat editorial cues as alias metadata, not intrinsically causal vectors. Prior D1/E1 neutral-key null results remain constraints. Source-authored links are not neuronal synapse semantics.
4. Before production changes, create independently blinded source relevance judgments permitting *multiple acceptable source sets*, run stronger independently developed model families, and test contradictory updates, relationship histories and abstention versus action.

**No production changes are being requested or reported by this note.** No L1 record, source-owner cache, FlyWire CSR, neural weights, associative retrieval defaults, or canonical character state was changed in carrying out E2.


## Prospective validity upgrade

The [E3 Tribunal of Memory study design](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/PROTOCOL.md) responds to E2's weaknesses with independently blinded relevance annotations, multiple acceptable evidence sets, decisions counter to convenient moral defaults, temporally versioned synthetic revocations, cost-matched record subsets, and independent model-family replication. **It is not executed.** Production Pretorius-Connectome interfaces should remain unchanged until such evidence passes validity and architectural audit.
