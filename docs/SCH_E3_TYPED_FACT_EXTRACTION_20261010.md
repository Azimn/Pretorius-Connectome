# E3-T1 Clerk: source-attested typed extraction and immutable-memory integration boundary

**Date:** October 10, 2026. **Experimental owner:** [Attractomancy](https://github.com/Azimn/Attractomancy/tree/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY). **Canonical source owner:** Pretorius-Connectome.  
**Status:** Cross-project research finding and staged architecture recommendation. No production implementation, source-memory update, trained model change or new lived Pretorius event.

## Why this matters

Recent D1/E1/E2 experiments distinguished (a) retrieving the correct source record, (b) selecting multiple relevant autobiographical records, (c) extracting actual content, (d) computing the required action, and (e) enforcing source ownership and temporal eligibility. The corrected synthetic E3-C v2 fixture showed three small models could be given both needed records and still fail the paired counterfactual decisions. The follow-up **Clerk** program decomposed reading and deciding rather than adding another symbolic cue.

## Completed exploratory results

All three models ran original, verified source-local synthetic extraction prompts. The 12 complete-state scoring set consists of three artificial two-input rules, not the actual Pretorius L1 events.

| Model | Direct model decisions | Joint two-source field extraction + software | One-source-at-a-time cached extraction + software |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | 5/12 | 3/12 | **10/12** |
| Qwen2.5-1.5B | 6/12 | **9/12** | 6/12 |
| SmolLM2-1.7B | 5/12 | 4/12 | **8/12** |

On individually isolated source documents the models supplied **11/12**, **8/12**, and **10/12** strictly correct attested field values respectively. A separate deterministic software calculator then combined the source-verified values; the number of fully correct output flips under one changed fact was 5/6, 2/6, and 3/6. An exact source-only regex would succeed on all these fixed-format fixtures by construction. **These figures measure a hybrid model-plus-software pipeline, not independently learned causal reasoning by the models.** Different prompt templates, strongly simplified source clauses and source-value reuse preclude a direct efficacy estimate of a memory-cognition architecture.

The interesting engineering result is a **model-specific reversal**: smaller Qwen and SmolLM2 benefited from separate source prompts, whereas 1.5B Qwen extracted more accurately with both records together. A proposed universal cognitive extraction strategy would be premature. An actual acting character should be able to swap source parsing strategies after held-out testing without rewriting its autobiographical database.

## Versioned fact cache

A separate [experimental SQLite implementation](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/verified_fact_cache.py) keys a source-attested derived field by subject, source, event ID, record version and hash, extractor identity and schema. Synthetic tests validate persistence across reopen, rejected same-version forks and stale rollback, prompt-model value disagreement, field-schema isolation and invalidation of an amended permission record. The software drill begins with synthetic \`ALLOW\` plus \`KEEP\`, then adds a newer \`REVERSE\` amendment: the old cached fact is unavailable until the newly revised source value is extracted and verified, after which the decision becomes \`DENY\`. [Passing unit tests and drill](https://github.com/Azimn/Attractomancy/actions/runs/38028840190) do **not** establish authenticated external provenance, real interpersonal consent or a psychologically informed permission system.

## Recommendation to source owner

Preserve the existing 450 canonical reconstructed L1 memories and owner-provided verified source reader, the original BioCircuit 256-dimension lexical cache, FlyWire retrieval results, and neural/connectome artifacts. Do not infer a new \`lived\` event from synthetic source updates. A later, separately reviewed adapter can expose typed **source witnesses** with \`subject_id\`, \`event_id\`, \`source_manifest\`, \`record_version\`, \`content_sha256\`, \`extracted_field\`, \`evidence_span\`, \`extractor_version\` and \`eligibility_status\`, provided an independent evaluation shows it helps real-L1 retrieval and interpretation.

For genuine deployment the interface must distinguish authenticity from checksum consistency, enforce current relationship/commitment state against stale versions, and measure the cost of refreshing fields versus model calls on each interaction. Neither experiment authors nor the source owner should promote the best of the tested model-specific approaches without held-out real evidence and blinded source relevance judgments.

**The full E3 Tribunal blinded trial remains unexecuted**, and the toy Clerk experiments make no claim of machine subjective memory, continuity or consciousness. Results and raw model responses are maintained only in Attractomancy.


## October 10 follow-up: source-owned L1 direct witnesses

The subsequent [E3-T2 source-field protocol](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/L1_FIELD_PROTOCOL.md) selected twelve original L1 events from twelve different episodes and tested two **existing author-written source metadata fields** per event: belief change and relationship change. These values already exist in the original reconstructed-fiction source. The [experimental read-only witness adapter](https://github.com/Azimn/Attractomancy/blob/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/l1_witness_adapter.py) allows software to recover them directly and annotate exact evidence spans in the original canonical verified record.

A [passing 24-witness source audit](https://github.com/Azimn/Attractomancy/actions/runs/38058594338) verified one witness per field/event with source checkout, source-manifest SHA, source record digest, canonical full-content digest, exact character offset, value hash, \`L1v1\` snapshot class and reconstructed-fiction provenance. It rejected intentionally modified witnesses and forged claimed record text. This is **24/24 exact retrieval by ordinary software construction** without model inference; it cannot be credited as learned cognitive recollection or independent truth verification. The 450 original source events were read, not rewritten.

A first small-model L1 field-copying pilot was **confounded by the response prompt**: the 0.5B model repeated the literal JSON example \`"source text"\` in 70/72 completions, rather than source field values. The original raw results are preserved and this is diagnosed as *prompt-example copying*, not a general inability to use original L1. A separately labeled corrected v2 removes the exemplar and must be scored on its own original outputs. The parallel 1.5B v1 and v2 jobs should not be assigned scores before their actual artifacts are verified.

A separate [unlabeled narrative review packet](https://github.com/Azimn/Attractomancy/tree/main/experiments/SCH_E3_TRIBUNAL_OF_MEMORY/packets/l1-semantic-calibration) mixes source-authored belief/relationship interpretations with claims from other source events. It contains 24 narrative cases and 72 proposed interpretations, **zero independently obtained ratings**. Its original author-written sidecar statements are hypotheses about a memory, not certified semantic ground truth. An independent reviewer must evaluate whether each statement is supported, contradicted, or underdetermined by the full narrative. Even successful literal field copying would not establish autobiographical reasoning or fidelity to Pretorius's relationships.

**Action:** Do not add another duplicative permanent vector or encoded field store to this source repository merely to copy existing \`belief_changes\` or \`relationship_changes\`. Reuse source-owned direct structured access. A future optional adapter can preserve event/manifest provenance and evidence spans while delegating unresolved narrative semantics to an independently evaluated cognition path. The production L1, source-owned caches, FlyWire links and neural weights remain unchanged.
