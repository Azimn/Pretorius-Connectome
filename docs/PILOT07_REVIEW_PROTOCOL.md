# Pilot 07: independent semantic-memory challenge review protocol

**Status: PROTOCOL ONLY. No independent human annotation, prompt collection, or model validation has occurred.**

## Rationale

Pilot 04 was written by the assistant after reading the source; Pilot 05A and Pilot 06A then repeatedly evaluated against those same 68 questions. Thus their observed performance is developmental, not confirmatory. Pilot 07 must create a **new and independently reviewed benchmark** *before* any new semantic-inference models are selected or their thresholds optimized. The source narrative remains the frozen v12 collection of 450 reconstructed memories.

## Authoring and isolation

1. Publish a fixed sampling manifest of eligible memory IDs and associated source blob hashes. Exclude Pilot 04's 34 challenged target IDs from the confirmatory test pool. For a clean confirmatory sample, consider excluding near-duplicate narrative clusters, not just episode IDs. Record which clusters overlap and how clusters were determined; if cluster isolation is not possible, explicitly identify the leakage risk.
2. An **external prompt author** (not the model developer) sees selected original event narratives and writes at least one meaning-preserving paraphrase, one explicit negation, one same-entities/different-outcome contradiction and one genuinely unanswerable/nearby-event prompt for each event. Record source evidence and claimed truth value separately. This author's prompts must not be copied into encoder training or retrieval threshold fitting.
3. Two **independent reviewers** with access to the source but without knowledge of the model's predictions independently classify each candidate claim as `entailed`, `contradicted`, `not_enough_information` or `ambiguous_or_invalid`. They also mark whether the claim uniquely identifies its source event, name a minimally sufficient supporting/refuting source span, and flag impossible-to-judge or multiple-event questions.
4. Freeze the two raw review sheets, compute agreement before adjudication, and have a third adjudicator resolve discrepancies and invalid cases. Do not overwrite the two independent raw judgments. Exclude invalid questions from the primary fixed benchmark, but preserve their exclusion counts and reasons.
5. Commit a **versioned, hashed, never-retuned** challenge manifest containing the final cases, gold event/absence category, truth labels, minimally sufficient evidence spans, original reviewer agreement and adjudication provenance. Keep the confirmatory set inaccessible to feature selection, prompt rewriting and score threshold tuning until the evaluation protocol is frozen.
6. Require positive-recall and contradiction-rejection metrics **jointly**, with macro averages by episode and cluster, coverage-risk curves, per-case grounded citations, and explicit source-absence response. Report both selection accuracy and accepted-and-correct answers, not only hallucination rate. Compare BM25, TF-IDF, any NLI module, and masked Hebbian variants against the same source access and compute budgets wherever possible.

## Recommended controlled metadata

| Field | Meaning |
| --- | --- |
| `case_id` | Unrevealing random ID, never a target encoder feature |
| `source_event_id` | Provenance for annotators and evaluator only |
| `author_id` | Anonymous stable external author code |
| `claim` | Natural-language prompt, the ONLY question-side input the model receives |
| `source_blob` | Exact frozen origin version |
| `reviewer_a_label` and `reviewer_b_label` | Pre-adjudication labels |
| `reviewer_a_span` and `reviewer_b_span` | Proposed textual evidence range |
| `adjudicated_label` | Final claim-level conclusion |
| `adjudication_notes` | Explanation for changed or excluded cases |
| `narrative_cluster_id` | Group for withheld related events |
| `partition` | Calibration, dev or final confirmatory test, chosen before model fit |

## Model evaluation gate

An optional local pretrained three-way NLI model may be evaluated **only once the fresh reviewed cases are frozen**; pin checkpoint revision/license, token budget and exact CPU/GPU timings, and preserve a non-NLI retrieval-only baseline. Neither a high retrieval similarity score nor a lexical negation check should be described as logical entailment.

**Human review is not simulated in this repository.** The dataset is *awaiting external prompt authors and validators*. This document provides the procedure rather than pretending that independent review has already occurred.
