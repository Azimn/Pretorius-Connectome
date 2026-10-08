# Pilot 07A results: a real local NLI model rejects almost all autobiographical claims

**Measured 2026-10-08. Full test suite and real model inference green. Scientific result: negative for useful autobiographical claim verification at frozen thresholds.**

## Exact immutable experiment provenance

- Repo run: https://github.com/Azimn/Pretorius-Connectome/actions/runs/37837161852 . Both "infrastructure" and "real-local-nli" jobs **succeeded**. Raw per-case machine-readable JSON is in the GitHub Actions artifact "pilot07-real-local-nli-results", artifact ID **11575997202**.
- Raw JSON: 804,243 bytes, SHA-256 **7065ed30afee5701e354799ee76cde4b8c7484859ef80d0de94cba449584b29c**. A small tracked permanent summary is at results/imprinting/PILOT07_SUMMARY.json .
- Earlier seed-31 real execution: https://github.com/Azimn/Pretorius-Connectome/actions/runs/37836846908 ; validated that this was an actual pretrained checkpoint and not the synthetic NLI test double.
- Checkpoint: **cross-encoder/nli-MiniLM2-L6-H768**, exact Hugging Face commit **c847a3c0e1cad93a5343183ef183f3044e3fc7c2**. Reported Apache-2.0 license; NLI output label order contradiction, entailment, neutral. 82.1M pretrained parameters. CPU PyTorch **2.5.1+cpu**, Transformers **4.57.6**, NumPy **2.4.6**. Max input 384 tokens; premise-only truncation. Inference was real, not mocked, and uses no paid API.
- Frozen source: Pretorius v12 reconstructed fictional archive of 450 memories, 27 episodes, from exact existing source/sidecar Git blob pins. No original memory was changed. Same episode-disjoint training/validation/test seeds **31,37,43** as Pilots 03–06.
- Frozen challenge: the previously examined **68 assistant-authored, human-unreviewed** Pilot 04 paraphrase and contradiction cases for 34 distinct events, Git blob 1e1c6f0273519bf3cd5868404ddfa5f783df6abd. **NOT a new independent test.**
- Model premises: exact first-person narrative text of BM25-retrieved training events, hypotheses: benchmark query strings. The scorer never sees gold event ID, prompt kind or source anchor during inference.
- Policies frozen before evaluation: BM25 decision-text-calibrated score baseline; NLI on BM25 top 1; NLI reranking BM25 top 3; reject-all control. Entailment accepted if p(entailment) >= 0.60 and at least 0.12 above both p(contradiction) and p(neutral). Possible contradiction is advisory if p(contradiction) >= 0.60 and above p(entailment) by 0.12.

## Three-seed results

Metrics are descriptive averages of three **small, repeated challenges**, not publication-grade independent generalization. Rates below are percentages.

| Method | Correct & accepted true-event recall | All true prompts accepted | Contradictions incorrectly accepted | Absent-event prompts incorrectly accepted | Direct false confirmations of contradicted target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original BM25 calibrated | **16.35%** | 60.87% | **62.69%** | **52.96%** | 32.98% |
| Pretrained local NLI after BM25 top-1 | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% |
| Pretrained local NLI over BM25 top-3 | **1.67%** | 1.67% | 0.00% | 0.00% | 0.00% |
| Reject all | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% |

**In particular, neither NLI policy demonstrates a useful support/rejection balance.** One true memory was correctly accepted by top-3 reranking on seed 37 (1/20 true challenge prompts). The other 66 true probes across these three seed partitions were rejected. The strict top-1 NLI policy rejected every true prompt. Reporting "0% hallucination" alone would therefore be misleading, since always-reject also achieves zero false acceptance.

### Exact per-seed support-versus-rejection outcomes

| Seed | Train events | True challenges | BM25 correct+accepted | BM25 contradictions accepted | NLI top1 correct+accepted | NLI top3 correct+accepted | NLI top3 contradictions accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 317 | 23 | 4/23 (17.39%) | 17/23 (73.91%) | 0/23 | 0/23 | 0/23 |
| 37 | 316 | 20 | 3/20 (15.00%) | 12/20 (60.00%) | 0/20 | 1/20 (5.00%) | 0/20 |
| 43 | 310 | 24 | 4/24 (16.67%) | 13/24 (54.17%) | 0/24 | 0/24 | 0/24 |

BM25 test-episode false acceptance in seeds 31,37,43 was 5/6, 5/9, and 1/5; NLI false acceptance was zero for all those cases. The model saw 156, 147, and 159 premise/claim candidate pairs for the respective seed runs. Source texts are recoverable by each event ID from the unchanged archive.

### Raw top-1 NLI probability diagnostics

Across the 67 trained positive challenge prompts, the NLI model's **top BM25 document** predicted "neutral" for 57 and "contradiction" for 10; **zero** were classified as entailment as their largest probability. Mean probability was entailment 0.0605, contradiction 0.2135, neutral 0.7260.

Across 67 trained counterfactual prompts, the NLI model predicted "contradiction" for 58 and "neutral" for 9. Mean probabilities were entailment 0.0137, contradiction 0.7559, neutral 0.2304.

Across 20 never-imprinted test-episode descriptions, 18 were top-labeled neutral and 2 contradiction; mean entailment 0.0258, contradiction 0.1412, neutral 0.8329.

The NLI cross-encoder seems capable of assigning different probabilities to false-versus-true wording **in this previously inspected and limited sample**, but it does **not establish that it identified the correct source memory or reasoned over all facts correctly**. The top BM25 candidate itself can be the wrong memory. A wrong retrieved premise may naturally be neutral to a true claim.

## Interpretation: why this is not a victory

- The real pretrained NLI model was successfully integrated into a deterministic, reproducible CPU workflow. **That engineering milestone is complete.**
- With current full-narrative premises, truncation, top-K retrieval and frozen acceptance thresholds, the system has **near-zero useful recall**. The drop in accepted contradictions primarily reflects **aggressive abstention**, not proof of reliable truth checking.
- Long autobiography paragraphs may diffuse or truncate evidence; the NLI checkpoint is pretrained on ordinary sentence-pair tasks, not long literary autobiographies or this character's reconstructed history. Retrieval can select a thematically related but wrong story. These are possible causes, not tested causal explanations.
- The three-way score is a *classifier judgment*, never a human evidence gold label or an eyewitness fact. The original fictional corpus itself is reconstructed authorial history, not independently observed historical evidence.
- This run is not actual FlyWire neural synapse plasticity. The NLI verifier operates outside the original synthetic masked Hebbian structure; it does not show that the neural connectome learned meaning, remembered evidence, or acquired identity.
- The same AI-authored Pilot04 prompts have been used in multiple rounds of development; **all Pilot07A figures are explicitly post-hoc exploratory**. No two-reviewer independently validated holdout has been created.

## Research decision and recommended next experiment

1. **Do not relax thresholds on these exact 68 cases and call the result an independent improvement.** Any threshold exploration against them must be labeled post-hoc.
2. Preserve separation between event retrieval, **source-grounded evidence selection**, NLI verdict, and evidence-bound response/abstention. The trained NLI checkpoint should receive a minimally relevant premise that includes the supporting or contradicting clause **without silently omitting important context**. Avoid a one-sentence lexical gate that rejects everything as in Pilot 06.
3. Add a controlled diagnostic of top-1 vs top-K retrieval *candidate recall* (whether the known target lies in retrieved K candidates), since no NLI model can confirm an event absent from the retrieval list. Then compare short evidence windows vs full narrative on **development examples** without calling retuned performance confirmatory.
4. Complete the new external reviewer packet via actual separate human author and two independent annotators plus adjudicator under docs/PILOT07_REVIEW_PROTOCOL.md. The automatically generated CSV files are blank and remain **UNREVIEWED**. A reviewer-set must be frozen and kept blind before confirmatory evaluation.
5. A later project milestone may connect source-grounded claim results to behavioral Pretorius choices and actual neural topology. Neither was measured by Pilot07A.

**Decision:** a real pinned local verifier now exists, but useful autobiographical verified recall remains unachieved. Preserve this negative result and prioritize evaluation integrity and retrieval candidate coverage rather than a larger synthetic mask.
