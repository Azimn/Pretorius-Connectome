# Pilot 05A: executed post-hoc narrative-grounding diagnostic

**Recorded:** 2026-10-08. **Status:** experiment executed, all tests green, substantive result mixed and insufficient for verified recollection.

**Evidence:** [Dedicated Pilot 05 successful run 37829454893](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37829454893); [foundation suite success 37829455020](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37829455020). An earlier successful experiment run [37829365826](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37829365826) published the complete `pilot05-narrative-results` artifact. Per-case predictions, scores, calibrated thresholds, and retrieved evidence snippets are in the JSON artifact; numbers below are actual program output from that artifact.

## Immutable data and central qualification

The v12 autobiography remains frozen at **450 reconstructed events across 27 episodes**. Event and annotation Git blob SHAs are checked against Pilot 01's pinned values. The Pilot 04 challenge Git blob `1e1c6f0273519bf3cd5868404ddfa5f783df6abd` is also verified at runtime. There are **68 assistant-authored and not independently reviewed prompts**, paired across **34 specific source events**. These questions were already examined in the preceding Pilot 04 study.

**Pilot 05A is a POST-HOC diagnostic, not a fresh independent generalization benchmark.** No results from this challenge should be cited as evidence of out-of-sample semantic validity. The selection of narrative-grounded methods was motivated by observed Pilot 04 failure. No prompt labels, correct IDs or target anchors entered neural or lexical model fitting, but the experimental design is not blind to the benchmark's existence.

The corpus is split into the same imprinted, validation-negative and test-negative episode groups as Pilot 04, with seeds `31,37,43`. Event counts: training 317/316/310; test-positive known paraphrases 23/20/24; matched trained counterfactuals 23/20/24; test-episode absent-event paraphrases 6/9/5.

## Changed intervention

The original experiments learned from only short recall cues, then associated those cue vectors with an opaque SHA-256 narrative fingerprint. **Pilot 05A supplies the complete narrative TEXT as training input** to a document word/character TF-IDF representation, or lets a conventional BM25/TF-IDF retrieval system retain full documents. The neural model keeps the original 512-cue by 256-fingerprint-unit synthetic topology, fixed 0.55 mask density and masked Hebbian update rule. Its decoder still requires an external oracle narrative-fingerprint codebook.

No free-form verbal memory can be generated from neural weights. BM25 and TF-IDF retain full documents and an index, so they are not matched-storage baselines. We used existing `rank-bm25` and `scikit-learn`, not new handwritten retrieval engines.

A prespecified abstention threshold per method was calibrated **only from the original frozen event `decisions` statements** for imprinted events and unknown validation episodes. Neither true nor false assistant-authored Pilot 04 prompt was used to fit vectorizers, neural weights, or thresholds. Decision text is an out-of-index source field, not the challenge label.

## Main outcomes, descriptive unweighted means of three seeds

Rates are fractions, not percentages. "Correct and accepted" requires **both** correct event identity and acceptance above the threshold. "Counterfactual false acceptance" measures any accepted known-false description, even if the chosen event ID differs.

| Method | True paraphrase top-1 | True correct-and-accepted | True accepted (TPR) | Contradiction false acceptance | Contradicted target falsely confirmed | Absent-event false acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-text BM25 | 0.298068 | 0.163527 | 0.608696 | 0.626932 | 0.329831 | 0.529630 |
| Full-text word TF-IDF | 0.134541 | 0.120652 | 0.551932 | 0.467029 | 0.105193 | 0.577778 |
| Full-text character TF-IDF | 0.090097 | 0.056763 | 0.629348 | 0.672222 | 0.000000 | 0.514815 |
| Word TF-IDF input to masked Hebbian network | 0.089493 | 0.056159 | 0.400846 | 0.375241 | 0.058937 | 0.451852 |
| Character TF-IDF input to masked Hebbian network | 0.014493 | 0.014493 | 0.773913 | 0.702899 | 0.014493 | 0.685185 |
| Word input, deliberately shuffled narrative targets | 0.000000 | 0.000000 | 0.487319 | 0.489734 | 0.000000 | 0.477778 |
| Unmodified synthetic mask | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Prior Pilot 04 cue-only neural model:** 0.000000 top-1 for the IDF-character condition on these authored paraphrases. This is context only; Pilot 05 changes the information provided at train time and therefore is a different architectural condition.

### Calibration drift warning

BM25's threshold was fitted using **decision statements** as positive calibration queries, where it accepted only 17% to 25% of those known training events. On the previously authored challenge it accepted about 61% of true paraphrases but also 63% of contradictions. Character-TF-IDF retrieval's calibration accepted only 2.2% of known decisions in one seed and 74.1% in another. Neural character association had validation false acceptance from 58% to 96%. These fluctuations indicate considerable score-distribution shift and unstable generalization of calibrated thresholds.

A zero-imprint overlay rejects all inputs, which attains perfect nonacceptance of contradictions and absent events but provides zero recall. A model must be assessed on useful **correct-and-accepted recall together with false acceptance**, never rejection in isolation.

## Interpretation

**Full narrative access helps substantially but does not solve the key problem.** A conventional full-text BM25 index identified nearly 30% of true paraphrases while the narrative-input word Hebbian overlay identified about 9%. Word TF-IDF retrieval identified roughly 13.5%. These results show that direct text access is a valuable engineering baseline; the specific tested associative overlay does not outperform it.

**More importantly, retrieving a thematically related event is not determining the truth of the proposition.** BM25 incorrectly accepted 62.7% of known counterfactual questions and 53.0% of new descriptions from unlearned episodes. In 33% of contradicted event prompts it accepted and specifically returned the event being misrepresented. Its evidence snippet is a real extract from the stored source narrative, but this does not make the question true, nor verify entailment.

The current neural associative pathway still uses the **opaque narrative fingerprint target and external oracle**. It has no independent narrative-language decoder, calibrated entailment detector, ability to negotiate conflicting testimony, or behavioral-choice mechanism. This is not neural direct synaptic imprinting of an actual FlyWire connectome and not evidence of stable persona identity.

The new results are on an already studied set of 34 selected autobiographical events, only 20 to 24 true trained-event questions per seed. The author of the challenge viewed the originals, and nobody independently audited semantic equivalence. The unweighted seed mean and its small denominators are descriptive and should not be framed as publication-ready confirmatory accuracy.

## Research decision and next gate

The next stage should keep **full-text source provenance** and add a distinct **evidence-verified answer or abstain** decision. Merely selecting a relevant narrative is insufficient. Build an explicit policy that checks whether the retrieved passage actually supports, refutes, or leaves the question unanswered; this policy must operate on the retrieved text, **not an exposed benchmark label**. Log contradictory evidence and allow explicit uncertainty. A compact pretrained NLI cross-encoder or equivalent local zero-API model could be tested as an *explicitly separate architecture*, with deterministic controls, model/version hashes, and compute limits.

**Before any confirmatory claim**, commission or independently review a *new* case set partitioned by narrative clusters as well as episode; freeze it before model selection, fitting or threshold tuning. Include paraphrases, deceptive near-cues, negative/negated facts, and decisions with alternatives. Benchmark evidence verification plus BM25 and word TF-IDF **before** seeking advantage from biologically constrained topology.

Do not modify or re-evaluate Pilot 04 with newly tuned parameters while labeling the original cases untouched. This research line remains a measurable software engineering experiment, with positive and negative outcomes preserved.
