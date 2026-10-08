# Pilot 06A: evidence-bounded memory claims

**Protocol frozen before execution, date: 2026-10-08.** Pilot 06A is a limited post-hoc engineering diagnostic on the **already studied** Pilot 04 questions, not a blind semantic generalization benchmark.

## Research question

Pilot 05A showed that BM25 can retrieve approximately 30% of target events from altered descriptions, but a relevance-based acceptance threshold also accepted nearly 63% of false descriptions. This experiment isolates a small intervention: **require a source-linked sentence evidence match before answering, and optionally reject explicit mismatches in lexical negation**. We measure whether that reduces false confirmation without destroying useful correct-and-accepted retrieval.

No fact is "verified" simply because a sentence has similar words. This module explicitly represents a third state, insufficient evidence, and a narrow **possible contradiction** state. It has no language-model NLI or factual entailment implementation.

## Frozen source and conditions

The 450 first-person reconstructed v12 events and existing sidecars remain immutable. A source commit and Git blobs are enforced at runtime. The 68 assistant-authored, unreviewed Pilot 04 case prompts remain frozen and previously examined. The existing source `decisions` field is read only for threshold calibration.

For each seed `31,37,43`, reuse Pilot 05's episode-disjoint train/validation/test split. Train BM25/full-text TF-IDF and the original 512-by-256 masked word-Hebbian narrative model **solely from training episodes**. Train a supplementary TF-IDF document-term matrix over **sentences from the exact same trained first-person source narratives**; do not read or index evaluation questions or labels when fitting. The evidence index is separate from the neural matrix and adds extra text and vocabulary storage.

Six controls share the same inputs and splits:

| Name | Selection and response policy |
| --- | --- |
| `bm25_calibrated` | Pilot 05A BM25 top-1 with decision-based score threshold, no sentence verification |
| `bm25_sentence_only` | Same BM25 top-1; require sentence match above independently calibrated evidence score, and at least two shared non-stopword anchors |
| `bm25_sentence_polarity` | Same sentence gate, and decline if lexical negative markers disagree with the evidence sentence |
| `tfidf_sentence_polarity` | Train-only word TF-IDF retrieval plus identical sentence and polarity gate |
| `hebb_word_sentence_polarity` | Narrative-input synthetic Hebbian retrieval plus **external source sentence access** and identical gate; therefore a hybrid, not pure synaptic memory |
| `reject_all` | Deliberate always-abstain control |

The evidence snippet is an unaltered substring of the source event text with a stable event ID and sentence index; a 300-character display cap is used in the JSON only. It is a **citation candidate**, not a guarantee that the cited sentence supports the question. The neural + evidence hybrid is particularly not architecture-resource matched to stand-alone neural weights, since it fetches stored original narrative text.

## Thresholds and leakage guards

For each seed and retrieval method, train the textual retriever only on training narratives. Fit and calibrate its retrieval-score threshold and evidence-score threshold **only on source decision statements from train events versus statements from separately held-out validation episodes**. Correct case IDs, case kind, event targets, source anchors, prompt text, or benchmark labels **never enter index building, threshold calibration, or the evidence verifier**.

The evidence gate's two shared non-stopword token anchors and shallow negation set (`not`, `no`, `never`, `without`, `neither`, `nor`) are frozen in source code before tests. They were selected as auditable, deterministic heuristics, not optimized against Pilot 04 outcomes. Their effects on idioms, quantification, tense and unexpressed negation are unknown and likely limited. A lexical match does not entail a story claim. A polarity mismatch is only a **possible** contradiction.

For evaluation, report both top-1 source event identification (regardless of whether it answers) and **correct AND accepted true-event retrieval**. Report general false acceptance and direct false confirmation of a known contradictory query, plus acceptance of queries about never-imprinted test episodes. Report abstention on genuine positives to prevent an always-reject control from appearing strong. Additionally record the count of contradicting prompts explicitly labeled "possible contradiction" by the heuristic, and publish each event quote, index, overlap score, threshold, selected event ID and user-facing verdict.

## Critical statistical status

This is **not** an independently validated benchmark. We already read results from Pilot 04 and Pilot 05 on exactly this question set. Its labels were authored by this assistant using original narratives and never independently adjudicated. Descriptive three-seed averages should not be labeled fresh generalization, model comprehension or rigorous NLI accuracy.

A genuinely confirmatory Pilot 07 must use **new prompts independently reviewed by humans**. Before deployment, institute a blind procedure where at least two reviewers independently mark what a source passage entails/refutes/doesn't address, with adjudication of disagreement and inter-rater agreement; isolate related narrative clusters between model selection and final test. Do not claim those reviewers or labels are already available.

## Reproduce

Python 3.11+:

```sh
python -m pip install 'numpy>=1.26,<3' 'pyarrow>=17,<24' 'scikit-learn>=1.5,<2' 'rank-bm25>=0.2,<1'
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot06.py --seeds 31,37,43 --output results/imprinting/pilot06.json
```

Dedicated CI verifies the frozen source/checkpoint, runs all regressions, and uploads complete per-case JSON as an artifact. A successful GitHub Actions run means the assay executed, **not that reliable truth verification was achieved**.
