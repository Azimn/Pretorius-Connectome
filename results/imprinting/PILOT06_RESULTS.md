# Pilot 06A results: sentence evidence gates and explicit abstention

**Date:** 2026-10-08. **Status:** executed, tests green, **negative evidence for useful truth verification**. The measured experiment was intentionally post-hoc. Its frozen sources remain unchanged.

## Evidence and exact reproduction

Dedicated experiment [GitHub Actions run #37832193714](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37832193714): success. Dedicated `pilot06-evidence-results` artifact (GitHub artifact ID `11573907929`) contains the complete per-case records as `pilot06.json`. Uncompressed JSON: 592,670 bytes; SHA-256 `c6bb4034d6092217401286f0dc9a659a93247b7a8d34e3446c97147b83f93b5d`. Foundation regression run [#37832193635](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37832193635): success.

The source and annotations are the **same immutable 450 memories in 27 episodes**. The original Pilot 04 challenge is pinned to Git blob `1e1c6f0273519bf3cd5868404ddfa5f783df6abd`, containing 68 assistant-authored, **human-unreviewed**, previously examined prompts across 34 event IDs. No question text or label enters retrieval-index fitting, synaptic updates or threshold calibration.

All models used the same seeds as Pilot 05A (31, 37, 43), same episode splits, identical trained full-text source documents and original decisions-field calibration queries. Each evaluation trial had 23/20/24 trained true paraphrases, 23/20/24 trained contradictions, and 6/9/5 never-imprinted episode prompts. Train events numbered 317, 316 and 310; training evidence contained 3,464 / 3,417 / 3,315 sentences. The sentence evidence TF-IDF index was built **only** from trained narrative sentences, not from the benchmark.

The fixed masked Hebbian model remains a 512-input / 256-fingerprint-output synthetic overlay with oracle external codebook. The evidence-hybrid condition additionally accesses original source text: it is *not neural-weights-only memory*. All conditions are resource-unmatched against full-text BM25 and each other, and this is not a FlyWire brain graph.

## Main outcomes: mean across three paired seeds

All numbers below are *proportions* and descriptive unweighted means of three trials. A true prompt is useful only if its original event is **both correctly identified and accepted**. False confirmation of a contradictory statement is never desirable. A low contradiction-acceptance rate is useless if the model rejects every true claim.

| Condition | True top-1 ID without gating | **Correct AND accepted true ID** | Accepted true prompts | Accepted contradictions (false) | False confirmation of targeted contradiction | Accepted unlearned episode prompts (false) | Mean possible-contradiction flags / seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 retrieval with Pilot05A score calibration | 0.298068 | **0.163527** | 0.608696 | **0.626932** | 0.329831 | 0.529630 | 0.00 |
| BM25 plus source sentence gate | 0.298068 | **0.000000** | 0.078382 | **0.016667** | 0.016667 | 0.159259 | 0.00 |
| BM25 sentence gate plus explicit negation mismatch | 0.298068 | **0.000000** | 0.045048 | **0.016667** | 0.016667 | 0.159259 | 0.00 |
| Full-text word TF-IDF plus evidence/polarity gate | 0.134541 | **0.000000** | 0.078382 | **0.042270** | 0.028382 | 0.000000 | 0.00 |
| Narrative Hebbian word imprint plus **external** evidence/polarity gate | 0.089493 | **0.042270** | 0.119082 | **0.070652** | 0.028382 | 0.103704 | 0.33 |
| Always reject (reuses BM25 raw rankings only for measurement) | 0.298068 | **0.000000** | 0.000000 | **0.000000** | 0.000000 | 0.000000 | 0.00 |

The unchanged BM25 top-1 number across its three gate conditions confirms that those conditions **only change acceptance**, not which event was retrieved. Their lower false acceptance does not mean improved event identification. The always-reject control has perfect rejection but completely fails recall. All models remain dependent on query distributions and the source-vocabulary fit.

## Outcome and failure analysis

**The simple evidence gate is too conservative to support useful recall from independently reworded descriptions.** BM25's correct-and-accepted recall dropped from 16.4% without evidence filtering to **zero** with either sentence gate, despite its contradiction acceptance dropping from 62.7% to 1.7%. This is principally an abstention trade-off, not a correctness breakthrough.

An earlier ambiguity reappears: the sentence index can retrieve a real sentence relevant to a query, but **word overlap plus TF-IDF cosine is not entailment**. The additional rule requiring two shared content tokens and calibration on original decision statements rejects substantially different paraphrases even when they are true. Conversely, some false statements share enough words with a legitimate source sentence to pass.

**Negation matching was functionally unhelpful on this challenge:** the BM25 variants had identical contradiction-acceptance rates of 1.67% with and without polarity checks, and neither flagged any benchmark contradiction as an explicit polarity mismatch passing the evidence gate. The neural-plus-evidence hybrid flagged only about 0.33 negative cases per seed. Most false statements are **semantic contradictions without overt negation**, outside the heuristic's capabilities.

The neural + external sentence gate retains 4.23% correct-and-accepted positive recall, down from 5.62% in Pilot 05A's word-Hebbian model without the gate; it still falsely accepts 7.07% of contradictory queries. This hybrid benefits from **external narrative retrieval/storage**, so it is not evidence that its synaptic weights themselves verify truth. It also returns oracle event IDs, not generated recollections.

**The result is negative for semantic evidence verification.** More layers of fixed word matching do not yet solve autobiographical recall, truth-sensitive responses or user-context decisions. It is scientifically useful to preserve the negative result and the exact failure traces rather than optimize against known benchmark answers.

## Reproducibility and publication boundaries

Run `python scripts/run_imprinting_pilot06.py --seeds 31,37,43 --output results/imprinting/pilot06.json` after installing NumPy, PyArrow, scikit-learn and rank-bm25 as listed in [the protocol](../../docs/IMPRINTING_PILOT_06.md). CI also runs all existing regressions. Per-case JSON contains the selected event ID, target ID **for the evaluator only**, source sentence index and exact quote, lexical similarity, anchor count, classifier verdict and accepted/rejected state. Those fields allow inspecting where genuine true prompts were rejected and where false claims appeared related to authentic passages.

The Pilot 04 benchmark was created by this assistant with access to original narratives and later repeatedly examined. **It is not an independent human-reviewed evaluation set.** Pilot 06A is intentionally a post-hoc diagnostic, and even a strong number here could not establish confirmatory performance. No pretrained entailment classifier or generated answer was evaluated; no actual FlyWire biological connectivity, persona continuity or independent cognition was tested.

## Decision for the next phase

1. **Do not accept lexical citation matching as a semantic truth validator.** The ablation demonstrated substantial false-acceptance reduction only by discarding every correct BM25 retrieval.
2. Preserve a **three-stage separation:** broad narrative retrieval, claim-to-passage entailment/contradiction/unknown inference, and evidence-bound answer/abstain. Claims must remain distinguishable from questions about nearby but different incidents.
3. For Pilot 07, require **a new independently human-reviewed set of meaning-preserving, negated, contradictory and ambiguous prompts** that is frozen before model selection. Reuse these exact Pilot 04 prompts only for development diagnostics, never for a second claimed independent test.
4. Evaluate a small, version-pinned **local pretrained NLI cross-encoder** or comparable trained entailment model as a separate mechanism from both BM25 and the neural overlay, under the no-paid-API and reproducibility constraints. If unavailable, record a clear blocked capability rather than substituting a word overlap proxy and calling it entailment.
5. Report evidence precision and support/rejection trade-offs **together** with correct-and-accepted event identification. Do not assume expanding the connectome to more synapses can repair missing semantics.

No further source autobiographical records are required for this gate.
