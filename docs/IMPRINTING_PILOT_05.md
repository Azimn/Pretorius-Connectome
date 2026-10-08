# Pilot 05A: Narrative-grounded memory representation, post-hoc diagnostic

**Status:** experimental code and protocol. Measured results will be recorded separately after GitHub Actions executes the registered command. No semantic or behavioral success is presupposed.

## Motivation and scope

Pilots 01 through 03 established learned associations for *previously familiar lexical recall cues*, but Pilot 04 found essentially zero useful recall when asked to identify events from entirely rewritten incident descriptions. The reason may be straightforward: the earlier overlays encoded a small cue registry and linked it to nonsemantic SHA-256 fingerprints; they did **not** encode autobiographical event content as input. Pilot 05A isolates this architectural omission.

The original **450-event, 27-episode v12 autobiography**, its sidecars, and the **68 previously examined Pilot 04 cases** all stay immutable, pinned by Git blob SHA. The former is real training-source content for this experiment; the latter is explicitly **a post-hoc diagnostic, not a fresh independent holdout**. No new, human-reviewed cases are asserted to exist.

## Changed intervention

The original story narrative (`memory_text`) is now the *source input representation* of the imprinted event. The decoder's 256-dimensional narrative fingerprint remains opaque and externally indexed. This is not semantic neural output: it tests whether richer input features improve learned identification under a fixed topology. No original memory, cue or narrative is added, edited, or removed.

The experiment contrasts seven methods:

| Method | Training input and retained knowledge | Retrieval mechanism |
| --- | --- | --- |
| `bm25_narrative` | Full first-person narratives | Standard `rank_bm25` BM25 inverted-document statistical scoring |
| `tfidf_word_narrative` | Full narratives; TF-IDF unigram/bigram word features | scikit-learn cosine similarity |
| `tfidf_char_narrative` | Full narratives; TF-IDF character 3-5 grams | scikit-learn cosine similarity |
| `hebb_word_narrative` | Full narrative word TF-IDF vectors at imprint time | Masked additive Hebbian weights; oracle hash-code decoder |
| `hebb_char_narrative` | Full narrative character TF-IDF vectors at imprint time | Matched masked weights; oracle decoder |
| `hebb_word_shuffled` | Same narrative word features but mismatched fingerprints | Wrong-association negative control |
| `hebb_word_unmodified` | Same fixed mask but no updates | Unmodified neural control |

These conditions are **not resource-matched**: BM25 and TF-IDF explicitly store searchable full narratives and a vocabulary, whereas the neural weights do not retain those texts. The neural evaluator separately holds a fingerprint codebook containing the narratives. The neural overlays share the original synthetic 512-input by 256-output mask, density 0.55, and seed. BM25/TF-IDF are strong evidence-retrieval baselines, not implementations of FlyWire or biological neuromorphic memory. No external language model is required and no paid API is used.

## No challenge leakage

For each seed `31,37,43`, group original source memories by episode into distinct training, validation and test partitions with the unchanged Pilot 02 splitter. Fit lexical vocabularies, document frequency weights and neural updates exclusively using **training narratives**. The independent original structured `decisions` field supplies calibration-positive queries for training events and calibration-negative queries for validation events. That field comes from the unchanged source JSONL, **not the challenge metadata** and not the Pilot 04 question labels.

Calibrate the top-match acceptance threshold per method by maximizing balanced accuracy on the training/validation **decision statements**, with stricter rejection breaking ties. Fix it before scoring Pilot 04 cases. The case IDs, positive/negative labels, source anchors, and expected event IDs remain evaluation-only metadata; query strings are the sole model input. Do not fit the TF-IDF or BM25 vocabularies using challenge questions.

The original Pilot 04 paired prompts are scored in three distinct categories: true descriptions of events actually imprinted, false descriptions that contradict those imprinted events, and true descriptions of events belonging to wholly untrained test episodes. The latter must be rejected. Contradictions must be rejected **because of their truth status**, but this implementation has **no reliable logical entailment or contradiction classifier**: therefore its acceptance on these cases is explicitly a *false support* diagnostic. A low false-acceptance rate may also mean rejecting every valid paraphrase.

## Metrics and evidence

For each method and seed, record top-1 identification, acceptance and **correct-and-accepted** identification for genuine trained-event paraphrases. Separately report false acceptance and direct false confirmation of contradictory descriptions; and false acceptance for untrained episodes. Record both sensitivity and rejection on non-challenge calibration queries. Retain all raw per-case predicted IDs, scores, margins, and accepted/rejected decisions in the workflow artifact.

For conventional BM25/TF-IDF methods only, expose a short *inspectable narrative sentence* for the top-ranked document. That snippet is **retrieved provenance**, not a claim that the sentence logically entails the query. The neural model has no independent textual evidence generator and must not be credited with textual recollection because its oracle decoder prints an event ID.

## Reproducibility

Run from the repository root with Python 3.11+:

```sh
python -m pip install 'numpy>=1.26,<3' 'pyarrow>=17,<24' 'scikit-learn>=1.5,<2' 'rank-bm25>=0.2,<1'
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot05.py --seeds 31,37,43 --output results/imprinting/pilot05.json
```

The dedicated GitHub Actions workflow checks both pinned sources, runs the full regression suite, executes the three-seed comparison and preserves `pilot05.json` as an artifact. No hidden GPU, LLM inference, network download of trained embeddings or API key is needed.

## Interpretation

A gain over Pilot 04's cue-only model is exploratory **post-hoc** evidence that narrative-grounded input and full-text access matter. It is not a fresh unbiased estimate of generalization, semantic comprehension, identity, future action selection, FlyWire direct synaptic imprinting, or consciousness. Failure is also informative and should be preserved.

A true next confirmatory phase requires a **new independently reviewed, episode/cluster-disjoint challenge** that is authored and frozen before architecture fitting. It must test question entailment, negated or confusable episodes, evidence-grounded abstention, and genuine decisions separately. If BM25/TF-IDF outperform the neural associative overlay, report that clearly rather than retrofitting an interpretation in favor of the connectome hypothesis.
