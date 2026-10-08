# Pilot 04 executed results: semantic transfer failure

**Execution:** 2026-10-08. **Status:** challenge executed, test suite green, results NEGATIVE for semantic autobiographical recall. Pilot 04 is a meaningful failure mode finding, not a successful semantic-memory demonstration.

**Provenance:** 450 immutable v12 memories in 27 episodes; 34 sampled original event IDs in 23 episodes; 68 assistant-authored, non-blind and **human-unreviewed** prompts. Challenge source Git blob `1e1c6f0273519bf3cd5868404ddfa5f783df6abd`. Input archive and sidecars use Pilot 01's existing pinned SHA values. No source memories, annotations, or earlier model definitions were changed.

**Source evidence:** [Pilot 04 successful CI run 37828101750](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37828101750); [foundation suite success 37828101654](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37828101654). Full exact case-level and seed-level values reside in the `pilot04-semantic-results` artifact of the Pilot 04 workflow. Experiment version is immutable via source and challenge Git blob checks. All numbers below reflect actual output, not predictions.

## What was actually tested

The seven original Pilot 03 models were **retrained by identical frozen Pilot 03 procedures** on training episodes only, with identical synthetic 512-by-256 fixed masked topology, SHA-256-derived opaque fingerprint targets, additive Hebbian updates and seeds `31,37,43`. Calibration thresholds were selected from original word-swap probes of learned events and separate validation-episode negatives. Challenge text never entered training or threshold calibration.

For each seed, the authored challenge cases were separated by whether the original target was in imprinted training episodes or unseen test episodes. Learned-event true paraphrases tested oracle-codebook event identification and correct-plus-accepted decisions. Learned-event counterfactuals tested whether the system would falsely endorse a contradicted memory, **not** ordinary unknown-ID rejection. Real events from entirely withheld test episodes tested open-set rejection.

| Seed | Trained events | Eligible trained paraphrase cases | Paired trained contradiction cases | Unlearned-event paraphrases |
| ---: | ---: | ---: | ---: | ---: |
| 31 | 317 | 23 | 23 | 6 |
| 37 | 316 | 20 | 20 | 9 |
| 43 | 310 | 24 | 24 | 5 |

One paired challenge per 34 source events gives a small trial and a limited per-seed negative sample. Means are descriptive averages of seeds with unequal denominators, not precision-weighted population estimates.

## Main test results

Values are fractions. `True paraphrase top-1` measures choosing the true event from the evaluator's codebook without abstention. `Correct and accepted` is a stricter useful-recall fraction. `Counterfactual acceptance` is **a failure**: the network responds to a prompt known to contradict its paired memory rather than abstaining.

| Method | True paraphrase top-1 | Correct and accepted | Legitimate paraphrase acceptance | Counterfactual false acceptance | Direct false confirmation of contradicted event | Unlearned episode false acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy full | 0.013889 | 0.000000 | 0.347343 | 0.440821 | 0.045048 | 0.462963 |
| Legacy surface | 0.000000 | 0.000000 | 0.290580 | 0.276691 | 0.028382 | 0.148148 |
| Token | 0.000000 | 0.000000 | 0.013889 | 0.028382 | 0.000000 | 0.000000 |
| Character trigrams | 0.013889 | 0.000000 | 0.016667 | 0.043478 | 0.000000 | 0.000000 |
| IDF character trigrams | 0.000000 | 0.000000 | 0.016667 | 0.016667 | 0.016667 | 0.000000 |
| Token-shuffled narrative labels | 0.000000 | 0.000000 | 0.066667 | 0.016667 | 0.000000 | 0.074074 |
| Never imprinted | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Lexical retrieval reference using original recall-cue words (no abstention):** paraphrase top-1 mean **0.090097** (seed 31: 0.086957, seed 37: 0.100000, seed 43: 0.083333). It picked the specifically contradicted event in **0.104589** of negative probes; because this is forced-choice, it inherently cannot abstain.

**Cue lexical-distance diagnostic:** author-authored positive paraphrases shared on average only about **1.87% of query tokens** with the original cue-word sets for the corresponding trained memory (per-seed mean overlap fractions 0.013181, 0.020647, 0.022368). This is a deliberately difficult near-zero-overlap challenge, but the statistic is not independent proof that the queries accurately preserve the source event's meaning.

## Interpretation and caution

**Pilot 03's roughly 77.9% identification from swapped original cues did not transfer to the freshly authored incident descriptions.** The token and IDF-trigram models each identified **zero** trained events from authored paraphrases on all three seeds. Even the rare correct top-1 legacy/character cases were rejected by their frozen threshold. Across all seven methods and all three seeds, **no system produced any correct AND accepted autobiographical identification**.

Low contradiction acceptance by the IDF, token and character models must NOT be credited as contradiction reasoning. They also accepted only about 1.4% to 1.7% of legitimate paraphrases. Their behavior is close to a reject-everything default under this out-of-distribution wording. The unmodified overlay, as expected, rejects absolutely everything and achieves zero recall. The legacy-full model accepts substantially more legitimate prompts, but also incorrectly accepts 44.1% of contradictory versions and 46.3% of events it never imprinted.

This result is consistent with a fundamental **representation/content bottleneck**. The train-side model encodes a handful of literal recall cue strings, then associates those vectors with *opaque pseudorandom fingerprints* of the narrative. It does not train an embedding that preserves the semantic relations between questions and story facts, nor does it contain a textual memory readout or entailment checker. Moving from word-order perturbation to new descriptions of what happened removes nearly all cue-feature overlap. Merely increasing the number of synapses cannot be assumed to fix that.

This is not a demonstration that connectomes cannot hold memories, that 450 autobiographical records are insufficient, or that all synaptic imprinting approaches fail. It demonstrates that **this particular cue-to-opaque-hash synthetic implementation does not transfer to these authored semantic challenges**.

## Limitations affecting inference

The challenge author inspected the original fictional memory texts and knew their identities; labels are assistant-authored and **not independently human adjudicated**. The test set includes 34 selected events, not 450 random events, with 20 to 24 positive cases per seed after episode withholding. Multiple episodes may share related motifs and narratives. The underlying training material contains only cue phrases as learnable inputs and unrelated hash fingerprints as output targets; the assay is not an equivalently semantic-input model benchmark. For semantic text matching, a strong retrieval comparator should be allowed access to the **actual narrative text**, not only to the original cue words. The cue-words-only lookup included here is deliberately limited and is not that comparator.

No free-response recollection, decision production, autobiographical reinterpretation, environmental embodiment, real biological wiring, or identity stability is tested. The held-out prompts are fictional recollections, not independent historical records.

## Next research gate: narrative-grounded representation, not more records

1. **Freeze Pilot 04 as an untouched result**. Do not tune encoder or rejection threshold against these exact challenge cases and then report the same scores as unbiased held-out performance.
2. Develop a clearly marked **post-hoc diagnostic** comparing cue-only lexical retrieval, a narrative-text BM25/TF-IDF retrieval comparator, and the existing synthetic overlay. The narrative retriever will have additional information (full text) and must be reported as a distinct architectural class, not as matched weights.
3. For a genuine Pilot 05, build a content-aware semantic memory architecture that encodes both event content and questions into a related representation, with explicit provenance, entailment/contradiction validation, and an abstention decision requiring evidence. Preserve immutable original source and learning overlay, and keep event text out of the neural weights *unless that change is the registered intervention*.
4. Have newly authored challenge labels reviewed independently and freeze a **new** episode- or narrative-cluster-disjoint evaluation set before fitting or choosing the Pilot 05 architecture. Include a strong full-text retrieval baseline and confidence/false-support tradeoffs. Only then ask whether synaptic weights add value beyond retrieval.

This outcome corrects the overly broad interpretation one might otherwise make from Pilot 03's token-order and typo robustness. The scientifically relevant question now is what *semantic information* the substrate can actually store, use and verify.
