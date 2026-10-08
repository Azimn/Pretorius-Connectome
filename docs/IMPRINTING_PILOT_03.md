# Pilot 03 protocol: cue representation under fixed synaptic capacity

**Status:** prespecified experiment, results unverified until workflow completes. Do not use this text as an outcome claim.

## Purpose

Pilot 02 exposed brittle cue transfer. The full synthetic overlay identified only around 28.6% of trained memories from word-swapped cues, while an ordinary word-overlap lookup identified around 89.4%. Pilot 03 asks whether *encoding* explains some of this difference without increasing the number of model synapses or the size of the first-person autobiography. It does **not** assert that a biological neural substrate has acquired a fictional life history.

## Immutable source and constraints

The same frozen 450-record, 27-episode `memories/current/Pretorius_v12_450_Events_Complete.jsonl` and `memories/annotations/v12_450_sidecars.jsonl` are mandatory and verified by Git blob SHA at execution. Baseline commit `60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f`. The memory content, candidate cue registry, and existing Pilot 01 and 02 code are not altered.

All 7 conditions share each seed's fixed random boolean bipartite mask, 512 cue units, 256 fingerprint units, 55% mask density, and masked additive Hebbian plasticity. The narrative target remains the Pilot 01 nonsemantic SHA-256-based fingerprint. The evaluation-only decoder is an oracle that retains the candidate story fingerprints. Nothing in the imprinted network is a text recollection.

## Comparison

The primary comparison includes `legacy_full` (Pilot 01 cue hashing, including all supplementary IDs), `legacy_surface` (Pilot 02-style surface-only exact cue ID plus phrase hashing), `token` (order-invariant signed feature hashing of lowercase word tokens only), `char3` (signed hashing of within-word character trigrams), and `idf_char3` (same character trigrams, with inverse-document-frequency scaling estimated from **training episodes only**). `token_shuffled` trains word features against a deranged narrative-target mapping, while `token_unmodified` never receives plasticity updates.

Fixed topology does not mean fixed whole-system memory expenditure: the IDF approach maintains a separate train-only feature-frequency dictionary. Report this overhead rather than silently claiming a completely resource-matched comparison. The character and token encoders ignore event IDs and unique literal-cue identifiers; the two legacy controls retain their prior ID-dependent behavior.

Three prespecified seeds, **31, 37 and 43**, create episode-disjoint training, validation-negative and test-negative sets. The previously reported Pilot 02 seed-level outcomes are not used for selecting thresholds or fitting the new encoders. As in Pilot 02, splitting by episode does not guarantee narrative-cluster independence across different episodes.

## Probes and calibration

The primary query reverses the first two words of an eligible surface cue. The first secondary query removes its final word. A second secondary query changes the middle character of the longest eligible word, preserving every other character. All queries have **new identifiers** and are not appended to the imprinted memory records.

These deterministic edits are morphological/lexical perturbations, **not independently authored paraphrases or semantic generalization**. Their production rules are authored before the evaluation run. They may preserve most original words. The token-bag condition is intentionally invariant to the primary swapped-token challenge. It should not be celebrated as evidence of concept learning.

Within the trained episodes, available known-event swapped-cue probes are randomly divided into calibration positives and untouched test positives. Validation-negative episodes contribute only to the calibration of an abstention threshold; disjoint test-negative episodes are unseen until scoring. A threshold is chosen per condition to maximize calibration balanced accuracy, with ties favoring stricter rejection. It is applied *without retuning* to swapped, dropped and typo probes. For secondary positive challenges, use the same target event IDs as the primary test-positive set.

Per seed and condition, record top-1 event identification without abstention, known acceptance (TPR), unknown false acceptance (FPR), correct-and-accepted recall fraction, threshold-calibrated balanced accuracy, and test-set score discrimination AUROC. Threshold-independent AUROC is still an oracle fingerprint-decoder score, not a confidence that a fact was remembered. A separate text word-overlap retriever is measured for all known challenges without abstention; it is not resource-matched.

## Reproduction

Python 3.11+ and NumPy, from repository root:

```sh
python -m pip install "numpy>=1.26,<3"
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot03.py --seeds 31,37,43 --output results/imprinting/pilot03.json
```

The Pilot 03 GitHub Actions workflow runs this command after source checks and unit tests, and uploads a JSON artifact to preserve each seed's partitions, trained counts, thresholds, results and controls. A passing CI build alone is never a claim of successful generalization.

## Decision gate

Inspect the three-seed mean **and range**, compare new encoders against both legacy conditions and the strong lexical retrieval reference, examine false acceptance at calibrated thresholds, and explicitly record costs of the train-only IDF dictionary. No improvement on synthetic word swaps or typos can establish human-quality autobiographical recall, persistent persona identity, biological FlyWire imprinting, or improved behavior.

Independent human-authored paraphrases, semantic near-duplicate cluster isolation, and genuine choice-conditioned behavioral probes remain *future* experiments. Do not use Pilot 03 test outcomes to optimize the encoder and still call this run an untouched holdout. If Pilot 03 fails, preserve it as a negative result and investigate learning-rule and decoder limitations before scaling up biological wiring.
