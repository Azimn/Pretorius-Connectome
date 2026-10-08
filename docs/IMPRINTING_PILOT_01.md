# Pilot 01: synaptic imprinting of v12 autobiographical cues

**Status:** reproducible synthetic associative-memory engineering assay. No biological imprint, natural-language recollection, identity, subjectivity, or behavioral agency is claimed.

## Frozen v12 input

The experiment reads existing, unchanged `memories/current/Pretorius_v12_450_Events_Complete.jsonl` and `memories/annotations/v12_450_sidecars.jsonl`. These are 450 reconstructed autobiographical records from 27 episodes. Candidate cue annotations are **not** manually verified perceptual observations. The corpus itself must not be modified for Pilot 01.

The baseline is repository commit `60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f`.

| Source | Git blob SHA-1 |
| --- | --- |
| Complete v12 events | `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5` |
| v12 cue sidecars | `ad32025166c382caf13e07c7e3b0863eb89e1adb` |
| Candidate cue registry, retained but not queried by this pilot | `7bb59e35674baf09f435e5d6551eae0fbb17a24f` |

The runner fails closed when the first two content blobs differ from these checksums. It reads all 450 events before making any load subsets. The 500- and 1,000-record goals remain future scaling conditions, **not launch gates**.

## Assay mechanism and safeguards

The current `src/pretorius_connectome/substrate.py` is a separate small recurrent smoke-test. Pilot 01 adds an **independent, fixed synthetic bipartite topology**, not a recurrent whole-brain simulation. The fixed boolean mask defines permitted contacts between hashed cue units and imprint units. A distinct numeric overlay accumulates signed, masked Hebbian updates. The wiring mask remains read-only throughout. Neither text, cue lookup tables, event IDs, nor oracle answer keys enter this overlay.

The cue encoder hashes each candidate cue ID, phrase, word, and neighboring word pair into normalized signed features. The imprint target is an opaque pseudorandom bipolar fingerprint derived from the **memory narrative only**. It is not a semantic embedding. After training, the readout is a numeric fingerprint approximation, **not generated autobiographical text**. For evaluation only, an oracle decoder compares the readout against candidate narrative fingerprints held outside the synaptic model. Identification therefore measures a narrow learned association; it does not demonstrate actual remembering or independent text recall.

A fixed episode-balanced curriculum reduces early-load chronological bias. Loads 50, 100, 200, and 450 share the same order and topology within each seed. The runner uses multiple seeds and measures one original cue per previously imprinted event. That cue was included during training as part of the original memory's full cue set. This is **incomplete-cue reconstruction of trained memories**, not unseen-memory generalization or a true held-out episode benchmark.

Each load measures four conditions: learned overlay, overlay trained on mismatched narrative targets, entirely unmodified overlay, and conventional exact cue-ID lookup. The text lookup is a distinct oracle-style engineering reference, not a resource-matched neural comparator. Learned and shuffled overlays share the identical frozen connectivity per seed.

Reported metrics are top-1 event identification via the external codebook, mean cosine with the target fingerprint, and stability of the earliest 50 events under a **fixed 50-candidate decoder** while the overlay accumulates new memories. The fixed decoder controls one distractor-count confound, but does not by itself establish causal catastrophic forgetting.

## Reproduce

Requires Python 3.11+ and NumPy. From the repository root:

```sh
python -m pip install "numpy>=1.26,<3"
python -m unittest discover -s tests -v
python scripts/validate_autobiographical_corpus_v12.py
python scripts/run_imprinting_pilot.py --seeds 0,1,2 --loads 50,100,200,450 --output results/imprinting/pilot01.json
```

The result JSON contains all settings, pinned source blob IDs, episode coverage, seed-level measures, and explicit interpretation warnings. The GitHub Actions pilot workflow stores results as a run artifact, not as a manually invented permanent metric.

## Scientific limits

This phase does not use the FlyWire v783 biological connectivity matrix, sign inference, biological neural dynamics, or neural tissue. It is **not** an in vivo connectome imprint, and learned overlay entries are not measured biological synaptic strengths. The fly connectome's original unsigned synapse counts, when eventually imported, must remain immutable; learned states must remain a separate layer.

Cryptographic target fingerprints are deliberately nonsemantic. The protocol tests whether a fixed topology and learnable weight layer can preserve many mappings and reconstruct latent codes from incomplete input. It cannot yet evaluate Pretorius's decisions, relationships, autobiographical reinterpretations, speech style, or cross-model identity continuity. All identities and memories here are fictional reconstructions.

A top-1 score after training on the same event should never be described as generalization. External holdout must group near-duplicate narratives and related event clusters *before* any future encoder selection, threshold tuning, or evaluation. The fixed 450-item source must not be silently rewritten.

## Gate to Pilot 02

After CI has passed and actual Pilot 01 measurements are archived, compare scaling curves, control separation, per-seed variance, and earliest-50 retention. Pilot 02 should add genuinely unseen cue perturbations, cluster-disjoint evaluation and an explicit out-of-set abstention criterion. It should add a learned and independently tested action/choice readout if behavioral influence is the claim. Only after these gates is there a basis for testing the same overlay semantics against a verified sampled FlyWire graph and a separate 4,096-unit recurrent baseline.

**Stop condition:** if the learned condition does not reliably outperform mismatched and unmodified controls, debug collision rate, topology density, update stability, and decoding artifacts before creating more autobiography.
