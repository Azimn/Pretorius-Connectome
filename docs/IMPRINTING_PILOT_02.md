# Pilot 02: unfamiliar lexical cues, absent-memory rejection, interference

**Scope:** Follow-on from the successful Pilot 01 synthetic associative overlay. Runs on the frozen 450-record v12 corpus, using the same read-only source pinning in `scripts/run_imprinting_pilot.py`. No new autobiographical records or source edits. This stage still does NOT use the biological FlyWire v783 graph.

## Why this pilot exists

Pilot 01 measured 85.2% mean top-1 oracle fingerprint identification at 450 imprinted memories, compared with zero for its shuffled-label control. But all cue phrases used during its probes were present during training; even the 100% fixed-50 identification concealed a decline in representation cosine. Pilot 02 isolates different weaknesses rather than expanding the autobiography.

## Two independent experiments

**A. Episode-disjoint absent-memory rejection with perturbed cues.** All 27 episodes are assigned to exactly one of train, calibration-negative, and test-negative partitions for each seed. The model sees events only from train episodes. Withheld calibration episodes are distinct from withheld test episodes. This is *episode-disjoint*, not narrative-cluster-disjoint: cross-episode references and thematic overlap can remain, and do not imply that the network can encode an unseen lifetime episode.

Original training memories supply both their candidate associative registry IDs and literal surface cue phrases. At query time, a deterministic probe uses the first eligible surface containing at least two words. For the primary probe it swaps the first two words, preserving the remaining words, and replaces its cue ID with a newly generated identifier based only on the transformed words. The secondary stress condition deletes its last word, again giving it a new probe-only identifier. This creates a genuinely unseen **surface string and ID**, but it is a small lexical perturbation, not an independent paraphrase or novel semantic cue.

Within trained episodes, the target event remains available in the *evaluator-only* narrative fingerprint codebook. Query positives are divided between a calibration-known group and a test-known group. Unknown calibration probes come from held-out calibration episodes, unknown test probes from different held-out test episodes. The threshold is selected exclusively by maximizing balanced accuracy for known versus unknown calibration probes, with ties selecting the stricter rejection threshold. It is then frozen for test-known and test-unknown probes. The reported top-1 identification uses the oracle codebook and does not imply that the network generates recollection. The decoder has no entry for unknown episodes: the negative task is **correct abstention**, not identification of an untrained memory.

Matched fixed-topology conditions are full cues, literal surface cues only, supplementary cue IDs only, shuffled narrative fingerprints, and no imprinting. A separate deterministic word-overlap retrieval baseline predicts the best trained item without abstention. Because the lexical baseline preserves exact token overlap on primary swapped probes, it may outperform synaptic imprinting; it is not resource-matched.

**B. Representation interference.** Independently imprint the entire 450-memory archive for each seed at matching 50, 100, 200 and 450 loads. Re-evaluate the earliest fixed 50 events against the same 50 fingerprints at every load. Report top-1 alongside mean target cosine and winner-versus-runner-up cosine margin. Stable top-1 alone is not proof of preserved representation.

## Input lock

Frozen base commit: `60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f`. Event Git blob: `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`. Annotation Git blob: `ad32025166c382caf13e07c7e3b0863eb89e1adb`. These exact bytes are tested at runtime, and the original candidate cue IDs are deliberately kept **separate** from literal surface strings. v12 candidates have not been independently perceptually verified.

## Reproduce

Python 3.11+, NumPy installed. Run from root:

```sh
python -m pip install "numpy>=1.26,<3"
python -m unittest discover -s tests -v
python scripts/validate_autobiographical_corpus_v12.py
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot02.py --seeds 0,1,2 --output results/imprinting/pilot02.json
```

GitHub Actions publishes the complete JSON result as the `pilot02-imprinting-results` workflow artifact, including per-seed episode partition IDs, counts, calibration thresholds, primary and secondary rejection rates, matched control results, and fixed-50 cosine/margin trajectories.

## Primary outcome and interpretation

For each seed, report the number of usable train and withheld probes, swapped-cue identification without abstention, calibrated true-positive rate on the known probe set, false-acceptance rate on test-negative held-out episodes, correct-and-accepted fraction, and balanced accuracy. Compare all against surface-only and ID-only ablations, shuffled association and unmodified controls. Report the lexical lookup benchmark separately, and avoid promoting a larger positive rate when false acceptance also increases.

For dropout probes, reuse the **swap-calibrated threshold** without changing it. Degraded performance measures sensitivity to lexical deletion under that threshold; it is not semantic generalization or a second independently calibrated test.

## Important limitations

The core imprint target is a SHA-256-derived pseudorandom content fingerprint, not a semantic embedding, autobiographical interpretation, relationship model, or decision representation. Neural weights cannot output text or choices in this pipeline. Decoder candidate texts reside outside the learned network. This is not a FlyWire connectome experiment, biological synaptic learning, whole-brain simulation, identity preservation, or artificial consciousness. Multiple seeds reduce stochastic sensitivity but do not establish statistical generality.

Splitting by episode is stricter than random record splitting but may not isolate narrative-duplicate clusters linking episodes. A later pilot must cluster semantically overlapping episodes/events *before* splitting and use independent, pre-authored paraphrases or environmental inputs for any true semantic transfer claim.

Behavioral decision tests are **deferred**, not implicitly passed. To make such a claim, collect independently authored decisions with context, alternatives, and prespecified labels, separate identities and situations across splits, and evaluate decisions from the learned substrate against prompt and retrieval baselines without oracle access to the target outcome.

## Decision gate

Proceed toward a biologically constrained synaptic overlay only if the new perturbation and rejection results justify more complex topology. If false acceptance or lexical fragility is high, investigate representation and encoding first. In either case, record the actual outcomes rather than declaring success from the fact that CI executes.
