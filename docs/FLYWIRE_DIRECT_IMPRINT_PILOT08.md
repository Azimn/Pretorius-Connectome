# Pilot 08: direct content-conditioned synaptic imprinting on real FlyWire v783

**2026-10-08. Status:** implemented exploratory experiment; **real-data outcome pending CI evidence**. Primary project [Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17). Distinct from external-document retrieval and [Pilot 01 graph coactivity plasticity](../results/associative/FLYWIRE_MB_PLASTICITY_PILOT01_RESULTS.md). Start with the canonical [shared memory archive](../artifacts/shared_memory/v1/manifest.json) rather than reproducing biographical preparation.

## What is and is not imprinted

This pilot actually **writes content-dependent numerical updates onto a subset of existing directed edges** from the exact original whole-brain FlyWire v783 graph. A finite learned-weight vector is indexed one-for-one by original CSR edges. The biological graph's exact root IDs, indptr, post neuron indices, integer synaptic contacts and source files are never modified. The model has **no source-text database, event-ID table, document vectors, target fingerprints, retrieval index or candidate search**.

Its input and output representations are explicitly **lexical, not semantic**. Source-linked BC01's existing stateless signed hashed 256-sensory vector serves as BOTH the cue encoder and the full-narrative target encoder. This is intentionally a modest first direct biological-topology test, not a biological brain simulation, semantic neural memory, LoRA adapter, language generation, or a conscious Pretorius.

For each feature coordinate, a publicly reproducible random seed allocates 32 presynaptic and 32 *different* postsynaptic neurons (on the original full biological graph). The two sets are disjoint. Cue encoding selects the eight largest signed feature magnitudes; content selects 32. Training coactivates these populations and modifies ONLY the eligible original pre-to-post directed edge weight indices, scaled by log1p(original synaptic contact count), clipped to [-4,+4]. This signed efficacy overlay is a computational assumption; transmitter identity and biological learning kinetics are not established.

At inference, a never-before-presented one-cue surface is encoded by the frozen BC01 rule. Only the **trained sparse synaptic deltas on existing FlyWire edges** and the fixed neuron-coordinate maps produce a 256D distributed output. There is no access to source narratives in that forward pass. The model returns a vector, not the name, first-person story, or event ID.

**Evaluation-only oracle:** separate from model inference, an external evaluator holds the canonically sourced target-content vectors and compares the model's output to training-event targets. Event top-1 and readout cosine therefore assess narrow learned content association, **NOT autonomous autobiographical recollection**. This assessor cannot be used by or passed to the model's infer() method. Checkpoints save ONLY sparse learned deltas, graph array hashes, map seeds and training counters, with exact source-bound load/replay checks. Any statement that the fly remembered a human autobiography would be false.

## Experiment design

Use all 450 preserved event records and 27 episodes as the source universe. A fixed seeded episode-disjoint split defines train, calibration-unknown and test-unknown sets (seed 31 normally gives 317 imprinted train events). Each train event reserves its last literal recall cue for inference, while earlier literal cues condition its synaptic training. This is **leave-one-cue-out surface retrieval**; source candidate cue annotations are unreviewed and may share words across scenes. The experiment does not claim evaluation on independent human-written paraphrases.

Evaluate three independently instantiated models on the same exact graph and fixed feature/neuron mapping: (1) source-matched content-conditioned synaptic updates; (2) identical cue/update procedure with a deranged event-content assignment; (3) original untouched zero-delta synapses. Record every probe output's externally scored prediction, cosine to the intended content vector, whether the event is in the imprint set, and rejection of withheld test-episode cue prompts. The acceptance threshold is calibrated ONLY using separate train-event cues and withheld validation episodes, and then applied unchanged to methods and withheld test episodes. A deterministic temporary checkpoint is reloaded and compared bit-for-bit with the trained model on probes.

For scaling, the optional --full-corpus flag imprints all **450** events, but cannot simultaneously claim withheld-episode negative performance. It is labeled development-only; use the episode-separated comparison for rejection/generalization claims. Source topological capacity and healthy graph routing are still possible bottlenecks. There is currently **no support/degree-matched rewired whole-brain comparator** for this direct-imprint pilot. That must be added before making claims of a topology-specific benefit rather than merely using a graph-shaped parameter mask.

The principal test is whether a nonzero, cue-dependent numeric learned readout with externally scored content association exists, and whether that signal is better than shuffled and frozen controls. Synaptic change counts alone are not a successful memory result. Case labels from the existing source sidecars are not human-validated. Do not rebrand earlier synthetic Pilot01-07 evaluation as real FlyWire evidence.

## Execute

Python 3.11 with NumPy, SciPy, scikit-learn and rank-bm25. Real biological import needs PyArrow. Start with:

~~~sh
PYTHONPATH=src:. python -m unittest discover -s tests -p test_direct_flywire_imprint08.py -v
~~~

The [Pilot 08 workflow](../.github/workflows/direct-flywire-imprint08.yml) independently validates the source and BC01 cache, passes synthetic leakage/structural tests, then checksum-verifies publisher FlyWire v783 inputs and converts the original 139,255-neuron / 15,091,983-aggregate-edge biological graph. The real experiment fails closed unless hashes of **all four** arrays equal the full-brain independently measured source result from [real CSR integration run 37847525950](../results/shared_memory/flywire-v783-real-csr-invariance-run37847525950.json).

The real workflow command:

~~~sh
python scripts/run_direct_flywire_imprint08.py \
  --topology data/derived/flywire_v783_csr.npz \
  --bc01-dir data/derived/direct08-bc01 \
  --seed 31 --cells-per-feature 32 \
  --checkpoint data/derived/direct08-real-checkpoint.npz \
  --output results/imprinting/direct08-real-whole-brain.json
~~~

Complete per-case JSON, topology/source/representation versions, number of updated connections, baseline vs shuffled numerical comparisons, false acceptance and recovered numerical checkpoint are in the original Actions artifact. **A passing synthetic smoke test cannot be reported as a biological result.** Once the full biological run succeeds, the measured JSON must also be committed permanently and the handoff updated. A numerical null result is a valid result and must not be hidden.

## Next research steps after results

Assess whether the real biological original has enough structural overlap between selected populations. Compare full 450 vs 317-event load, sequence retention, density sensitivity and fixed resource budgets. Then add capacity/degree matched rewiring on the relevant subgraph, an independent frozen cue set and a more informative validated content encoder. Do not silently replace lexical coordinates with purported semantics, and do not transfer trained weights into BioCircuit or The Doctor Lives without an explicit interoperable model contract.
