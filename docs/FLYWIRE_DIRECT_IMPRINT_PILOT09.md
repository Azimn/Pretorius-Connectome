# FlyWire direct imprint Pilot 09: frozen synaptic-state cue-transfer diagnostic

**Date:** 2026-10-08. **Status:** exploratory diagnostic implemented. This DOES NOT introduce a new learning rule or modify frozen Pilot 08 scores.

## Scientific question

[Direct real FlyWire Pilot 08](../results/imprinting/PILOT08_REAL_V783_PROVISIONAL.md) modified 25,780 existing directed synaptic-weight entries on the original 139,255-neuron/15,091,983-edge connectome, but correct event identification was **0/159** from last literal source cues, equally unsuccessful as shuffled content. There are at least two failure modes: inability to write even a known cue-content association, or failure to generalize from a trained set of literal cue surfaces to a separate literal cue for the same record. Pilot09 separates them, without performing any tuning or retraining of the real learned synaptic state.

## Frozen inputs and safeguards

Use exactly [Pilot08 run 37863818653](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37863818653), its original real 139k-neuron checkpoint (SHA-256 e6edf8dd68e84540140612dbcbc42f827e00ce707d9f2947adf3342e215fcb6d) and source-linked 450-event case JSON. Reverify the pinned publisher v783 release, regenerate original whole-brain CSR, then fail closed unless the root_ids, indptr, indices and integer synapse_counts match all four pinned source-array SHA-256 values in the preceding actual v783 audit. Load the exact original saved overlay and confirm 317 previous imprints at seed 31, with the same coordinate/neuron population mapping.

Rebuild existing BC01 lexical 256 vectors from the immutable L1 archive. Reuse exact episode-disjoint seed-31 train and withheld test split, the fixed 159-event evaluation sample (selected by the ORIGINAL Pilot08 seed and permutation), the original 71 withheld-episode absent events, original train event candidate pool, and **the original acceptance threshold 0.0082783**, rather than deriving a favorable new calibration.

Keep the 3 originally compared conditions: original content-linked checkpoint, re-created identical-procedure deranged cue-content control on the same graph and mapping, and the frozen zero-delta control. Any source content, event IDs, target codebooks and score calibration live strictly outside the unchanged model.infer(cue) function. That forward pass sees only the cue string and learned synaptic state.

## Paired primary comparisons

**Trained cue**: Feed each of the 159 previously selected training events the concatenation of literal source recall cues originally used during training. **Separate literal cue**: Feed the last literal recall cue that was excluded from training for precisely the same 159 event IDs, preserving the original Pilot08 test. **Absent episode**: Feed the last literal cue of each of 71 originally held-out episode events. For each of the three conditions, compare evaluator-only candidate identity top 1, correct and accepted fraction, false acceptance on absences, cosine with the expected independently encoded target, nonzero outputs and individual case rows. Record exact input hash feature overlap and dot-product cosine between original training cue and withheld last cue per known source case.

The diagnostic must reproduce the original real Pilot08 learned-condition per-case results **field for field** for the withheld literal cue and absent groups, or fail. This guards the replay from drifting to another feature representation, accidental retraining or changed query set. Use the exact unmodified source anatomy hashes throughout.

There are several possible results: failure to identify even the trained cue means the current learned synaptic overlay lacks separable source-content capacity; correct known-cue identity but zero withheld-cue accuracy indicates a lexical input-generalization bottleneck; good known/withheld performance with absent false acceptance would suggest calibration/rejection limitations. None of these diagnoses proves semantics or autonomous memory because output stays a 256D lexical feature vector and event identity requires an **external oracle-only candidate dictionary**.

## Implementation, evidence, next action

[scripts/diagnose_flywire_imprint09.py](../scripts/diagnose_flywire_imprint09.py) and [GitHub Actions v09](../.github/workflows/direct-flywire-imprint09.yml) implement synthetic isolation tests followed by source-verified biological replay. The workflow downloads the *actual previously learned state and source case data* by original run ID rather than retraining a new graph and substituting results. Test-only synthetic mode may generate and replay its own original Pilot08 numerical checkpoint but is never described as biological evidence.

The first measured real result must be preserved as exact case-level JSON in the repo, with provenance, three paired comparisons and an explicit assessment of whether this is a storage bottleneck, an encoding/cue-transfer bottleneck, or both. Then undertake a **new separately versioned** topology-aware synaptic input mapping if and only if the diagnostic supports it. Future controls require capacity/degree-matched rewiring and a human-validated, independently written cue set. Never retroactively replace Pilot08 or claim meaning, identity or first-person experience.

## Causal limitation of heldout-cue protocol

The Pilot08 direct-learning rule changes synapses **only for presynaptic BC01 feature coordinates active in the training cue**. In the fixed direct one-hop readout, if a separately heldout literal cue has **no nonzero feature-coordinate overlap** with its own event's training cue, that event's own update cannot contribute a cue-specific readout. Any response to that heldout cue must come from shared updates attributable to other events (or happenstance collisions), not retrieval from the intended event's own weight changes.

This is a mathematical implication of the current implementation, independently of whether the original physical support comes from real FlyWire or a synthetic mask. The synthetic source-linked diagnostic found 143 of 159 (89.94%) tested training/withheld literal-cue pairs with no shared selected BC01 feature coordinate, so the original heldout-cue criterion was strongly mismatched to the one-hop lexical input binding rule. An exactly repeated cue or a perturbation retaining lexical coordinates is a necessary **storage/readout sanity test**, but it is not evidence of semantic generalization. Future independent distinct-cue recall requires either training explicit multiple cue bindings for an event, an encoding that brings independently paraphrased cues into overlapping coordinates, or a nontrivial internal learned cue association mechanism. All would be separately versioned interventions and compared to controls.

The full real-v783 checkpoint replay remains required to measure how much source-specific information was actually stored and decodable when its input cue is repeated exactly. Do not equate nonzero synaptic weights or cosine correlation with verified identity retrieval.
