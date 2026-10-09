# Pilot 11: frozen real FlyWire unseen-source-cue transfer and rejection

Date: October 8, 2026. Status: predeclared protocol, code and biological workflow. No new Pilot11 biological result claimed prior to real CI.

## Scientific questions

Pilot10 on the checksum-pinned actual FlyWire v783 whole-brain graph trained 317 source memories with three separate original source literal cues per event. It identified 20/159 previously trained last-cue events with an external content-codebook evaluator, but falsely accepted 56/71 episode-absent probes. This was not semantic or autonomous autobiographical recall.

Pilot11 is a **frozen learned-state diagnostic**: no learning, no changed model, no changed source, no altered original biological CSR, no new probe selection and no changes to previous results.

First test: among exactly the original 159 Pilot10 positive test cases, identify events having 4+ original source-authored recall cues. Their Pilot10 training exposures were cue positions first, middle and last, so select the first literal original cue whose index and text were **not exposed in the three original training presentations**. Report the exact eligible-case count rather than implying all 159 have an untrained fourth cue. Compare each eligible source event with its own previously trained last cue under exactly the same model and external evaluator, and measure input-feature overlap against all three trained BC01 signed top-8 coordinates. These cues were already collected as lexical recall surfaces and are **not independently human-validated semantic paraphrases**.

Second test: determine whether empirical absent-event false acceptance can be reduced with a diagnostic threshold chosen *only from separate calibration data*. The original seed31 episode disjoint split has 317 training events, 62 validation-episode absences and 71 test-episode absences. The original 159 trained-memory test positives are held out for evaluation. Their complement of 158 train events supplies positive calibration. The 62 validation absences supply negative calibration. Pick the threshold with at most 10% validation-absent false acceptance that maximizes correct accepted calibration positives, then maximizes positives accepted. Do not inspect 159 test positives, 71 test absences or unseen fourth-cue cases in threshold selection. Compare against the frozen Pilot08 reference threshold 0.0082783 on *identical* test cases. The 10% constraint is a small-sample empirical calibration rule, not a statistical guarantee of real-world reliability.

## Provenance, anatomical and inference boundary

Use actual original publisher-verified full FlyWire v783 with 139,255 neurons, 15,091,983 aggregate directed edges and 54,492,922 integer synaptic contacts. All four original biological array SHA-256 checks must match the permanently stored original real CSR invariance result. All original root IDs, source edge targets and contact counts are untouched before and after the assay.

Use original real Pilot10 learned synaptic checkpoint permanently archived in artifacts/imprinting/checkpoints/pilot10-original-v783-run37869353939.npz, SHA-256 fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a. The complete original Pilot10 case JSON is results/imprinting/runs/direct-flywire-imprint-pilot10-run37869353939.json, SHA-256 ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8. Exactly 450 canonical L1 source events and separately versioned stateless BC01 lexical 256D vectors must match. The model has 951 original training presentations and 28,938 original existing-edge learned synaptic deltas and receives only a cue string at inference.

The neural inference function returns a 256D **lexical numerical vector**, never a narrative, event label, candidate list or truth judgment. Event-ID ranking and any acceptance threshold are exclusively **EXTERNAL oracle-assisted diagnostics**, not model inference. The trained synapses are not transferred or rewritten. The script must reproduce the source Pilot10 last-cue and absent episode event decisions with all 159 positive and 71 absent cases in unchanged order; only negligible numerical rounding is permitted. A discrepancy fails closed.

## Measured comparisons to report

Record full source-linked per-case data for familiar positive calibration, absent validation, original 159 familiar test positives, original 71 absent test negatives, paired familiar positive subset eligible for unseen literal recall, and genuinely untrained source fourth cue. For each provide the correct event top-1 fraction, correctly accepted fraction, falsely accepted absent fraction, output activation, oracle cosine, and source feature-overlap distribution under original and validation-only gates.

Do not call a small untrained source cue result semantic paraphrase learning. Pilot11 lacks a topology/capacity-matched rewired whole-brain control, a validated semantic encoder, independent human-reviewed source paraphrases and autonomous natural-language readout. This protocol cannot show fly consciousness, human autobiographical memory, biological plasticity law or a definitive Pretorius.

## Execution and permanent archival

Code: scripts/diagnose_flywire_imprint11.py. Tests: tests/test_flywire_imprint11.py. CI: .github/workflows/flywire-imprint11.yml. The synthetic smoke generates and evaluates an explicitly synthetic checkpoint to validate software but must not be reported as biological evidence. The real workflow checksum-verifies FlyWire publisher files, rebuilds canonical BC01 cache, loads previously frozen real checkpoint and original case JSON, tests original row-level parity and evaluates predeclared unseen source cues and calibration.

After successful whole-v783 CI, commit original raw full per-case JSON unchanged, its SHA-256, measured result report and handoff to GitHub. Preserve negative results as evidence. Research tracker: https://github.com/Azimn/Pretorius-Connectome/issues/17.
