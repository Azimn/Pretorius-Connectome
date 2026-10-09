# Pilot 12: frozen language-cue representation × degree-preserving FlyWire wiring

**October 8, 2026. Predeclared before original real-v783 Pilot 12 results.** Continuing cumulative direct synaptic memory research under [Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17), not a separate isolated character prototype. All results from earlier pilots are immutable, including [Pilot 11 original real case evidence](../results/imprinting/PILOT11_REAL_UNSEEN_CUE_REJECTION_RESULTS.md) and permanent [Pilots 08–11 scorecard](FLYWIRE_DIRECT_IMPRINT_PROGRAM_SCORECARD.md).

## Why

Pilots 09–11 established a sparse original FlyWire-edge overlay can learn limited event-specific associations under an **exact trained literal cue**, but **0/31** novel fourth original source cues recovered the correct event; 30/31 had no BC01 lexical input-coordinate overlap with their own trained cues. Validation-only false-acceptance control suppressed the majority of correct familiar recalls.

The next explanation to test is whether the missing cue relation comes from the **input feature representation**, and whether any benefits arise from the original FlyWire directed wiring **rather than simply a graph-shaped plastic parameter mask**.

## Frozen prior, source and content controls

All experimental arms consume the canonical v12 **450 Pretorius reconstructed first-person events** from 27 episodes, their original L0/L1 source and the exact BC01 content vectors aligned to source record IDs. Episode-disjoint seed31 split: **317** trained memories, 158 non-test trained-memory positive calibration cases, **159** trained-memory positive test cases, **62** validation-episode absent queries and **71** test-episode absent cases. Exactly **31** of the 159 original positive cases have a fourth original literal source cue that was never one of the first/middle/last three Pilot10 presentations. Only those 31 form the novel source cue comparison, paired to their own familiar last cue.

All six conditions use identical original fixed source event ordering (order_by_episode seed20261008), exactly 951 first/middle/last literal cue presentations (three per trained event), the same signed **BC01 256D narrative CONTENT vector** for each event, the same top-32 sparse target features, the original synaptic plasticity rule, input feature count top-8, learning rate 0.7/3 per exposure, clipping limit 4.0, feature-to-neuron assignment seed31, same fixed 317 target candidates in the EXTERNAL oracle scorer, and no test-case tuning.

A shuffled-control condition permutes *event content targets*, retaining the source cue text and neural graph. Event-ID labels and autobiographical narratives are **never available to neural infer(cue)**, which returns only a 256D signed numerical readout.

## Two distinct input encoders

**Original BC01 baseline**: stateless signed hashed 256D lexical cue features, precisely as Pilot10. The runner MUST re-create source-original Pilot10's per-case last-trained and absent-episode predictions, within 2e-6 score roundoff, or fail.

**New frozen pretrained sentence encoder**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0, Hugging Face model revision **1110a243fdf4706b3f48f1d95db1a4f5529b4d41**, run locally on CPU with its frozen ONNX file and pinned tokenizer. The model's 384D pooled sentence vectors undergo a fixed source-independent, label-independent random Gaussian **384→256 projection** (seed 20261008), normalization and the original top-8 signed feature selection. Neither input pretraining nor projection is updated on Pretorius texts. **This is pretrained external language knowledge**, not a learned semantic substrate inside FlyWire, and its effective ability to represent sparse gothic-era source recall cues is unknown. No subscription or inference API.

The original autobiographical target CONTENT vectors remain BC01 for both encoder arms, so the input-encoder change does not simultaneously change the learned synaptic target, oracle ranking dimension, or source memory content. The pinned model commit, full ONNX file SHA-256 and projection SHA-256 are included with each experiment. Model inference can cache cue-only feature vectors transiently for computational efficiency; this is not a target/narrative retrieval index and the neural model cannot access ID mappings or content vectors.

## Matched biological wiring null

Original publisher-verified **FlyWire v783** whole-brain CSR contains exactly **139,255** root IDs, **15,091,983** aggregated directed neuron pairs and **54,492,922** integer source synaptic contacts. Verify publisher source MD5/size and all four complete original CSR array SHA-256 values, before and after operations.

Use original model seed31 to choose the EXACT same disjoint presynaptic and postsynaptic 32-neuron-per-256-feature populations for every condition.

Construct the **degree-preserving biological-wiring null** only on the directed synaptic support from these designated presynaptic input neurons to designated postsynaptic output neurons. A deterministic seed73 double-edge switch changes only destinations among originally eligible source→readout edge slots. Every source neuron retains its eligible outgoing **binary degree**, every readout neuron retains its eligible incoming **binary degree**, the trainable edge-slot count remains identical and no duplicate directed source→target pair can be created. The original source edge-index array is NEVER changed: the null gets its own in-memory copy of outgoing destination indices.

The null **does not preserve each destination's contact-weighted incoming strength**, local motifs elsewhere in the full brain or the number of nonzero learned synaptic values after training. The original integer synapse_counts remain at the same source-row edge positions, preserving each source neuron's total contact count and the overall contact distribution, but changing their destination targets under the null. Such limits preclude interpreting a positive original-vs-null result as uniquely biological unless more controls are added. The null is nevertheless a materially stronger topology control than an unrelated synthetic graph.

## Six predeclared conditions

| Condition | Cue representation | Directed learnable wiring | Source→content pairing |
|---|---|---|---|
| real_bc01 | Frozen BC01 original lexical | Original full FlyWire | Matched |
| degree_rewired_bc01 | Frozen BC01 original lexical | Degree-switched eligible FlyWire null | Matched |
| real_minilm | Frozen MiniLM sentence input + fixed 256D projection | Original full FlyWire | Matched |
| degree_rewired_minilm | Identical frozen MiniLM encoder | Degree-switched eligible FlyWire null | Matched |
| real_minilm_shuffled | Frozen MiniLM | Original full FlyWire | Shuffled content labels |
| degree_rewired_minilm_shuffled | Frozen MiniLM | Degree-switched eligible FlyWire null | Shuffled content labels |

All six arms have *exactly the same number of nominal synaptic trainable slots*, source and event exposure schedule and per-exposure numerical learning rule. **Not** the same number of nonzero learned deltas after training or exactly equal final weight norms, so these are reported as separate resource measures. A conventional retrieval benchmark remains an external reference, not a neural inference backdoor.

## Outcome measures and falsification

On the exact original 159 test-positive records: (1) trained original last literal cue; (2) last cue with mechanical deletion of one final word or character. On the precisely 31 original positive records with valid unseen source cue: (3) their last *trained* cue as paired baseline; (4) their **never-trained fourth source literal cue**. On the original 71 episode-absent test records: (5) fixed source absent cue. Report event correct top1 and correct-and-accepted event fraction, heldout absent false acceptance, source paired feature overlap, number of learned synapse deltas and cue processing footprint.

A separately selected EXTERNAL acceptance threshold can be calibrated using only the 158 non-test train cue cases plus **62** validation-episode absent examples, constrained to no more than 10% empirical validation false acceptance, then held constant on each arm's test. Report this alongside the original fixed 0.0082783 historical threshold. Treat high external oracle correctness with unacceptable false acceptance as an incomplete outcome.

**Scientific success criteria to investigate, not yet findings:** semantic input should improve novel fourth-source-cue top1 above its paired BC01 lexical baseline and above the same semantic encoder trained with shuffled associations. The original biological wiring can be said to matter in this assay only if original vs degree-switched results materially differ **holding cue encoding and source learning constant**. Given the 31-event unseen subset and exploratory reuse of existing source recall fields, positive differences will be hypothesis-generating, not statistically definitive; require independent human-written cue evaluation, more seeds and a stricter contact-weighted null before causal claims. The mini-model can itself carry semantic knowledge and must never be mistaken for synaptic storage of that knowledge.

## Reproduce and evidence

Source code: src/pretorius_connectome/semantic_cue12.py, src/pretorius_connectome/rewire12.py, scripts/run_flywire_pilot12.py. Tests: tests/test_flywire_pilot12.py. CI: .github/workflows/flywire-pilot12.yml.

The synthetic 8,192-neuron fixture must pass first as **nonbiological engineering** before any real graph is downloaded. Then the complete original publisher v783 Feather importer must pass source SHA and original case-parity gates. Publish all raw case JSON, control graph degree statistics, frozen semantic ONNX digest, exact original synaptic weight state hashes, original unchanged biological CSR hashes and a durable research handoff. A failed pretrained download, failed rewiring degree check or source mismatch fails closed; no synthetic stand-in may be reported as whole-brain FlyWire evidence.

A positive learned semantic-input neural vector is not native first-person autobiography or a definitive Pretorius. This is an experiment in cue-controlled associative signal, not subjective character continuity.
