# Direct FlyWire Pilot 10: explicit multi-cue synaptic binding

**October 8, 2026. Status:** implementation and synthetic/real GitHub workflow prepared. **No actual Pilot10 real biological results are claimed before CI passes.**

## What we learned and why this is a separately versioned experiment

The unmodified real [Pilot08](../results/imprinting/PILOT08_REAL_V783_RESULTS.md) wrote content-dependent signed numerical weights on 25,780 existing real FlyWire edges, but recovered **0/159** memories from the source's separate, never-trained last literal recall cue. [Pilot09](FLYWIRE_DIRECT_IMPRINT_PILOT09.md) replayed the exact original real learned checkpoint, without retraining, and measured **46/159** correct event IDs from the *original trained input* versus **0/159** from a separately withheld last cue. A shuffled-content control recovered **1/159** from the training cue. Critically 143/159 cue pairs had no overlapping active coordinates in the stateless 256D lexical-hash input, meaning the original one-hop synaptic learning rule had no direct path for the intended event's synaptic changes to influence a completely disjoint cue.

We therefore do not use Pilot08's zero as evidence of absolute failure to imprint memory, and we do not call it semantic generalization. Pilot10 tests whether **explicitly training several literal source cues to the same content target** raises numeric content recoverability when prompted by each *trained* cue, while retaining negative-event rejection and independent novelty controls.

## Original frozen inputs

The 450 reconstructed Pretorius v12 memories, source order/IDs/27 episodes, published immutable L1 archive and original stateless BC01 450 x 256 signed lexical cache remain unchanged. The real primary topology is checksum-verified publisher FlyWire v783 full-brain graph with exactly 139,255 neurons, 15,091,983 aggregate directed connections, and 54,492,922 integer biological synaptic contacts. All four original arrays are hashed and checked before/after learning and readout. A separately saved sparse numerical delta is the *only* learned neural state, indexed by original existing directed edges. Do not import BioCircuit recurrent weights or modify biological wiring.

Use seed 31 and exactly the same episode-disjoint source split as prior pilots: 317 train memories, 62 validation-absent and 71 test-absent events; evaluate **159** train-event probes selected by the exact original Pilot08 random permutation and **71** fully absent test-episode probes. Source sidecars were not human-reviewed semantic descriptions.

## Training intervention and matched computational controls

Keep the original Pilot08 model's deterministic random pre/post-neuron assignments and top-8 cue-feature / top-32 target-feature coding. Write each train event's own signed BC01 narrative-content representation to the same synaptic overlay using three separate source-authored literal cues (first, middle and last). There are 337 three-cue records and 109 four-cue records in the 450-event corpus, two with six cues and two with only two. For the rare two-cue cases, repeat the last cue and report the number represented in the actual 317-train set. All methods share the same original graph and fixed feature-neuron map.

For matched exposure, use exactly **three imprint calls per training event at gain/learning rate 0.7/3**, rather than increasing the nominal sum of rate from original 0.7. Compare explicit three-cue bindings with (a) three exposures to the *old concatenated first-N-minus-last literal cue*, using exactly the same numerical rate, (b) three-cue presentations with a deranged mismatched narrative-content target per event, and (c) an untouched zero-delta baseline. The control equalizes number of presentations and nominal rate budget, **not exact count of modified edges, capacity, source feature frequencies, or final weight norms**. All of those observed values are reported to avoid mislabeling the comparison as fully parameter matched.

This comparison **does not** establish that the fly-specific biological wiring is better than a synthetic or rewired network. A separately designed support/degree/capacity-matched rewired real whole-brain condition remains mandatory before any topology-specific claim.

## Predeclared probes and scoring

On the exact same source-linked test events, evaluate first source cue (explicitly trained in the multi-cue condition), last source cue (also explicitly trained in multi-cue, but **not trained** as a separate cue in repeated-concatenation control), and a **controlled lexical deletion** of the last source cue (remove the final whitespace-delimited token, or one final character for single-word cues). Deletion is a mechanically perturbed lexical probe, NOT a novel semantic paraphrase. Probe unchanged withheld-episode last cue for false acceptance, and retain source-level train/validation/test separation.

The neural model only receives the textual cue, a frozen stateless encoder, original graph and its learned signed sparse synaptic overlay. It cannot receive the source text, memory IDs, ID dictionary, target codebook or previously computed document vectors at inference. An **external diagnostic oracle** compares the readout to content vectors for the train event pool to score correct event identity; this is not autonomous first-person recall. No source narrative is generated.

Primary metrics: correct top-1 within the external oracle's 317 training-event candidate pool, correct-and-accepted fraction at Pilot08's fixed old threshold **0.0082783**, absence false acceptance, fraction of nonzero outputs, mean cosine to actual content target, number of synapses with learned nonzero deltas, total directed-edge update events and exact source-bound state checkpoint hash. The threshold is a frozen reference, not optimized for Pilot10. Do not interpret it as a calibrated posterior probability or claim reliable rejection.

## Reproduce and decision

[Runner](../scripts/run_direct_flywire_imprint10.py) and [Actions workflow](../.github/workflows/direct-flywire-imprint10.yml) first execute unit guards plus synthetic four-condition evaluation, then download and checksum-verify the original publisher FlyWire data and run the same four conditions on the full original whole-brain topology. The real-data assay uses:

~~~bash
python scripts/run_direct_flywire_imprint10.py \
  --topology data/derived/flywire_v783_csr.npz \
  --bc01-dir data/derived/direct10-bc01 \
  --seed 31 --cells-per-feature 32 \
  --checkpoint data/derived/direct10-real-multicue.npz \
  --output results/imprinting/direct10-real-v783.json
~~~

Never report the synthetic graph result as biological. After a successful actual-v783 Actions job, preserve original machine-readable per-case scores and learned-state checksum permanently in GitHub, not only in its expiring Actions artifact.

If multiple trained lexical cue surfaces produce above-shuffled event identity while token-deletion/generalization remains poor, prioritize better cue embedding and same-memory binding mechanisms. If even trained cues do not separate well from shuffled despite neural updates, prioritize source-to-neuron routing capacity and degree/support-matched controls. The result does not imply semantic understanding, human autobiographical recollection, fly cognition, biological synaptic dynamics or definitive Pretorius.

[Master direct imprint research Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17).
