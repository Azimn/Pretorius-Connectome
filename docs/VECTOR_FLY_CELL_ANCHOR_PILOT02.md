# Vector Fly Pilot 02: anatomical neuron-class lexical cue routing

**Status:** pre-results protocol plus runnable code and GitHub Actions workflow. Do **not** claim biological findings until an executed measured run succeeds. This is separate from the completed Pilot 01 synaptic overlay; no recorded data, memory narrative or learned weights are modified.

## Motivation and prior registered measurements

The prior shared-L2-v2 FlyWire Pilot 01 real mushroom-body experiment measured 7/67 correct-and-accepted positives for the actual frozen anatomical graph, 7/67 for the trained actual graph, and 3/67 for both frozen and trained rewired controls. That suggests a possible **topology-dependent** effect under the specified lexical encoder. It provides **no demonstrated gain from train-only plasticity**. Contradiction false acceptances also differed (trained real 11/67; trained rewired 8/67), so gains cannot be reported in isolation. These previously examined, assistant-authored prompts are post-hoc diagnostics and do not establish biologically specific memory.

The next question is whether constraining the *input cue population*, rather than shuffling neuron targets uniformly across the mushroom-body adjacency, changes retrieval. This step tests a constrained graph-entry hypothesis rather than asserting language is encoded in fly neurons.

## Pinned biological sources

- FlyWire v783 original CSR converted using existing checksum-verified original release acquisition, unchanged synaptic-count input arrays, and `MB_` neuropil selection.
- Official [FlyWire neuron annotation repository](https://github.com/flyconnectome/flywire_annotations), pinned to **v3.2.0**, commit `a83b2776d60d5764cef36b927f5f9679c16c47a2`.
- Exact source file `supplemental_files/Supplemental_file1_neuron_annotations.tsv` at **Git blob SHA `02e72f6c8161d3465f77fec0edf96c5d98027a9e`**. Download only this pinned revision, never the moving `main`.
- The stable `root_id` join selects only cell labels present in the **already imported** neuropil-local network. Exact matching by label substring is a *computational selection rule*, not validation of cellular function.
- The canonical 450-record autobiography, frozen L1, deterministic train-only shared TF-IDF L2 v2, and split-seed checks are reused without rebuilding encoders.
- Annotated cell metadata came from Schlegel, Yin, Bates et al., “Whole-brain annotation and multi-connectome cell typing of Drosophila,” Nature 634 (2024), DOI 10.1038/s41586-024-07686-5. Annotation release v3.2.0 is a later tagged revision than the original paper; both are explicitly recorded.

## Prespecified hypothesis and controls

**Exploratory directional hypothesis:** Lexical cues hashed deterministically into MB graph neurons whose pinned `cell_class` label includes `Kenyon` will produce more correct-and-accepted source identifications than cues sent to a disjoint, same-sized degree-stratified MB neuron pool, when all else (graph, source documents, retrieval parameters, seed and training rule) is held fixed.

1. Baseline original whole-network lexical TF-IDF and no-restriction graph score using shared deterministic L2 v2.
2. Frozen anatomical graph with **only the feature-to-neuron hash pool changed** to the official-label-selected class.
3. Frozen same-graph control with a disjoint pool of the *same cardinality*, sampled with nearest available log2(in+out stub count) bins; preserve the selected/control bin mismatch report.
4. Same paired class/control setup after **exactly the same frozen Pilot 01 coactivity overlay** applied independently to existing edges using training-only narratives. No model-fitted or labeled semantic embeddings.
5. Three episode-disjoint seeds 31, 37, 43; all true positive, contradiction and absent-event metrics recorded alongside paired prediction changes. The held-out challenge must never influence feature fitting or the synaptic training rule.

The null preserves **input pool size, non-overlapping membership, existing physical graph and frozen tuning**. It does *not* guarantee exact degree matching when no candidate in a degree bin remains; the system publishes exact-match rate and mean/max bin differences and never silently calls an approximate null “perfectly degree matched.” It also does not preserve Kenyon-cell biology. Those are limitations of this first input-routing experiment.

## Identifiability restrictions

The feature hashes remain arbitrary symbols; they do not encode odor, compartment identity or known fly mushroom-body activity. Differences could arise from neuron degree, collision rates, embedding distribution, neuroanatomical community structure, or even the choice of labeled cell pool. This is **not** an independent test of synaptic memory learning or verified persona continuity.

Positive results require replication with biologically meaningful inputs, class-preserving randomizations, and an independently authored, human-reviewed retrieval set. Existing Pilot 04 questions are a *development-only assay*. Stop further hyperparameter fitting against these reused prompts and report negative outcomes explicitly.

## Reproduce

```bash
python -m unittest discover -s tests -p test_cell_anchors.py -v
python scripts/download_vector_fly_annotations.py --output data/flywire_v783/neuron_annotations_v3_2_0.tsv
# Use existing verified FlyWire downloader and MB graph converter.
for seed in 31 37 43; do
  python scripts/build_shared_memory_l2.py --seed "$seed" --output-dir "data/derived/vector-fly-l2/seed$seed"
done
OPENBLAS_NUM_THREADS=1 python scripts/run_vector_fly_cell_anchors.py \
  --topology data/derived/flywire_v783_mb_csr.npz \
  --annotations data/flywire_v783/neuron_annotations_v3_2_0.tsv \
  --cache-root data/derived/vector-fly-l2 \
  --seeds 31,37,43 --class-field cell_class --class-match Kenyon \
  --output results/associative/vector-fly-cell-class-pilot02.json
```

The [workflow](../.github/workflows/vector-fly-cell-class-anchors.yml) tests synthetic fixtures before acquiring biological data. Its full-per-case JSON must eventually be committed as an immutable file in `results/associative/runs/`, alongside a measured report, original run ID, source checksums and next-stage disposition. A transient CI artifact alone is not a complete archival record. If the fixed Kenyon population is absent from the pinned release under the selected label, stop and record the exact source schema; do not silently alter the target definition to optimize results.
