# FlyWire Trace-Plasticity Pilot 01: first train-only graph reweighting experiment

**Status (2026-10-08):** protocol and executable implementation; **real biological results pending completed CI and archive**. No outcome, test pass, inference run or topology advantage is claimed here.

## Research question and frozen decision

The previous verified v783 whole-brain and MB-tagged neuropil retrieval runs did not outperform capacity-matched rewired controls on accepted autobiographical recall. This experiment asks whether **training-dependent, graph-constrained synaptic weight reweighting** makes real FlyWire topology more useful than a matched rewired topology when both encounter the same Pretorius narratives.

**Working directional hypothesis, fixed before results:** the original MB topology with an activity-based weight overlay will have more *correct-and-accepted true autobiographical identifications* than the same training rule applied to a support-and-degree-matched MB null. Gains must be reported alongside contradiction and absent-episode false acceptance, positive top-1, score thresholds and paired per-case changes. Improvements achieved only by more false confirmations do not establish useful memory. A null result remains publishable.

This is a minimal computational plasticity probe, **not** a model of dopamine, local inhibition, Kenyon cell classes, neurotransmitter signs, STDP or true biological memory formation. The anatomical MB_ neuropil mask selects contacts by location, not by neuron functional class. The current L2 TF-IDF feature-to-neuron hash is not anatomically aligned.

## Immutable shared resource boundary

* Original 450 reconstructed narrative events / 27 episodes, v12, source Git blob 718dcc2d5ba4feccdef1690d447edfcebaa9bfb5, and v12 sidecars remain untouched.
* The already published canonical L1 gzip and manifest at artifacts/shared_memory/v1 are consumed and verified using existing load_v12 guards. L1 is a **17-field projection** and does not replace original L0 narrative/causal records.
* Reuse the *existing* train-only source-pinned TF-IDF L2 via scripts/build_shared_memory_l2.py, one cache per seeded train-episode partition. Following PR #15, the real assay requires **shared TF-IDF L2 v2**, with explicit frequency and lexical tie-breaking. Each cache rejects different fitted splits and carries a pinned encoder hash. The historic uncached max_features=8192 vectorizer is a DIFFERENT feature-selection baseline and must never be assumed equivalent to v2. Two independently built v2 caches with different PYTHONHASHSEED values must produce identical feature shards and full case/weight audit results.
* BioCircuit's **independently versioned 450 x 256 BC01 lexical sensory cache added in PR #12 is retained**, but not used as a substitute for TF-IDF in this particular graph assay. It is signed, lexical, and not a drop-in nonnegative graph input. The BioCircuit recurrent weights are not transplanted.
* The original FlyWire v783 root IDs, original converted CSR indptr, edge indices, and integer synapse counts remain unchanged. A *separate float64 transition operator* is created in RAM for each condition. No trained overlay is serialized over the biological source.

## Exactly specified mechanism (precommitted)

For each episode-disjoint seed 31, 37, 43:

1. Fit or verify train-only TF-IDF word/bigram features. Take the existing deterministic hashed feature-to-neuron map and the existing positive log1p contact-count transition P. The shared-cache-root option reads source-pinned, split-specific L2.
2. For each **training narrative only**, form L2 TF-IDF presynaptic activity and keep at most 256 active neurons. Construct a one-hop postsynaptic activity vector using P and keep at most 256 active postsynaptic neurons. Both are L2 normalized. Never expose withheld queries or labels to the learning rule.
3. Accumulate nonnegative coactivity for each **existing directed edge**: trace(i,j) = sum over training memories of pre(i) * post(j). No new edge can be created. Preserve every unmodified synapse count separately.
4. Scale the coactivity by its **global maximum** across that condition, multiply each existing normalized P weight by 1 + 2.0 * trace/max(trace), then re-normalize outgoing rows. If no trace exists, retain P exactly. This is a bounded deterministic Hebbian-like heuristic, not spike timing.
5. Recompute train-document graph representations using the learned operator. Query activity is propagated by the same learned operator **without online updates**. The original baseline graph diffusion remains two steps, activity cap 256, diffusion fraction 0.4 and hybrid fraction 0.25.
6. Independently fit the **same rule, gain and capacity** on a new directed edge-swap null using seed+900. Swaps are permitted only between distinct source neurons when neither new edge already exists and neither creates a new self-loop. This retains **the exact effective number of unique CSR edges per row**, global target in-stub degree, original source out-stub degree, root IDs and original row-wise synapse-count multisets. It deliberately changes the biological edge destinations, and does not preserve weighted target indegree, neuron-class structure or motifs. We retain the old unconstrained target-stub permutation in historical runs, labeled as a different weaker null.

Compare nine fixed conditions on the identical cases and per-condition source-decision-calibrated thresholds: narrative TF-IDF, frozen real graph, frozen real hybrid, frozen rewired graph, frozen rewired hybrid, learned real graph, learned real hybrid, learned rewired graph, learned rewired hybrid. Each result retains the **complete source-linked per-case predictions**, acceptance decisions, and changed-correct paired diagnostics.

The learned layer's parameter count equals the original number of directed unique CSR entries after deduplication, **now exactly equal in the original and matched-null conditions**. The runner records the count of successful directed edge swaps, effective support size, training traces and the number of edges actually updated. The earlier target-stub-permutation v1 benchmark collapsed roughly 18,000 duplicate directed edges in its rewired transition and was therefore not capacity matched. The revised null is a methodological correction introduced before the final reproducible v2 execution, not a post-hoc adjustment of v2 performance. No conditional tuning against challenge labels, no new semantic embeddings, no implanted query answers.

## Scientific interpretation and caveats

The challenge is still experiments/pilot04/challenge_v1.jsonl, 68 previously inspected assistant-authored prompts over 34 source targets, not a new human-blind benchmark. The evaluation is exploratory only. The training episodes are disjoint from validation and test by seed, but pooled case observations share underlying prompts and are not independent sample points.

Positive and contradiction acceptance scores come from source-decision calibration, **not entailment**. This cannot establish that recalled facts are true, that Pretorius has a stable identity or that a biological connectome has experienced autobiographical memories. If the rewired learning control is equal or superior, state that explicitly, including raw paired cases. Biological specificity requires later annotated neuron classes and class-preserving randomized controls; the new null fixes graph capacity and raw in/out-degree but does not fix weighted indegree, cell types or synaptic motifs.

The implementation computes SHA-256 of baseline and learned transition weight arrays and records the count of coactivity-touched edges. Tests reject changes to original synaptic CSR arrays, new edges, invalid gains, nondeterminism and unexpected changes to the lexical fallback. CI requires byte-identical v2 source cache fingerprints and exact whole-result reproducibility, including raw float-array learning fingerprints, across independent Python processes. The historical uncached encoder runs only as a separately labeled synthetic diagnostic. This test does not imply v1 and v2 encoding parity.

## Reproduce

Python 3.11. Install numpy>=1.26,<3 scipy>=1.11,<2 scikit-learn>=1.5,<2 rank-bm25>=0.2,<1 pyarrow>=17,<24.

~~~bash
python -m unittest discover -s tests -p test_flywire_plasticity.py -v
PYTHONHASHSEED=13 python scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/plasticity-l2-a/seed31
PYTHONHASHSEED=941 python scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/plasticity-l2-b/seed31
PYTHONHASHSEED=13 python scripts/run_flywire_plasticity_pilot01.py --synthetic-test --seeds 31 --shared-cache-root data/derived/plasticity-l2-a --output results/associative/plasticity-synthetic-v2-a.json
PYTHONHASHSEED=941 python scripts/run_flywire_plasticity_pilot01.py --synthetic-test --seeds 31 --shared-cache-root data/derived/plasticity-l2-b --output results/associative/plasticity-synthetic-v2-b.json
python scripts/verify_flywire_plasticity_cache_parity.py results/associative/plasticity-synthetic-v2-a.json results/associative/plasticity-synthetic-v2-b.json
~~~

For the real MB topology, first run the repository's official SHA-validated acquisition and original neuropil converter. Build seeds 31/37/43 under data/derived/plasticity-l2/seedSEED, then:

~~~bash
OPENBLAS_NUM_THREADS=1 python scripts/run_flywire_plasticity_pilot01.py \
  --topology data/derived/flywire_v783_mb_csr.npz \
  --shared-cache-root data/derived/plasticity-l2 \
  --seeds 31,37,43 --gain 2.0 --trace-activity-cap 256 \
  --output results/associative/flywire-mb-plasticity-pilot01-full.json
~~~

Workflow .github/workflows/flywire-plasticity-pilot-01.yml first runs synthetic tests and independent v2-build parity, then fetches the original FlyWire data, rebuilds the MB subgraph and runs the **real** three-seed experiment. Synthetic results must **never** be reported as real biological evidence. Before marking this phase complete, record the CI run, archive the **full per-case real JSON** permanently under results/associative/runs/, add a human-readable result report and link both from results/associative/README.md and docs/RESEARCH_HANDOFF.md.

## Next gated experiment

If learning offers a signal specific to the original graph, repeat with a pinned FlyWire cell-type mapping (Kenyon, input projection, output, modulatory annotations if source verified), class-preserving nulls, independent human-authored/reviewed probes and carefully measured compute/parameter budgets. A positive result on the existing diagnostic cannot substitute for that confirmatory work. If learning does not outperform matched controls, preserve the negative result and prioritize the independent question of biologically informed representation.
