# Connectome-constrained autobiographical retrieval v1

**Status:** runnable software, not an experimentally confirmed performance advantage. This is a separate associative-retrieval study, not Pilot 06 of synaptic imprinting. It does not change the frozen 450 memories, earlier benchmark files, or Pilot 05 results.

## Design

Use the unchanged v12 first-person autobiographical text as the authoritative evidence store. Fit a conventional word/bigram TF-IDF representation to indexed narratives only. Create a deterministic feature-to-neuron mapping using a stable BLAKE2b hash. Project document/query vectors to neurons, propagate activity over a sparse, **directed** graph, retain only bounded high-activation nodes after each step, and rank source narratives by graph-space cosine. A hybrid ranker mixes conventional TF-IDF and graph similarity. Quoted excerpts are verbatim retrieved source passages, not generated recollections or claims of logical entailment.

The graph is taken **unchanged in adjacency and synapse counts** from the repository's verified FlyWire v783 CSR conversion. Numerical propagation uses log(1 + synapse count), outgoing row normalization, 2 diffusion steps, 256 active nodes per record and 0.4 diffusion. These are engineering choices, **not** measurements of membrane voltage, neural plasticity, excitation/inhibition, or memory engrams. No learned synaptic overlay is implemented in this v1. A target-stub-permutation null retains source outdegree and total target-stub frequency, not weighted indegree or simple-graph uniqueness.

## Use

Python 3.11+; install numpy, scipy, scikit-learn, rank-bm25 for compatibility with the existing repository. The runner also validates the original event and sidecar Git blob pins and the preexisting challenge pins.

Synthetic smoke benchmark, explicitly NOT FlyWire:

    python scripts/run_associative_memory.py --synthetic-test --benchmark --seeds 31 --output results/associative/synthetic.json

Inspectable recall demonstration, synthetic only:

    python scripts/run_associative_memory.py --synthetic-test --query "laboratory fire and betrayal"

Real FlyWire graph, provided as converted, checksum-verified CSR:

    python scripts/run_associative_memory.py --topology data/derived/flywire_v783_csr.npz --benchmark --seeds 31,37,43 --output results/associative/real.json

To acquire original biological data use existing scripts/download_flywire_v783.py and scripts/convert_flywire_feather.py, or manually dispatch the new GitHub Actions workflow with real_flywire=true. The workflow preserves raw per-case results in its downloaded JSON artifact. The automated push job runs only a synthetic end-to-end demonstration; **it must never be reported as FlyWire evidence**.

## Scientific comparisons and limitations

The benchmark reuses 68 assistant-authored, unreviewed, already examined Pilot 04 prompts, so it is a **post-hoc diagnostic only**. Episodes are split before fitting. No test prompt, benchmark target ID or challenge truth label enters the encoder or feature mapping. Thresholds are calibrated using original 'decisions' strings from train/validation episodes. It scores five conditions: original TF-IDF baseline, graph-only real adjacency, mixed original graph, graph-only target-stub null, and mixed null.

Metrics separately report correct top-1 on positive paraphrases, correct-and-accepted positive recall, false acceptance of contradiction prompts, and false acceptance of absent-episode probes. Retrieving thematically similar evidence does not verify an assertion. Every conclusion requires checking the executed data and comparing methods using identical training episodes and seeds.

**Not accomplished by this version:** online learning or long-term synaptic plasticity; biologically plausible neurotransmitter dynamics; verified contradiction/entailment analysis; newly blinded human-reviewed challenge items; causal comparisons against independent large modern memory systems; proof of identity or biological memory.

## Decision gate

Keep the feature hashing and model capacity identical between real and null networks. Compare graph-only and hybrid outcomes on at least three registered seeds and an independently reviewed, new challenge before claiming any advantage of actual biological topology. If null or lexical retrieval equals or exceeds the biological condition, publish the negative result. Do not retune on the previously examined 68 cases and then label those cases out-of-sample.
