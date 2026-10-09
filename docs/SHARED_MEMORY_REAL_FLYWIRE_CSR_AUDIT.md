# Real FlyWire v783 shared-feature CSR invariance audit

**Status, October 8, 2026:** Source-pinned executable assay and CI implemented; **real biological outcome pending run evidence**. Tracker: [Issue #10](https://github.com/Azimn/Pretorius-Connectome/issues/10). Separate from the direct synaptic-memory imprinting experiment and from the optional graph plasticity PR #14.

## Scope

The cross-project L1 and L2 reuse intervention has proven source and feature parity using a synthetic topology, but has not directly measured the **whole-brain original FlyWire v783 CSR after the shared-feature consumer runs**. This assay verifies that the real graph has not been mutated by any shared-memory consumer. This is an anatomical immutability test, not a claim of memory learning, accelerated preprocessing, semantics or a trained Pretorius persona.

The upstream original dataset is the Zenodo FlyWire v783 release pinned by scripts/download_flywire_v783.py (both files checked for exact byte size and MD5). Existing scripts/convert_flywire_feather.py deterministically produces the full-brain NPZ. The expected complete original graph has 139,255 root IDs and 15,091,983 aggregate directed connectivity entries. The source is never modified.

## Source and data handling

The 450-event v12 L0 corpus, 27 episodes and the existing published compressed L1 archive remain unchanged. First run the original L0 parser and shared-L1 parser and insist on equality of parsed Memory records. Load the separately versioned BioCircuit BC01 cache (450 x 256 float32 signed lexical hashes), verify its canonical record order and stateless fit scope, then load FlyWire's independently versioned seed-31, train-only TF-IDF **v2 rank-stable cache**, validating the source, manifest, exact seeded fit split and nonoverlapping episode partitions.

Next instantiate an actual AssociativeMemory on the unmodified whole-brain sparse topology and v2 training vectors, and run two queries in each lexical, graph and hybrid mode. This is a test of the real consumer path rather than inspection of a disconnected CSR file. BC01's 256D cache remains a distinct feature representation and is not silently projected into the FlyWire feature space.

At the moment the real CSR is loaded and after each consumer stage, compare all four biological arrays: root_ids, indptr, indices and synapse_counts, including exact shapes, dtypes, array SHA-256 and full content equality against original memory copies and the on-disk NPZ. Additionally require the original NPZ file's entire bytewise SHA-256 to remain identical throughout. Check total integer synaptic contacts and vertex/edge count. Rehash frozen L0/L1 bytes after the real consumer run. This is a **fail-closed check**, not a claim inferred from absence of obvious code mutations.

## Separate synthetic guard tests

The CI first tests mutation detection on deliberately corrupted in-memory synaptic contact counts, modified targets, and rewritten NPZ content. It also demonstrates that a separate modified numerical transition operator does not imply a change in the original anatomical arrays. A 96-neuron generated synthetic fixture exercises the full source and feature consumer flow, labeled **nonbiological only**.

Only after this suite passes does the real CI job checksum-verify the publisher dataset, convert the original complete v783 graph, rebuild the cache consumers without altering published L1 bytes, and run the full biological audit. Full machine-readable JSON includes release provenance, original NPZ SHA-256, original four CSR array SHA-256 values, total synaptic contacts, encoder versions and shard checksums, and six source-linked retrieval probes. This remains a narrow structural validation. The source's existing retrieval-quality results are not reinterpreted.

## Reproduce and completion gate

Run the unit tests with Python 3.11 and the installed numerical requirements:

~~~sh
python -m unittest discover -s tests -p test_real_flywire_shared_csr.py -v
~~~

GitHub Actions workflow: .github/workflows/flywire-shared-csr-integrity.yml. Its real job downloads and SHA/size-checks official publisher inputs, builds the whole-brain original CSR, generates isolated canonical BC01 and split-pinned TF-IDF v2 caches, and executes:

~~~sh
python scripts/verify_real_flywire_shared_csr.py \
  --topology data/derived/flywire_v783_csr.npz \
  --release-dir data/flywire_v783 \
  --shared-l2 data/derived/check-l2-seed31 \
  --bc01-dir data/derived/check-bc01 \
  --output results/shared_memory/flywire-v783-real-csr-invariance.json
~~~

After the actual CI succeeds, permanently commit the original machine-readable result (and a concise correctly measured human-readable interpretation) under results/shared_memory/, update Issue #10 to mark the **real CSR invariance** subtask complete, and link the exact workflow run, outcome and original file SHA. Do not label a synthetic pass as a real biological check or infer a pass before inspecting the real job.

## Next experiment: direct synaptic imprinting

The current shared-cache and associative retrieval path is not direct neural imprinting. The direct imprinting study should consume this pinned data and real unchanged CSR as input and learn **a separate overlay of content-dependent synaptic weights**. Test whether a withheld cue can recover a trained event identity solely from this learned state, with no searchable text index or event-ID lookup at inference. Compare against disconnected training, shuffled autobiographies, zero-learning, degree/capacity-matched rewired topology, and capacity-matched nonbiological networks. Keep original FlyWire anatomy and the new overlay as distinct versioned artifacts.

No data reuse success should be interpreted as evidence that the fruit fly originally encoded human autobiographical memories or that a learned overlay constitutes a biological or conscious subject.
