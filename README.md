# Pretorius-Connectome

An experimental biologically inspired neural substrate for Pretorius, investigating direct synaptic memory imprinting, persistent identity, neural plasticity, and behavioral continuity.

## Current experimental status

The v12 autobiographical archive contains 450 reconstructed first-person memories across 27 episodes. These are the frozen Pilot 01 inputs, not evidence of learned neural memories. The experimental protocol, source checksums, limitations, and reproduction instructions are in [Imprinting Pilot 01](docs/IMPRINTING_PILOT_01.md).

Pilot 01 learns associations between hashed autobiographical cues and opaque narrative fingerprints in a **fixed synthetic topology with a separate synaptic-weight overlay**. It compares 50, 100, 200, and 450 records against shuffled-target, unmodified, and retrieval-only references. Its output is measured by an evaluator with access to a candidate codebook. It cannot generate recollections or establish autobiographical identity.

The repository also includes FlyWire import and connectivity conversion tools, but Pilot 01 does **not** run on the FlyWire biological connectome. Biological synapse counts must remain unmodified if a FlyWire-derived topology is introduced later.

## Pilot 02: robustness and absent-memory rejection

Pilot 02 tests lexical perturbations, episode-disjoint absent-memory rejection, feature ablations, and representational interference. Its results reveal brittle transfer: the full synthetic overlay achieved about 28.6% mean identification of learned memories with word-swapped cues, compared with about 89.4% for a conventional lexical retrieval reference. Rejection of untrained episodes was unreliable. These are engineering findings, not evidence of semantic recollection or a simulated FlyWire brain. See [Pilot 02 protocol](docs/IMPRINTING_PILOT_02.md) and [measured findings](results/imprinting/PILOT02_RESULTS.md).
