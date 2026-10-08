# Pretorius-Connectome

An experimental biologically inspired neural substrate for Pretorius, investigating direct synaptic memory imprinting, persistent identity, neural plasticity, and behavioral continuity.

## Current experimental status

The v12 autobiographical archive contains 450 reconstructed first-person memories across 27 episodes. These are the frozen Pilot 01 inputs, not evidence of learned neural memories. The experimental protocol, source checksums, limitations, and reproduction instructions are in [Imprinting Pilot 01](docs/IMPRINTING_PILOT_01.md).

Pilot 01 learns associations between hashed autobiographical cues and opaque narrative fingerprints in a **fixed synthetic topology with a separate synaptic-weight overlay**. It compares 50, 100, 200, and 450 records against shuffled-target, unmodified, and retrieval-only references. Its output is measured by an evaluator with access to a candidate codebook. It cannot generate recollections or establish autobiographical identity.

The repository also includes FlyWire import and connectivity conversion tools, but Pilot 01 does **not** run on the FlyWire biological connectome. Biological synapse counts must remain unmodified if a FlyWire-derived topology is introduced later.

## Pilot 02: robustness and absent-memory rejection

Pilot 02 tests lexical perturbations, episode-disjoint absent-memory rejection, feature ablations, and representational interference. Its results reveal brittle transfer: the full synthetic overlay achieved about 28.6% mean identification of learned memories with word-swapped cues, compared with about 89.4% for a conventional lexical retrieval reference. Rejection of untrained episodes was unreliable. These are engineering findings, not evidence of semantic recollection or a simulated FlyWire brain. See [Pilot 02 protocol](docs/IMPRINTING_PILOT_02.md) and [measured findings](results/imprinting/PILOT02_RESULTS.md).

## Pilot 03: cue representation is a bottleneck

The prespecified Pilot 03 assay compares seven encoders/controls on the same frozen synthetic topology and the same 450-record source pool, withholding full episodes per seed. Replacing exact cue-ID-dependent encoding with train-only word/character features increased mean identification from word-swapped cues to about 77.9% for IDF-weighted trigrams (legacy full-cue: about 27.1%) under the new matched seeds. Non-neural word retrieval still reached about 88.3%, and unknown-event false acceptance remained approximately 30.6%. A shuffled-memory control produced zero correct associations but substantial absent-memory discrimination, warning that out-of-set rejection alone can reflect familiar stimulus features rather than correct memories.

This does **not** establish semantic paraphrase transfer, natural language recall, or identity. See [Pilot 03 protocol](docs/IMPRINTING_PILOT_03.md) and [executed results with counterexamples](results/imprinting/PILOT03_RESULTS.md).

## Pilot 04: authored semantic and contradiction challenge

The [Pilot 04 protocol](docs/IMPRINTING_PILOT_04.md) freezes 68 assistant-authored, non-blind and unreviewed paraphrase/counterfactual questions paired across 34 existing autobiographical events. Using the unchanged Pilot 03 neural encoders and three matched seeds, the strongest lexical models identified **0% of genuine new incident descriptions**, despite 77.9% top-1 on previously tested word-swapped cues. All learned conditions had **zero correct-and-accepted** event identifications on the authored paraphrase probes. Rejecting most contradictory prompts did not demonstrate truth checking: those same models also rejected almost all true paraphrases.

This negative finding is preserved in [the executed Pilot 04 report](results/imprinting/PILOT04_RESULTS.md), with full seed- and case-level outputs in the workflow artifact. It motivates **narrative-grounded semantic encoding and verified retrieval**, not simply expanding the number of memories or synapses. Pilot 04 is a fixed synthetic connectome *overlay* assay, not FlyWire biological imprinting or evidence of persona identity.

## Pilot 05A: narrative-grounded retrieval is better, but not truth verification

Pilot 05A reuses the **already examined** Pilot 04 challenge as a transparently labeled post-hoc diagnostic and preserves the original 450-memory v12 data. It compares established BM25 / TF-IDF full-text retrieval libraries against a content-input variant of the original fixed masked Hebbian substrate with shuffled and unmodified controls. The former retain searchable story text; the latter still requires an external opaque fingerprint codebook and does not produce prose.

On three seeds BM25 retrieved the correct event for about **29.8%** of true authored descriptions and correctly identified **and accepted** roughly **16.4%**. A narrative-input word Hebbian overlay yielded **9.0%** top-1 and **5.6%** correct-and-accepted. BM25 wrongly accepted about **62.7%** of explicitly contradictory descriptions and **53.0%** of descriptions for never-imprinted episodes, so matching subject matter is not equivalent to evidence of truth.

See [Pilot 05A protocol](docs/IMPRINTING_PILOT_05.md), [measured result report](results/imprinting/PILOT05_RESULTS.md) and the GitHub Actions case-level artifact. These are **post-hoc software engineering measures, not blind semantic comprehension, embodiment or FlyWire biological synaptic imprinting**.

## Pilot 06A: evidence-gated retrieval is not yet truth verification

The Pilot 06A [protocol](docs/IMPRINTING_PILOT_06.md) requires source-linked narrative sentence evidence, optional narrow lexical-negation checks, and explicit abstention. On the **previously examined** Pilot 04 authored challenge, the BM25-based evidence gate reduced accepted contradictions from 62.7% to 1.7% but **also reduced correct-and-accepted true-event identification from 16.4% to zero**. The neural-plus-external-evidence hybrid retained only 4.2% useful recall and still accepted 7.1% of contradictions. The improvements in rejection primarily reflect aggressive abstention, not demonstrated entailment or semantic truth understanding. See the [executed results with full cautionary analysis](results/imprinting/PILOT06_RESULTS.md) and attached workflow artifact for every verdict and cited source sentence.

The original 450-record source archive and existing test challenge remain frozen. The next evidence gate needs a genuine claim-to-passage entailment/refutation model plus independently human-reviewed, newly frozen prompts. No biological FlyWire topology was tested here.

## Connectome-constrained associative retrieval v1

The [associative memory runner](scripts/run_associative_memory.py) reads the unchanged 450-record archive and compares narrative TF-IDF, directed graph diffusion, a hybrid ranker, and a target-stub-permuted connectivity control. The [protocol and limitations](docs/ASSOCIATIVE_MEMORY_V1.md) explain how to run the explicit synthetic fixture or verified full FlyWire v783 topology. Retrieved passages are evidence pointers, not proof of their entailment or evidence of learned synaptic autobiographical memory. The benchmark reuses the post-hoc Pilot 04 challenge and does not constitute independent validation.

## Persistent research handoff and continuation

**Start every new chat/session at [RESEARCH_HANDOFF.md](docs/RESEARCH_HANDOFF.md) and [tracked Issue #7](https://github.com/Azimn/Pretorius-Connectome/issues/7).** They contain the immutable dataset versions, completed pilot results and limitations, exact rerun commands, current implementation state, parallel FlyWire research lines, open blockers and next independent-validation work. Github is the source of truth; conversation history alone is not.

## Pilot 07A — pinned local CPU NLI: measured negative result

[Pilot07A protocol](docs/PILOT07_LOCAL_NLI.md) and [real three-seed results](results/imprinting/PILOT07_RESULTS.md) document the first **actual pinned pretrained CPU entailment model** layered over BM25 narrative retrieval. Across the previously studied, assistant-authored (not human-reviewed) challenge, NLI verification over BM25 top three achieved **1.67% true correct-and-accepted recall** and 0% false acceptance of contradictions, compared with BM25 alone's 16.35% correct-and-accepted recall and 62.69% false contradiction acceptance. This is primarily abstention rather than useful verified recollection. [Source-level summary JSON](results/imprinting/PILOT07_SUMMARY.json), downloadable full-case GitHub Actions artifact and [external human-review protocol](docs/PILOT07_REVIEW_PROTOCOL.md) are recorded. The review packet contains blank forms only: genuine independent authoring and adjudication are still pending. This is a separate language model verifier, **not** direct biological FlyWire memory imprinting.


## Completed FlyWire associative experiments and permanent raw evidence

Both verified full FlyWire v783 and mushroom-body neuropil experiments completed their three-seed post-hoc comparisons. Neither demonstrated better correctly accepted autobiographical recall than its own rewired-graph control. The **complete case-level JSON outputs are now committed on main**, not only attached as expiring CI artifacts. Start at the [permanent Associative Memory result index](results/associative/README.md), read the [whole-brain report](results/associative/FLYWIRE_V783_THREE_SEED_RESULTS.md) and [mushroom-body report](results/associative/FLYWIRE_V783_MB_THREE_SEED_RESULTS.md), then consult [the research handoff](docs/RESEARCH_HANDOFF.md). The [historical archive workflow](.github/workflows/associative-archive-historical.yml) passed, including permanent preservation of all five named original JSON reports. No biological plasticity or persona identity advantage has been established.
