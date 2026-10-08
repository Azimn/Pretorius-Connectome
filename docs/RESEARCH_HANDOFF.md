# RESEARCH HANDOFF — Pretorius-Connectome
_Last reviewed 2026-10-08. Repository, not this chat, is the authoritative record._

## How to resume in a new ChatGPT / Codex / Hermes session

1. Open **https://github.com/Azimn/Pretorius-Connectome** and **[Issue #7](https://github.com/Azimn/Pretorius-Connectome/issues/7)**. Read this document and the latest comments/checkboxes on that issue.
2. Read the current **main** branch SHA and its recent commits. **Do not assume the SHA written in an earlier chat is still HEAD**, because the repository has concurrent research activity.
3. Inspect open PRs and active GitHub Actions before starting a new branch. Never duplicate already merged work, overwrite new commits, or leave an unmerged green PR without recording why.
4. Locate the exact protocol and measured result for the most recent pilot. Reproduce baseline with its existing runner (below) before changing the experimental meaning.
5. Continue the first uncompleted item in Issue #7, commit documentation, code and results together, then update this handoff and tracker. Inference results must be real measurements; GitHub Actions success is not a success claim about scientific validity.
6. On transfer to another chat, paste **this repository URL and Issue #7**; this document is meant to remove the need for a lengthy chat transcript.

## Immutable inputs / provenance

- Project: a biologically inspired autobiographical associative system for the fictional Dr. Septimus Pretorius. Our goal is experimentally **measurable retained information and behavior**, not claims of consciousness.
- Corpus: **v12 450 first-person reconstructed fictional memories / 27 episodes**. Records, associated sidecars, source/event hashes and validators are under `memories/`, `scripts/validate_autobiographical_corpus_v12.py` and `scripts/run_imprinting_pilot.py`. **Do not mutate frozen originals.**
- Known semantic challenge: `experiments/pilot04/challenge_v1.jsonl`: **68 AI/assistant-authored, NOT human reviewed**, nonblind paraphrase and counterfactual questions over 34 memory IDs. Git blob `1e1c6f0273519bf3cd5868404ddfa5f783df6abd`. Used and inspected extensively by Pilots 04, 05 and 06: **NEVER** call it a newly untouched confirmatory test.
- Seeds of cross-pilot 03–06 comparisons: `31,37,43`, with episode-disjoint train/validation/test splits. To claim robust holdout later, consider disjoint narrative clusters as well.
- Synthetic fixed dense-mask neural overlay: **512 inputs × 256 SHA-fingerprint outputs**, default density 0.55; masked additive Hebbian outer products; evaluator has an **external oracle codebook** of fingerprints. These are NOT generated textual memories.
- Biological FlyWire source data and structure can be accessed through the separate import/CSR tools and associative study; earlier Pilot 01–06 synthetic overlays **do not** use actual FlyWire biological synapse topology.

## Completed results — please read source reports, not chat summaries

| Track | Reproducibility/docs | Key observation (descriptive, not confirmatory) |
| --- | --- | --- |
| Pilot 01 | `docs/IMPRINTING_PILOT_01.md`; `results/imprinting/PILOT01_RESULTS.md` | Cue-hash association to opaque narrative fingerprints in synthetic mask; limited oracle-codebook identification |
| Pilot 02 | `docs/IMPRINTING_PILOT_02.md`; `results/imprinting/PILOT02_RESULTS.md` | brittle swapped-cue memory ID, full overlay ~28.6%; lexical retrieval higher; absent-event detection poor |
| Pilot 03 | `docs/IMPRINTING_PILOT_03.md`; `results/imprinting/PILOT03_RESULTS.md` | train-only IDF trigrams ~77.9% reordered original cues, versus lexical ~88.3%, but **not** independent semantic paraphrases |
| Pilot 04 | `docs/IMPRINTING_PILOT_04.md`; `results/imprinting/PILOT04_RESULTS.md` | newly authored story paraphrases: essentially zero correct AND accepted identification by cue-only imprint; rejection alone not useful |
| Pilot 05A | `docs/IMPRINTING_PILOT_05.md`; `results/imprinting/PILOT05_RESULTS.md` | full narrative BM25 ~29.8% true-event top-1, ~16.4% correct+accepted, yet ~62.7% contradictions falsely accepted; narrative-input Hebbian weaker |
| Pilot 06A | `docs/IMPRINTING_PILOT_06.md`; `results/imprinting/PILOT06_RESULTS.md` | BM25 sentence match gate drops contradiction acceptance ~62.7→1.7%, **but correct+accepted true IDs 16.4→0%**: aggressive abstention, not entailment |
| Separate associative track | `docs/ASSOCIATIVE_MEMORY_V1.md`; `scripts/run_associative_memory.py`; `.github/workflows/associative-memory.yml` | latest concurrent work tests FlyWire CSR graph diffusion + text retrieval vs synthetic and shuffled nulls. **Separate from the synthetic imprint pilots**; inspect newest commits/results before describing its current measured status |

**Scientific guardrail:** none of these results establish a realized, self-sustaining Pretorius, free-form autobiographical recall from biological synaptic weights, or identity stability. BM25 indexing stores the original text. Imprint decoder requires externally stored narrative fingerprints. A source-matched sentence is not evidence that a claim is entailed.

## Reproduction commands from repo root

```sh
python -m pip install 'numpy>=1.26,<3' 'pyarrow>=17,<24' 'scikit-learn>=1.5,<2' 'rank-bm25>=0.2,<1'
python -m unittest discover -s tests -v
python scripts/validate_autobiographical_corpus_v12.py
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot06.py --seeds 31,37,43 --output results/imprinting/pilot06.json
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot05.py --seeds 31,37,43 --output results/imprinting/pilot05.json
python scripts/run_associative_memory.py --synthetic-test --benchmark --seeds 31 --output results/associative/synthetic.json
```

The FlyWire-enabled run **requires verified original converted connectivity** (`data/derived/flywire_v783_csr.npz`); do not substitute a synthetic graph and report biological results. The official workflows contain immutable source validations and downloadable machine-readable evidence artifacts.

## Current priority — Pilot 07

**Decision:** stop interpreting increasingly strict lexical similarity gates as factual verification. Test a compact **local CPU NLI classifier** (entailed / contradicted / unknown) against retrieved original narrative context, before claiming progress. Use an existing released checkpoint, record the exact pinned revision, license and model ID/labels, and never pay for inference APIs. Distinguish the *retriever* (which event/passage?), *verifier* (does the evidence support/refute this claim?) and *response* (answer or abstain with provenance). Use controls: plain BM25, always abstain, and optionally BM25 + prior lexical gate. Retain source passages and raw per-case scores.

**Critical benchmark limitation:** previously authored Pilot 04 questions are permissible for **exploratory post-hoc debugging ONLY**. No results on them count as blind independent validation. Follow `docs/PILOT07_REVIEW_PROTOCOL.md` for a new review set. A separate human author and two independent reviewers must actually submit/adjudicate new claims; **do not fabricate human review or presume it has happened**. Build a validator/reviewer packet, but leave status explicitly pending until valid submissions exist.

## Pending decisions / blocking factors

1. Whether CPU inference of a pinned local NLI checkpoint succeeds in GitHub Actions and within resources; no model download or inference success should be claimed without an artifact and logs.
2. The proper entailment premise span: first-person narrative can be long; truncation or selecting only one lexical sentence can erase crucial facts. Record token lengths and supporting evidence; a prediction is not a gold truth label.
3. Reviewer recruitment is outside this automated test. No external authors/reviewers have been enrolled and no validated holdout exists.
4. Semantic proof of stable choices/persona and model-to-model continuity is a **later** task. Do not use a retrieval score as a proxy.
5. Concurrent branches may update FlyWire/associative memory results; always reconcile with latest `main` and their source documentation.

## Durable workflow

- GitHub Issue #7 is the canonical progress checklist, linked PRs, and blockers.
- Every experimental milestone should have protocol, runner, tests, workflow, machine-readable results artifact, human-readable honest result report, and README/handoff updates.
- Prefer a small *complete* tested increment over a long sequence of unmerged branches.
- At every major handoff, update this document's date, current outcome and next exact command. If a chat ends, work must still be reproducible from the repository without relying on conversation history.

## Completed Pilot 07A local NLI checkpoint (2026-10-08)

**Implementation:** src/pretorius_connectome/pilot07.py and scripts/run_imprinting_pilot07.py, with tests/test_imprinting_pilot07.py and workflow .github/workflows/imprinting-pilot-07.yml. The actual CPU NLI model was downloaded and inferred successfully. The checkpoint is Hugging Face cross-encoder/nli-MiniLM2-L6-H768 at pinned revision c847a3c0e1cad93a5343183ef183f3044e3fc7c2; CPU torch 2.5.1, transformers 4.57.6. Full measured report: results/imprinting/PILOT07_RESULTS.md, with tracked compact JSON results/imprinting/PILOT07_SUMMARY.json and all case-level results in GitHub Actions run 37837161852 (artifact ID 11575997202; SHA-256 documented in report). **No human-reviewed benchmark exists**; these are prior inspected assistant-written Pilot04 cases.

**Actual three-seed result:** BM25 correct-and-accepted recall 16.35% but contradiction false acceptance 62.69%; local NLI at BM25 top-1 produced 0% correct-and-accepted recall, and NLI reranking top-3 produced only 1.67%, with zero contradiction false acceptance. This mainly reflects abstention and is NOT verified autobiographical recall.

**Next technical experiment after Pilot07A:** inspect retriever candidate coverage (gold memory among top 1/3/10); compare relevant source passage/window extraction rather than whole-event text as NLI premise, without silently tuning against known benchmark labels. Preserve true recall / false-acceptance tradeoff and all source citations. **Next confirmatory human step:** recruit genuinely separate prompt author + two independent reviewers, adjudicate and freeze new prompts under docs/PILOT07_REVIEW_PROTOCOL.md. The generated 16-event / 64-slot reviewer packet is deliberately blank and available as GitHub Actions artifact. Do not assert its completion until verified.

**Concurrent main-line work:** other active commits published real FlyWire v783 three-seed and mushroom-body/neuropil experiments in the associative-memory track. These findings are independent of Pilot07A and MUST be preserved on merge; inspect current main docs and results before drawing conclusions.


## Complete associative-memory archive and next decision (2026-10-08)

The associative-memory line is now reproducible from GitHub without this conversation. The permanent [result index](../results/associative/README.md) links all human-readable reports and **five complete original machine-readable files** under `results/associative/runs/`. They were recovered from completed GitHub Actions artifacts by [successful archival run 37838615987](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37838615987) and committed to main. These full case-level JSON files and original MB extraction metadata will not expire with Actions artifact retention. Re-run the historical archival script only if verification or recovery is needed; it is idempotent for the five pinned run IDs.

**Measured whole-brain v783 study:** 139,255 neuron IDs, 15,091,983 directed aggregate connectivity entries, seed 31/37/43 post-hoc comparisons. The real FlyWire graph, rewired control, and lexical text baseline each identified 15/67 positive paraphrases correctly (22.4% top-1). The actual graph accepted 23/67 contradictions falsely versus lexical baseline 16/67 and rewired control 18/67. Read [whole-brain findings](../results/associative/FLYWIRE_V783_THREE_SEED_RESULTS.md) and the [complete cases](../results/associative/runs/flywire-full-three-seed-run37836107318.json).

**Measured mushroom-body neuropil study:** v783 `MB_` rows yield 14,025 incident neuron IDs, 574,660 directed aggregate pair entries and 1,535,281 synaptic contacts. The real MB graph correctly identified 13/67 positive paraphrases, versus 15/67 for the MB rewired control, and both accepted 7/67 correct positives. Both graph variants falsely accepted 10/67 contradictory prompts, so this is not a biological-wiring-specific improvement. Read [MB findings](../results/associative/FLYWIRE_V783_MB_THREE_SEED_RESULTS.md), [complete cases](../results/associative/runs/flywire-mb-three-seed-run37836674947.json), and [exact conversion report](../results/associative/runs/flywire-mb-conversion-run37836674947.json).

**Research choice:** do not build another general-purpose random feature-to-neuron topology mapper. The anatomical subgraph is a useful input dataset, but it has not shown a specific benefit. The next differentiated experiment, if prioritized, is cell-class-aligned input allocation and explicit plasticity with pinned FlyWire neuron annotations, plus properly matched randomizations of topology, cell labels, compute and capacities. Neither report justifies claims about learned biological memory or Pretorius identity. Parallel Pilot 07 local NLI work also remains separate; its human-reviewed gold evaluation is not complete.

**Operational reminder:** for EVERY new associative benchmark, commit (1) original frozen input identifiers / checksums, (2) code and CI run, (3) full per-case JSON, (4) compact summary and honest negative or positive interpretation, and (5) links in [this permanent index](../results/associative/README.md). An expiring workflow artifact by itself does not satisfy durable publication.


## Shared BioCircuit/FlyWire source integration, verified 2026-10-08

[Canonical shared-memory L1 report](../results/shared_memory/L1_INTERFACE_RESULTS.md) records working interoperability. [Portable manifest](../artifacts/shared_memory/v1/manifest.json) and [gzip archive](../artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz) are now committed directly to main, pinned to original v12 source Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5` and verified by [canonical CI 37839291647](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37839291647). The FlyWire `load_v12` and benchmark CLI can optionally load verified L1 without changing historical defaults. The BioCircuit BC01 CLI consumes the same artifact and has a [successful 450-event parity run 37839516626](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37839516626). See [cross-project Issue #10](https://github.com/Azimn/Pretorius-Connectome/issues/10). The **L2 feature cache is NOT interchangeable or implemented**. BioCircuit signed hashed lexical input and FlyWire TF-IDF remain distinct; preserve split-specific training and biological CSR original synapse counts. Performance comparison remains to be measured, not presumed.

## Shared memory L1/L2 integration status (2026-10-08)

Canonical 450-record source archive L1 is already published under `artifacts/shared_memory/v1/` and must never be overwritten by a new normalizer. [PR #13](https://github.com/Azimn/Pretorius-Connectome/pull/13) merged source-owned train-episode-fit TF-IDF L2 at commit `06bece269459a43d9e4ed09e5baabbacbd2082f7`, with `src/pretorius_connectome/shared_memory_l2.py`, `scripts/build_shared_memory_l2.py` and `.github/workflows/shared-memory-l2.yml`. Verification CI [37839758698](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37839758698) passed 2 original L1 tests, 6 L2 tests, two 450-event fit-split builds and exact synthetic-graph cached-vs-legacy baseline result parity. The full report, timing and limitations are [here](SHARED_MEMORY_L2_IMPLEMENTATION.md). BioCircuit independently consumed the identical cache through its optional L3 adapter; [18 green consumer tests and null neural-lesion results](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37840112970). Tracking issue [#10](https://github.com/Azimn/Pretorius-Connectome/issues/10) can be considered complete for the executable shared feature path, not for evidence of neural recall or semantic verification. Preserve immutable original FlyWire v783 CSR counts and historical experiment interpretations.
