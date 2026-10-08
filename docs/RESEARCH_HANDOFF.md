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
