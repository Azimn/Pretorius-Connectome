# Pilot 07A: pinned local NLI over narrative retrieval

**Status: experimental diagnostic and implementation. Full independent human validation is NOT available.**

## Precommitted design

Question: Will a pretrained three-way natural language inference classifier help distinguish supported versus false autobiographical claims when BM25 has found a related narrative? Pilot 06A's lexical evidence gate had 0% correct-and-accepted true identification despite excellent apparent rejection.

- Data: identical immutable v12 450-event, 27-episode reconstructed fictional autobiography. Inputs and sidecar Git blobs are validated by the existing runner.
- Prior benchmark: frozen Pilot 04 challenge with 68 **assistant-authored, human-unreviewed, already examined** claims. Used only for POST-HOC DIAGNOSTICS, not blind confirmation.
- Train/validation/test: identical Pilot 05A episode-disjoint split, seeds 31/37/43. Initial CPU trial seed 31.
- Retrieve top three ORIGINAL training narratives with the existing BM25 index. Classifier premise is the exact narrative, hypothesis is the claim. No event ID, expected answer or benchmark label enters retrieval or NLI inference.
- Model: cross-encoder/nli-MiniLM2-L6-H768, Apache-2.0, 82.1M parameters. Source https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768 . Pinned 40-digit HF commit: c847a3c0e1cad93a5343183ef183f3044e3fc7c2 . Three labels MUST be contradiction, entailment, neutral. Wrong model label mapping is a hard error.
- CPU: ~328 MB model weights, PyTorch 2.5.1 CPU, Transformers 4.x, no paid inference API. Input maximum 384 tokens, truncate only source premise; truncation may omit relevant evidence and degrade predictions.
- Policies: previously calibrated BM25 baseline, NLI classification of BM25 top-1, top-three BM25 reranked by probability of NLI entailment, and always-abstain.
- Frozen cutoff: accept if entailment >=0.60 and exceeds contradiction AND neutral by >0.12. Flag possible contradiction if contradiction >=0.60 and >entailment by 0.12; otherwise explicitly return insufficient evidence. These thresholds are not optimized on the previous benchmark.
- Record raw three-way NLI probabilities, candidate ranks and event IDs, predicted event, retrieved original source excerpt, rejection reason and source provenance for each query. Evaluation labels are assigned only afterward.

No claim of true textual reasoning, truthful Pretorius recall, biological FlyWire imprint or independence follows from a score. Any reported number must come from the real model rather than mocked regression tests. The model's SNLI/MultiNLI training distribution may be quite different from fictional 1890s autobiography.

## How to reproduce

From repo root, with Python 3.11:

    python -m pip install 'numpy>=1.26,<3' 'pyarrow>=17,<24' 'scikit-learn>=1.5,<2' 'rank-bm25>=0.2,<1' 'transformers>=4.46,<5' 'safetensors>=0.4,<1'
    python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/run_imprinting_pilot07.py --mode diagnostic --seeds 31 --output results/imprinting/pilot07.json

GitHub Actions workflow imprinting-pilot-07.yml runs all unit tests, produces a review packet and performs a separate real CPU NLI inference job; archives both kinds of output with distinct artifact names. The unit suite uses a **FAKE classifier** only to test plumbing, and is NOT a model inference benchmark. If checkpoint download/inference fails, the result must remain marked blocked; there is no fallback or fabricated score.

## Genuine independent review remains a separate human step

Existing protocol: docs/PILOT07_REVIEW_PROTOCOL.md . The new reviewer packet tool creates 16 source events from episode-diverse SHA-based sampling and excludes every Pilot 04 target event. Four requested claim types per event, 64 **blank** authoring slots, two **blank** reviewer sheets, and **blank** adjudication sheets. It does not invent the authors, reviewers, questions or labels.

Run:

    python scripts/run_imprinting_pilot07.py --mode packet --directory results/pilot07/review_packet_v1
    python scripts/run_imprinting_pilot07.py --mode validate --directory results/pilot07/review_packet_v1

The structural validator will refuse unfilled case rows, missing source quotes for entailment/refutation, altered source text or repeated participant codes. Passing structural validation is NOT itself proof of external reviewer identity or independence; that must be separately verified. Reviewer-provided new cases should be frozen before any model tuning against them. Pilot 07A's already-seen challenge is only developmental and will not be relabeled as a human-reviewed holdout.

## Next decision gate

Read the actual workflow model run, download the JSON evidence, and write results/imprinting/PILOT07_RESULTS.md with correct-and-accepted true memories versus contradiction false acceptance and absent-event false acceptance. Compare BM25 to the two NLI policies and always-reject. Record failures, model version, source hashes and limitations. Then update docs/RESEARCH_HANDOFF.md and Issue #7. Never silently adjust threshold after inspecting the old benchmark and claim unbiased confirmation.
