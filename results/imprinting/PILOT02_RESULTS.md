# Pilot 02: executed results, lexical robustness and rejection

**Date:** 2026-10-08  
**Data:** frozen v12, 450 memories in 27 episodes  
**Experiment code branch:** `experiment/pilot02-generalization-rejection`  
**Validated workflow:** [Pilot 02 run 37824490635](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37824490635)  
**Foundation regressions:** [Run 37824490681](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37824490681)  
**Exact numerical output:** the `pilot02-imprinting-results` artifact attached to the Pilot 02 workflow. All results below are derived from that artifact; the full seed-level output is authoritative.

## Status

**The code executes and tests pass, but the generalization result is poor.** Strong Pilot 01 identification on original learned cues did not transfer reliably to unseen word-swapped and word-deleted cue strings. The conventional word-overlap reference outperformed the learned overlay by a large margin. The current imprinting mechanism should not be described as reliable autobiographical recall.

The experiment does not simulate the FlyWire brain. This is a fixed synthetic feedforward mask with a plasticity overlay and an external oracle content-fingerprint codebook. All memory texts are reconstructed fictional history. The original v12 records and candidate annotations remain unchanged.

## Frozen protocol and partition

All 450 events are partitioned by episode independently for each of three seeds. Train events are imprinted, validation-negative episodes are used to choose the abstention threshold, and test-negative episodes remain separate. Known-event calibration probes and known-event test probes also use distinct records. Episode grouping does **not** guarantee that related narrative content across different episodes is absent.

Primary challenge: swap the first two words of an eligible original surface cue, assign a new query-only cue identifier, and score against an oracle codebook containing **only trained** events. Secondary challenge: delete the last word, reusing the primary threshold. Both preserve lexical overlap and are **not semantic paraphrases**.

The absent-memory task asks whether the system **abstains** when an episode was never imprinted. It does not ask the network to identify an event absent from its codebook.

## Mean across three seeds

All values are proportions. These are descriptive measurements, not confidence intervals.

| Condition | Known swapped cue top-1 | Accepted known probes (TPR) | Accepted unknown probes (FPR) | Calibrated balanced accuracy | Deleted-word cue top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full imprint | 0.286137 | 0.609242 | 0.553172 | 0.528035 | 0.122608 |
| Surface-only imprint | 0.322508 | 0.679002 | 0.539695 | 0.569654 | 0.158069 |
| Supplementary-ID-only imprint | 0.004274 | 0.406797 | 0.389895 | 0.508451 | 0.003212 |
| Shuffled target associations | 0.004302 | 0.576548 | 0.494648 | 0.540949 | 0.002151 |
| Unmodified overlay | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.000000 |

**Non-neural lexical reference:** 0.894418 top-1 identification on known swapped cues, with no abstention capability; it is a separate word-overlap lookup with access to the stored candidate cue texts and not a resource-matched comparator. It is unsurprising that it benefits from preserving the original cue tokens.

**Full-imprint seed-level primary top-1:** seed 0: 0.283871; seed 1: 0.300654; seed 2: 0.273885. **False-acceptance:** seed 0: 0.329114; seed 1: 0.451613; seed 2: 0.878788. The especially high seed-2 FPR shows unstable rejection performance, despite a threshold selected from separate validation episodes. A classifier that rejects everything can achieve zero FPR but has zero true-positive rate, as the unmodified-control row demonstrates.

## Fixed earliest-50 representational interference

The same first 50 imprinted memories were evaluated against the same fixed 50-candidate codebook while imprinting continued to 450. All three seeds kept top-1 at 1.000, but the cosine and winner margin degraded:

| Total imprinted | Mean earliest-50 target cosine | Mean earliest-50 winner margin |
| ---: | ---: | ---: |
| 50 | 0.721314 | 0.491687 |
| 100 | 0.619424 | 0.414002 |
| 200 | 0.488521 | 0.312334 |
| 450 | 0.351321 | 0.196823 |

Therefore, fixed-codebook top-1 alone substantially overstates signal stability. Representation dilution/interference is visible even while a small closed-set decoder identifies targets correctly.

## Interpretation

The synthetic masked Hebbian network can memorize mappings when the original cue features are available, but has brittle lexical transfer and poor, seed-sensitive absent-memory rejection. Removing supplementary registry-ID features yielded better swapped-cue performance in this assay. That is consistent with exact cue-ID feature dependence limiting transfer, but not by itself proof of a unique causal mechanism: the surface-only condition also changes feature density and weight normalization.

The primary result is not semantic retrieval, free-response recollection, contextual choice, identity continuity, or neural tissue imprinting. The external fingerprint decoder uses the narrative texts as target codes. No biological wiring was included. The model has no independent language generator.

## Next engineering gate

**Preserve these Pilot 02 results and thresholds. Do not tune against the test episodes.** For Pilot 03, preregister an alternate representation learned or constructed strictly from permitted training cues: an explicit surface-token baseline without cue-ID hash dependence, a subword/character perturbation encoder, and a compact encoder with a controlled training budget. Keep the same number of synapses and target fingerprints across controlled comparisons before considering a scaled topology. Use newly partitioned episodes and independent human-authored paraphrases as final untouched test cases; word swaps alone cannot validate semantic transfer.

Report calibration curves, FPR at a specified TPR, top-1, cosine/margins, and performance of the conventional retriever. Establish a genuine decision/choice dataset separately before claiming altered behavior. Full FlyWire integration remains a topology follow-up, not evidence that adding anatomical complexity will fix weak cue encoding.
