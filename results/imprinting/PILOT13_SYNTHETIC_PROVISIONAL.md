# Pilot13: synthetic-only source-linked memory interference — preliminary

**Status: nonbiological synthetic aggregate-neuron fixture only.** The original memory source and BC01 input/content encoders are the canonical 450 Pretorius reconstructed first-person source events (27 episodes). [Synthetic GitHub Actions 37874513816](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37874513816), stage synthetic-staged-capacity: five new and six previous relevant unit tests passed, full four-condition/seven-capacity checkpoint test passed, original output JSON artifact flywire-pilot13-SYNTHETIC-only-staged-capacity produced.

The tested neural source was a **synthetic 8,192-neuron unique-pair aggregated graph** generated and normalized from an artificial graph, not the actual publisher FlyWire v783 whole-brain connectome. Degree-preserving rewiring and all source-case/oracle invariants passed. Exactly 317 original source events were trained in the same fixed Pilot10 order with three literal source recall cues per event, 0.7/3 rate per exposure, signed BC01 top-8 cue and top-32 content feature code. At ALL loads, the event-ID selection oracle contained the same fixed **317 final candidate source memory targets** and existed *only in external evaluation*, not neural inference.

## Paired early-source retention

This test measured the **same 16 original earliest-trained source events** after loads 0,16,32,64,128,256,317. Scores are the original source memory's correct event top-1 identification out of 16 using the external target-codebook scorer.

| Total learned original source events | Original synthetic graph/correct association | Degree-switched synthetic graph/correct association | Original synthetic graph/deranged content | Degree-switched synthetic graph/deranged content |
|---:|---:|---:|---:|---:|
| 0 | 0/16 | 0/16 | 0/16 | 0/16 |
| 16 | **5/16** | **10/16** | 0/16 | 0/16 |
| 32 | 5/16 | 8/16 | 0/16 | 0/16 |
| 64 | 5/16 | 5/16 | 0/16 | 0/16 |
| 128 | 4/16 | 2/16 | 0/16 | 0/16 |
| 256 | 0/16 | 2/16 | 0/16 | 1/16 |
| 317 | **1/16** | **2/16** | 0/16 | 1/16 |

**Per-event paired attrition and source margin (from original full case JSON):** Of the original 5/16 anchors correctly identified after the first 16 source events, **one** remained correct after all 317 memories, **four became incorrect** and none newly became correct. For the rewired control, **two** of its original 10/16 correct anchors survived; **eight** became incorrect, with none gained. The mean signed *own-content minus strongest competitor* cosine margin on the identical first 16 memories declined from **−0.020592 to −0.146269** in original synthetic wiring and from **+0.014098 to −0.169297** under degree-preserving synthetic rewiring. This supports not merely shifting denominators but actual paired loss of source/event discrimination under this test. AUC over a growing intersection of the 159 preselected test memories against the same 71 absent episodes also dropped in the original arm (0.815728 at load16 to 0.594162 at load317), but **the known-case denominator varies by load**, so AUC across stages is not a fixed-sample longitudinal measurement.

**Preliminary causal interpretation:** early source-identified numeric association is materially displaced or interfered with by adding later autobiographical source memories under this exact sparse associative learning/readout. The rewired synthetic graph starts ahead of original (10/16 vs 5/16) and loses most of that lead by full training. However those tiny numbers do not demonstrate anything about the original real fly graph, semantic retrieval, human-like forgetting or native associative recall. The true source-target output margins and all original per-case rows are in the original CI artifact and must be analyzed jointly, not just the threshold-free top1 count.

The true biological-connectivity assay is independently gated behind successful synthetic CI: original publisher SHA verified full FlyWire v783 with 139,255 neurons, 15,091,983 aggregate directed pairs and 54,492,922 integer original synapse contacts. It must additionally reproduce the exact original real Pilot10 terminal source-edge weight delta matrix and historical 159 positive/71 absent event-ID decisions. **No real Pilot13 result is yet claimed by this synthetic report.**

Related preregistration: [Pilot13 protocol](../../docs/FLYWIRE_DIRECT_IMPRINT_PILOT13.md). Research issue: [#17](https://github.com/Azimn/Pretorius-Connectome/issues/17). Original inference accepts only a cue and learned edge weights and returns a 256D BC01 signed lexical vector; source IDs, ground truth and event ranking are external to it. This test is not autonomous first-person autobiographical recollection or biological neural plasticity.
