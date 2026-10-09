# Pilot11: frozen original biological FlyWire novel literal cue and rejection

Original publisher-verified biological workflow: [run 37870758781](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37870758781).
Full unmodified original per-case JSON: [source](runs/direct-flywire-imprint-pilot11-run37870758781.json).
Raw case JSON SHA-256: b55a467e2e6ac8e06d88848a39b5c074b611721ff26e56fd27489957de10b70e
Original learned source-bound checkpoint SHA-256: fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a
Original synaptic state unchanged: 139,255 FlyWire neurons, 15,091,983 aggregate directed edges and 54,492,922 integer biological synaptic contacts.

Genuinely untrained source literal cue eligible test memories: 31 out of 159.
Original acceptance threshold: 0.0082783. New threshold: 0.8043096000000001, selected using only 158 non-test training positives and 62 absent validation events; allowed <=6 false accepts in validation.

| Test condition | N | Original correct top1 | Original correctly accepted | Original absent false acceptance | Validation-gated correctly accepted | Validation-gated absent false acceptance |
|---|---:|---:|---:|---:|---:|---:|
| 159 original trained last cues | 159 | 0.125786 | 0.113208 | n/a | 0.000000 | n/a |
| Trained cues on unseen-eligible subset | 31 | 0.129032 | 0.129032 | n/a | 0.000000 | n/a |
| Genuinely untrained literal source cue | 31 | 0.000000 | 0.000000 | n/a | 0.000000 | n/a |
| 71 truly untrained-episode absent events | 71 | n/a | n/a | 0.788732 | n/a | 0.070423 |

Untrained source cues with no overlap with any of the three trained lexical cue coordinates: 30/31.

**Scientific boundary:** Neural inference receives the cue only and returns a numerical BC01 lexical content vector. Event IDs and accept/reject similarities are EXTERNAL oracle-only diagnostics. A fourth source cue is not an independently human-reviewed semantic paraphrase, and 10% validation false acceptance is not a guarantee for independent test episodes. The original fly synaptic anatomy and saved learned overlay were not changed. No fly cognition, autonomous recollection or uniquely biological advantage has been established.
