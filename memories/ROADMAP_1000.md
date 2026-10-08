# Roadmap from 385 to 1,000 memories

**Current baseline:** v9 with 385 memories across 24 episodes. **Remaining:** 615 memories. **Absolute stop:** before Pretorius reconnects with Henry Frankenstein in June 1899.

The last 48 new memories cover 1869-1898. Their emphasis is practical survival after dismissal, relationships with recurring adults, long-term laboratory errors, sensory experience, spiritual inquiry, inconvenient acts of care, obligations and missed opportunities. All 337 previous event texts are unchanged.

## Approximate remaining subject allocations

| Focus | Future records |
| --- | ---: |
| Laboratory work, maintenance, failed prototypes, scaling, aesthetic mistakes | 245 |
| Occult inquiry, mysticism, alchemy, ambiguous evidence | 85 |
| Sensory life, health, food, weather, everyday practices | 82 |
| Recurring friendships, queer intimacy, collaborators and commitments | 90 |
| Dreams, sensory recollection, associative memories | 50 |
| Travel, correspondence, peculiar ethics, obligations and local life | 63 |
| **Remaining** | **615** |

Categories overlap naturally. Give each event specific perceptions, expectations, choices, and downstream consequences without mechanically turning it into a personality lesson. V9 builds relationships and habits over thirty years, not a sequence of isolated triumphs.

## University arc handoff

Do not add new memories before 1869 until the owner delivers the separately authored university history. Integrate that narrative on its own terms, preserving earlier text where possible, logging incompatible statements, and never inserting reunion, Bride or Frankenstein Village future content into the past. The full circumstances of the 1868 dismissal and the secret homunculus formula may remain uncertain.

The German occult research context can be drawn from Andreas Sommer's 2013 study (https://pmc.ncbi.nlm.nih.gov/articles/PMC3580885/) and the Sphinx journal archive (https://www.leo-bw.de/en-GB/detail/-/Detail/details/DOKUMENT/ubf_digitalisate/sphinx_ga/Sphinx%20Monatsschrift%20f%C3%BCr%20Seelen-%20und%20Geistesleben), beginning in 1886. These historical sources are research anchors, not evidence the fictional events occurred.

At 400, 600, 800 and 1,000 events, check chronology, participant first appearance, duplicate coverage, experiment consequences, sensory and relationship variety, stable event IDs, original text preservation, and the stopping boundary. Train, validation and holdout splits must avoid letting near-duplicate narrative clusters leak across evaluations.

## 2026-10-08 sensory and associative coverage gate

A separate audit now tracks the first 385 events without changing their first-person accounts. See [sensory and associative readiness audit](logs/v9/SENSORY_AND_ASSOCIATIVE_AUDIT.md) and [optional sidecar JSON Schema](docs/MEMORY_ASSOCIATION_SIDECAR_SCHEMA.json). The narrative already contains sensory anchors but lacks consistent machine-readable sensory modality, physiological state, canonical cue and typed-link annotations. These **must be reviewed**, not guessed from lexical markers. The audit found 1,253 cue occurrences, but 1,174 distinct exact lowercased strings, so we should normalize shared cue identities while retaining original surface phrases.

The **615 remaining records** remain a flexible planning count until the separately authored university arc arrives. Concentrate current expansion in 1869 through the final pre-reunion event. Distribute the next work through ordinary tower life (1869-1874), longer prototype families and failures (1875-1880), psychic and occult inquiry beside scientific methodology (1881-1886), life and ethics in the laboratory (1887-1892), mature relationships and scale-up contradictions (1893-1897), and late-career synthesis (1898 to the boundary in 1899). Do not insert isolated manufactured recollections merely to satisfy quota.

At each milestone audit not just word count and scene uniqueness, but also participant/cue recurrence, contextual trigger diversity, the difference between inferred and perceived sensation, habit and expectation changes, linked consequence propagation, and false-memory contamination. The long-term integration gate is a demonstration that encoded memory changes future behavior relative to no-memory, shuffled-association, and retrieval-only controls, not a demonstration of evocative prose.
