# v9 sensory and associative memory readiness audit

**Date:** 2026-10-08. **Source:** `memories/current/Pretorius_v9_385_Events_Complete.jsonl`. SHA-256: `25bdf86765287225002e55c0303c77b65908342e2b6670f0d0a43daa1046a41a`.

## Findings

The current corpus contains **385 unique autobiographical events in 24 episodes** and approximately **56,119 narrative words**. All 385 records contain the required event text, location, participants, expectations, observations, decisions, consequences, belief and relationship changes, recall cues, and links to prior events. There are **755 total prior-event links** and **1,253 recall-cue entries**. The cues contain **1,174 distinct lowercased strings**, of which **57 exact strings occur more than once** (136 occurrences of those repeated strings).

The memory texts contain considerable visual, experimental and material specificity. They do **not** yet provide consistently normalized, machine-readable sensory modalities, bodily states, dream-state flags, salience, cue identifiers, or explicit association types. Consequently, the corpus is a good autobiographical narrative scaffold but **not yet a validated neural stimulus dataset**.

## Conservative lexical spot audit

The counts are **automatic keyword-hit proxies in first-person `memory_text` only**, not manually validated perception annotations. They will miss paraphrases, include false positives such as mentioning a glass object without a visual experience, and cannot establish actual subjective sensation. They are appropriate for identifying editorial gaps, not for scoring humanlike experience.

| Proxy | 385 event hits | Corpus share | Hits in newest 48 |
| --- | ---: | ---: | ---: |
| Olfaction-associated words | 32 | 8.3% | 13 |
| Taste, meals or drink-associated words | 88 | 22.9% | 15 |
| Sound or music-associated words | 87 | 22.6% | 8 |
| Touch, temperature or physical contact-associated words | 142 | 36.9% | 31 |
| Visual appearance, color or form-associated words | 247 | 64.2% | 33 |
| Body state, fatigue, breathing or pain-associated words | 28 | 7.3% | 1 |
| Dream or trance-associated words | 17 | 4.4% | 2 |
| Occult or psychical-associated words | 16 | 4.2% | 3 |
| Laboratory or measurement-associated words | 240 | 62.3% | 33 |
| Ethical choice or autonomy-associated words | 111 | 28.8% | 11 |

The first 287 memories have no explicit `memory_kind` field. Ninety-eight later records do have one. This is a **schema consistency gap**, not evidence that earlier memories lack kinds. All existing recall cues are text strings, without canonical identifiers or modality classifications. Even the apparent uniqueness of the cues is partly a lexical artifact: `gin`, `gin glass` and `bottle of gin` can represent related associations without matching exactly.

## Interpretation

Yes, there is genuine sensory data within the prose and recall cues. The representation is **uneven and untyped**. Visual form is easy to detect, which fits Pretorius's aesthetic priorities. Explicit physiological state and certain nonvisual sensations are less common. Stronger future memories should connect physical sensations to decisions and recurrent associations rather than simply add adjectives.

The strongest established structure is autobiographical: concrete first-person scenes, grounded people and places, expectations and violations, consequences that echo, and backward links. The weakest technical layer is a **standardized associative index** telling an encoder which object, person, sensory perception, emotional or physiological state, and prior episode are connected.

## Next changes

Keep `memory_text` and its event ID immutable wherever possible. Add **separate editorially reviewed annotations** in a sidecar file keyed by `event_id`. The optional schema in `memories/docs/MEMORY_ASSOCIATION_SIDECAR_SCHEMA.json` defines sensory percepts, body state, dream versus waking recollection, recurring canonical cues, experiential salience, and typed event associations. Empty or unknown values should remain unknown; do not populate fields by fabrication. Automatic keyword tagging can propose **candidates for human or model-assisted review**, but must not be silently treated as ground truth.

Future authoring should emphasize repeated cross-modal experiences involving the same people, objects, rooms and laboratory failures. Especially strengthen smell, internal bodily state, taste, sound, and recurring dreams, with realistic intervals and contrasting outcomes. A cue should be capable of triggering several memories and several conflicting expectations.

## Imprinting boundary

The future encoder is not a text file or a language-model prompt. Mapping narrative events to bounded stimulus patterns, training and retrieval, learned-weight overlays, cross-event interference, and behavioral influence remain **experimental requirements**. Do not confuse corpus completeness with evidence that synaptic memory imprinting works. See `docs/PRETORIUS_CORPUS_INTEGRATION.md` for topology versus learned-weight separation.

Do not place the university material currently being authored into this corpus until submitted. The boundary before Henry's reunion in June 1899 is absolute.
