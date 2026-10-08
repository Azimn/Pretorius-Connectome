# Candidate sensory and associative index (v10)

The sidecars provide one annotation record for every v10 event. Every record is `unreviewed_candidate`, including fifteen with model-authored evidence excerpts. Absence of a sensory annotation is **unknown**, not a verified absence of experience. `cue_surface_forms` preserve the original recall-cue strings and `cue_ids` point into the candidate registry. Do not treat string matching as demonstrated perception.

The registry produces deterministic identities for exactly normalized strings using an FNV-1a hash of JavaScript UTF-16 code units. A small, explicitly listed alias set is an editorial hypothesis, not an identity assertion. For example, `stale rosewater` and `rosewater` are grouped provisionally; `wet coal` is not conflated with `coal dust`. No confidence metric is inferred from the number of repeated cues.

New-event percept annotations preserve short direct text excerpts, percept modality, mode of experience, body-state reports where present, and low-confidence proposed typed associations. For events about dreams or imagined smells, the mode is included to prevent confusing dream experience or recall with external perception. A reviewer should examine each excerpt in context, resolve reported versus personally perceived stimuli, and approve or reject every alias and typed relation before neural encoding.

Use `python scripts/validate_autobiographical_corpus.py` to check structural alignment against the integrated narrative. Schema validation and an editorial review process are separate gates. Indexes can be used now for exploratory retrieval tests but are not validated learned connections.
