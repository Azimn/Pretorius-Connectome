# v10 first associative milestone | 400 / 1,000

**Date:** 2026-10-08. **Baseline:** commit 9717471, v9 with 385 events. **Added:** fifteen new post-university memories, E26-001 to E26-015. **Remaining under the current allocation:** 600 events. **Episodes:** 25. **Boundary:** E05-021, June 1899, remains the final event. The owner-authored university arc remains untouched.

## Production and preservation

The v9 385-event ID, title, date and first-person narrative text are preserved exactly. Only chronological_order is recalculated to place fifteen E26 events in sequence. The canonical file is `memories/current/Pretorius_v10_400_Events_Complete.jsonl`; the batch file contains only the new E26 records. The v10 Markdown edition is a newly generated chronological reading edition, with v9's documentary apparatus preserved in its previous edition.

## Associative candidate preparation

`memories/annotations/v10_400_sidecars.jsonl` contains 400 rows keyed by stable event IDs. Old events receive conservative cue candidate records without invented bodily states or modality labels. The fifteen new records contain directly quoted evidence candidates for their sensory references and explicit sleep or body-state observations where narrated. All remain `unreviewed_candidate`; this is not an expert-validated neural stimulus set.

`memories/annotations/v10_cue_registry.json` indexes exact normalized cue phrases plus a small candidate alias list. Semantic aliases have *not* been accepted as ground truth. Apparent overlaps (lamp oil, wet coal, glass door, rosewater and the old instrument handle) are intended to support multiple simultaneous competing recollections, not one-to-one story lookup. A cue ID records indexable narrative material, not neuronal learning.

## Checkpoint tests

The generation transaction checked the 385 preserved narrative strings, 400 unique IDs, fifteen additions, complete required fields, valid new link targets, backward within-batch links, candidate sensory excerpts present in the exact prose, cue registry collisions, annotation row count and the untouched stopping boundary. A repeatable validator is included under `scripts/validate_autobiographical_corpus.py`. No experiment on actual neural encoding, retrieval or behavioral influence was performed.

## Next editorial step

Review legacy narratives for explicit percepts versus object mentions, recollections versus current sensation, and dreams versus waking inference. Validate any cue alias by inspecting its event context. At the 600/800/1,000 milestones, compare paired retrieval and shuffled-index controls; do not mistake increasing cue reuse for improved conduct.

**Concurrent-episode reservation:** E25 is used by the separate 1867 scientific memoir files under `memories/1867_anatomist_of_the_soul/`. V10's adult-life memories therefore use E26. The separately authored 1867 files were neither imported into the 400-record integrated dataset nor edited by this transaction.
