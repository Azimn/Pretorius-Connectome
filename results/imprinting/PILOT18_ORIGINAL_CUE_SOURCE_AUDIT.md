# Pilot18 post-result source audit: these are event-associated cue anchors, not validated paraphrases

**Post-hoc descriptive provenance audit, NOT a predeclared hypothesis test or a metric-training input.** Source: the existing original [450 fictional Pretorius event narratives](../../memories/current/Pretorius_v12_450_Events_Complete.jsonl) and their [450 original cue annotation sidecars](../../memories/annotations/v12_450_sidecars.jsonl), analyzed by [a new reproducible script](../../scripts/audit_original_source_cue_candidates18.py). No source text, cue surfaces, frozen BC01 encodings, episode IDs, first/middle training partition, last heldout partition or learned model parameters were changed.

## Structural observations from ORIGINAL full 450-record cue annotation file

| Direct original source attribute | Observed count |
|---|---:|
| Fictional original source memory event records | **450** |
| Original cue annotation records | **450** |
| Annotation status `unreviewed_candidate` | **450/450** |
| All candidate cue surface phrases | **1,463** |
| Cue surface phrases of three words or fewer | **1,453/1,463** |
| Median number of words per cue surface | **2** |
| Events with three distinct selected first/middle/last cues | **448/450** |
| Of these 448, **no shared literal token** between first and last cue phrase | **446/448** |
| Original last cue phrase occurs VERBATIM inside its associated narrative `memory_text` | **107/448** |

Examples from the exact original cue sidecar (not assistant-generated test paraphrases): E09-001 is tagged `turned glove`, `millstream map`, `green tape`. E09-002 has `thank-you letter`, `fresh sheet`, `stove`. E01-001 has `camphor`, `beetle`, `second brass key`, `cracked lens`. E01-002 has `candle wax`, `singed cuff`, `telegraph diagram`.

These are **short episodic object/action/detail anchor cues**. They may all refer to the same original source experience without being synonyms, paraphrases or even semantically close phrases. Token nonoverlap is not proof of semantic difference, but the cue-design intent cannot responsibly be described as verified paraphrase equivalence. The annotation metadata explicitly says `unreviewed_candidate` in all 450 records. Their author/editor review status is not established. Previous documentation referring to them simply as human-authored semantic cues should be corrected where material.

## Effect on Pilot18 and earlier pilot interpretations

[Successful six-condition Pilot18 full original-source run 38058153904](https://github.com/Azimn/Pretorius-Connectome/actions/runs/38058153904) withheld the original third cue from training/storage, so **316 distinct source third cues** formed a legitimate *within-corpus untrained cue* test. The trained global 64D first+middle contrastive discriminant recovered **0/316**, same as random semantic projection **0/316**, mispaired **1/316**, PCA **1/316**, lexical **2/316**, and deliberately WRONG source CONTENT **3/316** by chance. All had **0 correctly retrieved and accepted** source third views after a 62-validation-absent-only gate. Yet the same models retained 144–145/159 trained FIRST literal cue source IDs. This is a valid negative result concerning **unobserved cue-anchor relationship recovery**, but it does NOT falsify semantic paraphrase understanding, because the source third cues have never been independently validated as paraphrases of the first two.

Likewise Pilot17's 149/159 semantic key recall after storing first/middle/last exact original source cues is reliable in-sample numeric event allocation, not natural-language episodic comprehension; the original 1/31 exploratory fourth cue accuracy in Pilot11–17 is not a controlled human paraphrase metric. Strongly worded older claims about semantic generalization failure should be narrowed to **cross-anchor cue-to-event binding failure on these particular candidate annotations**.

## Revised forward scientific direction

Two **separate** test batteries and mechanisms are needed:

1. **Episodic cue-graph / provenance-bound binder.** Train an event-specific *joint association* among distinct details, with the original narrative and annotation relation explicitly available during event formation. A candidate cue such as `green tape` should activate the E09-001 experience only if it was originally observed, described, or authored in that experience—NOT because it is a semantic synonym for `turned glove`. Compare numeric event-conditioned anchor co-occurrence, literal retrieval on original narrative, BM25 text search, dense narrative semantic search, and source-content wrong-person contamination controls. If the entire memory narrative/source sidecar is indexed, disclose that the previously withheld cue may now be PRESENT in the stored source, i.e. no longer an unseen-key experiment. A truly unseen cue needs a **separate independently reviewed test item** not available in stored narrative/annotations.
2. **Actual independently authored human semantic paraphrase equivalence battery.** For a curated frozen set of source statements, have genuinely separate annotators write meaning-preserving paraphrases and genuinely absent/false-memory distractors, blind to retrieval-arm outputs, with episode ownership/source provenance. Train only on ORIGINAL source; neither the heldout paraphrases nor distractors may enter metric training, gate calibration, numeric source keys or narrative expansion before test. Require **correct-and-accepted** identification paired with false-memory rejection at a predeclared operating point. With no such independent annotations yet, do not claim a human-paraphrase success/failure.

These lines answer different questions. **Do not retrospectively relabel the original 316 third views or original 31 fourth cues as a new independent sealed paraphrase battery**, nor tune Pilot18 penalty, projection rank or confidence thresholds after observing their results. The present cue file is not a quality-assured human semantic dataset.

See [original Pilot18 full-source provisional result](PILOT18_SOURCE_PROVISIONAL.md), [exact predeclared study](../../docs/PILOT18_MIRROR_KEYS_PROTOCOL.md) and [original source audit implementation](../../scripts/audit_original_source_cue_candidates18.py).
