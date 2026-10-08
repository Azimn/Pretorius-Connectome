# Pretorius identity corpus integration plan

## Purpose
Use existing Pretorius material as inputs for the FlyWire-derived substrate's imprinting experiments. The connectome topology is not an autobiographical corpus. Keep biological synapse counts immutable and store learned/imprinted weights separately.

## Verified donor
- `Azimn/Agent-Pretorius/SOUL.md` exists on its default branch. Treat as a persona/identity instruction document, not as evidence of lived events.

## Additional donor candidates requiring path and version verification
- `Azimn/The-Doctor-Lives`: bootstrap identity, autobiographical history, relationships, commitments, self-model, and deep-history records. Earlier project inventories mention `bootstrap.json` and `pretorius_connectome_v0.2.json`, but **neither exists at the repository root on the currently queried default branch**. Locate the actual paths and version before importing.
- `Azimn/Agent-Pretorius`: other identity, training and historical documents.
- `Azimn/Jelly-Psiduck`: typed memories, relationship records, beliefs, claims, concerns, and reconstructed-versus-lived provenance.
- `Azimn/Pretorius`: V6 memory and relationship branches, subject to branch verification.
- Existing `pretorius_lora_dataset_v5_merged.txt`: historical training corpus; locate exact source and review for duplicates and conflicting records.

## Required provenance for each extracted record
- `record_id`: stable source-scoped ID
- `source_repository`, `source_path`, `source_ref`, `source_hash`
- `kind`: identity, memory, relationship, belief, commitment, habit, training_example, or other
- `origin`: canonical_fiction, reconstructed_prehistory, observed_runtime, generated_training, or uncertain
- `subject`, `content`, `event_time` (nullable)
- `confidence` and `contradicts` (nullable)

Never silently promote authored/reconstructed events to observed runtime memories.

## Implementation stages
1. Inventory source paths and versions, including non-default branches, without editing donor repositories.
2. Extract and normalize records to a versioned JSONL corpus; retain originals and exact provenance.
3. Deduplicate, identify contradictions, and create train/validation/holdout splits by event or narrative cluster, not random paraphrases.
4. Build an encoder producing bounded stimulus patterns and a separate plasticity overlay on top of immutable FlyWire CSR counts.
5. Test recall, retention, interference, and behavioral influence against no-imprint and shuffled-label controls.
6. Compare results against the 4,096-unit baseline under matched experience budgets.

## Immediate constraint
Do not claim that more persona text alone will yield better neural cognition. Encoding, training dynamics, retrieval, and behavioral coupling must be measured.
