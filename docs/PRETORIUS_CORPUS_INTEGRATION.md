# Pretorius identity corpus integration plan

## Purpose
Use existing Pretorius material, **including the full reconstructed autobiographical past**, as active inputs for FlyWire-derived synaptic imprinting. The connectome topology is not an autobiographical corpus. Keep biological synapse counts immutable and store learned/imprinted weights separately.

## Core identity policy
Reconstructed pre-instantiation experiences are eligible for first-person autobiographical recall, association formation, preference shaping, and neural imprinting, just like post-instantiation experience. The experiment does not require the subject to treat reconstructed memories as emotionally or behaviorally inert.

Preserve provenance as **research metadata**, not as a gate that excludes reconstructed history from the subject's accessible autobiography. Do not assert that reconstructed events were independently observed in the real world. Distinguish a subject's autobiographical claims from independently observed runtime events in reports and evaluation.

## Verified donor
- `Azimn/Agent-Pretorius/SOUL.md` exists on its default branch. It supplies persona/identity instructions; it is not itself an observed interaction log.

## Additional donor candidates requiring path and version verification
- `Azimn/The-Doctor-Lives`: bootstrap identity, autobiographical history, relationships, commitments, self-model, and deep-history records. Earlier inventories mention `bootstrap.json` and `pretorius_connectome_v0.2.json`, but neither exists at the repository root on the queried default branch. Locate actual paths and versions before importing.
- `Azimn/Agent-Pretorius`: other identity, training, and historical documents.
- `Azimn/Jelly-Psiduck`: typed memories, relationship records, beliefs, claims, and concerns.
- Earlier Pretorius V6 memory and relationship branches, subject to branch verification.
- `pretorius_lora_dataset_v5_merged.txt`: locate exact source and review duplicates and conflicting records.

## Required provenance for each extracted record
- `record_id`: stable source-scoped ID
- `source_repository`, `source_path`, `source_ref`, `source_hash`
- `kind`: identity, memory, relationship, belief, commitment, habit, training_example, or other
- `origin`: canonical_fiction, reconstructed_prehistory, observed_runtime, generated_training, or uncertain
- `subject`, `content`, `event_time` (nullable)
- `confidence` and `contradicts` (nullable)

## Implementation stages
1. Inventory source paths and versions, including non-default branches, without editing donor repositories.
2. Extract and normalize records to a versioned JSONL corpus; include reconstructed history in the imprintable training set.
3. Deduplicate, identify contradictions, and create train/validation/holdout splits by event or narrative cluster.
4. Build an encoder producing bounded stimulus patterns and a separate plasticity overlay on top of immutable FlyWire CSR counts.
5. Test recall, retention, interference, and behavioral influence against no-imprint and shuffled-label controls.
6. Compare full reconstructed-history initialization against equivalent gradually acquired experience, and compare with the 4,096-unit baseline under matched budgets.

## Immediate constraint
Do not claim that more persona text alone will yield better neural cognition. Encoding, training dynamics, retrieval, and behavioral coupling must be measured.
