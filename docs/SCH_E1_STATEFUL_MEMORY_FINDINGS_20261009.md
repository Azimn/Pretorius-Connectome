# SCH E1: State-aware cue-equivalence experiment using the canonical Pretorius L1 archive

**Research date:** October 9, 2026, America/Chicago  
**Canonical source owner:** Pretorius-Connectome, source snapshot \`6d2768211f5c2184c8bbdb833c06e169b5137197\`  
**Experimental owner:** [Attractomancy](https://github.com/Azimn/Attractomancy), inspected at commit [\`19b6b62\`](https://github.com/Azimn/Attractomancy/commit/19b6b62cdbc5a5a371fe2e7c8c3adef2246ad8c2)  
**Experimental report:** [E1 completed results](https://github.com/Azimn/Attractomancy/blob/19b6b62cdbc5a5a371fe2e7c8c3adef2246ad8c2/experiments/SCH_E1_PRETORIUS_STATEFUL/RESULTS.md)  
**Status:** Cross-repository experimental reference only. No changes to L0, L1, L2, FlyWire, BioCircuit, or the production Pretorius renderer.

## Scope

SCH E1 consumes the existing checksum-validated L1 export through this repository's **own** \`pretorius_connectome.shared_memory.read_l1\`. It does not regenerate or edit the 450 reconstructed fictional memories, their 27 episode IDs, editorial recall cues, source hashes, L2 fitted caches, or neural state. A separate Attractomancy-owned SQLite adapter indexes 54 representative event records using both a source-authored editorial recall cue and a deterministic arbitrary opaque key. Both keys resolve to precisely the same guarded L1 content.

A strictly **synthetic sandbox stream** simulates permission for a fictional counterpart, then a grant and a revocation. This overlay is explicitly not canon Pretorius biography, not lived autobiographical data, and not part of the source archive. Its purpose is to test whether durable external state can be reloaded across separate Python processes and applied to a character action.

## Measured findings

- The indexed editorial cue and opaque key found the correct canonical event in **54/54 matched cases** at each of three SQLite process-restart phases, with all content hashes equal. The unindexed FTS5 narrative-only search identified the editorial-cue source at rank one **16/54** times and in the top ten **22/54** times. These are deterministic observations over the same source fixture, not independent model replications or semantic truth tests.
- The two local Qwen renderers received identical source content and separate synthetic relationship state under cue and opaque-key conditions. Both produced **identical answers in all nine paired cue-versus-key decisions**. Qwen2.5-0.5B applied the toy disclosure rules correctly in **5/9** situations; Qwen2.5-1.5B applied them correctly in **2/9**. Both exhibited problematic decisions despite correct retrieval.
- A subject/source/checksum gate rejected all nine predefined invalid record-owner/content checks per model. This validates local consistency checks, not remote authorship or source authenticity if an attacker controls both a document and its hash.
- A separate deterministic post-inference **action eligibility gate** reprocessed the original outputs using the trusted synthetic relationship status. It overrode **8/18** 0.5B responses and **14/18** 1.5B responses. Its resulting 18/18 compliance is programmed software behavior, **not** an improvement in model cognition or a general proof that relationship judgments are safe.

## Engineering recommendations, not an authorization to modify production

The source of truth should remain in this repository and be consumed via its verified public L1 interface. Keep reconstructed historical records, externally supplied synthetic state, and actually observed post-instantiation events in distinct namespace/provenance classes. Place any subject-identity, source-manifest, and record-version verification **upstream** of an LLM renderer. Treat disclosure or commitment eligibility as a separate typed policy check where the system requires strict guarantees. Do not infer that symbolic names alone create memory, preserve identity, or enforce decision rules.

The current implementation is **an experimental adapter**, not a running Pretorius cognitive architecture and not a source of validated semantic identity scores. No merge to an acting character or production policy engine is proposed by this note. Before integrating, perform an independent provenance/security review, a controlled end-to-end task battery, and audits of state versions, unintended personal-data mixing, and over-abstention.

## Provenance

Frozen source archive: \`artifacts/shared_memory/v1\` at the source snapshot noted above. Exact experiment code, synthetic fixtures, nine action pairs per model, all 54 selected memory IDs and alias pairs, original text responses, measured tokenizer costs, two model revisions, and derived post-gate replays are in [the Attractomancy E1 experiment folder](https://github.com/Azimn/Attractomancy/tree/19b6b62cdbc5a5a371fe2e7c8c3adef2246ad8c2/experiments/SCH_E1_PRETORIUS_STATEFUL).

This is research documentation and should not be counted as an independent replication in the Character Continuity Program evidence register.
