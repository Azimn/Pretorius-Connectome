# Memory-to-Cognition Port v0.1

**Status:** implementation candidate on a dedicated feature branch. Not a production Pretorius migration and not a new memory database. The main deliverables are `src/pretorius_connectome/cognition_port.py`, `src/pretorius_connectome/vector_cognition_adapter.py`, tests, and `scripts/demo_cognition_port.py`. Source owner: Pretorius-Connectome. Cross-project rationale: [Artificial Life Research Journal](https://github.com/Azimn/Artificial-Life-Research-Journal/blob/main/programs/CROSS_PROJECT_SHARED_MEMORY_INTEGRATION_2026-10-09.md).

## Intended architectural seam

An agent's canonical events remain in the **agent-owned** immutable/event-sourced ledger. Deterministic or model-specific indexes are **disposable derived caches**. Retrieval produces candidate records. This port checks the trusted bound subject, source, source pin and provenance, then yields a typed, read-only `CognitionEvidence` object. Cognitive admission, belief revision, active goals, renderer selection, action eligibility and writing genuinely lived events remain with the **consumer's cognition and event writer**, not this port.

```
owner's canonical event ledger
       |
       v (verified source adapter, source-owned preprocessing)
read-only retriever (e.g. VectorFlyStore; other subjects use their own index)
       |
       v
MemoryCognitionPort (namespace + subject + capability + SHA/version checks)
       |
       v
CognitionEvidence (candidate_not_entailment; eligible_for_lived_write=false)
       |
       v
consumer's separate provenance, perception/self-binding and action gates
```

This port exposes a Python protocol that **can** be adapted to other storage systems; it does not host a multi-tenant API, authenticate end users or replace Vector Fly's existing local HTTP server. The standard types are `SourceDescriptor`, `MemoryCandidate`, `CognitionEvidence`, `ReadOnlyMemorySource`, and `MemoryCognitionPort`.

`CognitionEvidence` includes subject ID, archive namespace, source version/hash, index encoder and scope, stable record ID, provenance class, excerpt/full source text, optional score, channel, epistemic status and an explicit false value for lived-write eligibility. It does **not** confer perceptual truth, narrative certainty, actuation authority, autobiographical status or permission to modify a subject's state.

## Security and isolation boundary

`MemoryCognitionPort.register(source)` is for a **trusted host** that has already independently verified the source. `grant_self(subject_id, namespace)` may only be called by that host, outside prompts, tool arguments and untrusted agent code. It returns an opaque, random **ephemeral in-process capability**. Do not expose capabilities to a model or browser, log them, or serialize them. This is internal scope control, **not user authentication**; a malicious consumer able to invoke `grant_self` directly in the same Python process is outside its security model. The host must authenticate any remote caller *before* granting access. There is no cross-subject grant in v0.1.

On search and get, the port validates subject, namespace and source descriptor **before** calling the backend, refuses revoked or stale tokens, and validates every returned candidate's owner, namespace, pinned hash, recognized provenance, bounded score and record ID. It does not silently skip wrong-subject results. Source adapters **must** independently validate their canonical data and per-record visibility. Do not combine multiple privacy classes in one adapter unless visibility is filtered *before ranking* by the backend. Do not use this MVP to expose Evennia private rooms, Kiki personal history, Calibos's `mind.db`, or other people's messages across trust boundaries.

The existing `VectorFlyCognitionSource` is restricted to an actual `VectorFlyStore` that validates a read-only SQLite index and its source-fitted cache. It accepts only pinned Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`, scopes `all` or `train`, and `lexical-tfidf-not-semantic`. It checks each search excerpt against the hydrated original source, maps source `provenance=reconstructed` explicitly to `origin_class=reconstructed_prehistory`, and never writes to the database. The `all` scope is for browsing, **not** a held-out test. No new lexical, semantic or neural encoder is invented by this bridge.

## Reproduce

The synthetic core needs only Python 3.11+:

```sh
python -m unittest discover -s tests -p test_cognition_port.py -v
```

The full canonical 450-event integration also uses the already-existing local CPU libraries (`numpy`, `scipy`, `scikit-learn`, `rank-bm25`):

```sh
python -m unittest discover -s tests -p test_vector_cognition_adapter.py -v
```

Alternatively build the **existing** verified Vector Fly cache and index once (or point at the previously built ones), then run:

```sh
python scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/mci-l2
python scripts/vector_fly_db.py build --cache-dir data/derived/mci-l2 --database data/derived/mci-450.sqlite --scope all
python scripts/demo_cognition_port.py --cache-dir data/derived/mci-l2 --database data/derived/mci-450.sqlite --query "camphor beetle brass key" --top-k 3
```

The demo prints the actual read-only evidence packet as JSON and revokes its token. It does not query an LLM, move synaptic weights, update Pretorius's `brain.sqlite3`, or inject any candidate into the live subject.

## Acceptance controls

The synthetic suite uses **two distinct fictional subjects**, separate ledgers, independent source hashes, and non-interchangeable capabilities. It tests legitimate recall, a cross-owner request that must fail *before* either backend runs, different-owner candidate injection, altered source hash, revoked tokens, source revision drift, duplicate IDs, invalid scores, invalid inputs and wrong-ID hydration.

The full-source suite checks exact parity with existing `VectorFlyStore.search`, actual source record hydration, the reconstructed-history evidence status, rejected cross-subject access, rejection of an invented excerpt and that the episode-held-out train index cannot retrieve a test-only event. The CI workflow runs both on a fresh CPU runner. **Only a green completed CI execution can establish full-source compatibility**; creating this document or a pending PR does not.

## Future adapters, in sequence

**The Doctor Lives:** reuse its existing read-only Vector Fly evidence-slot interface and admit the new typed envelope only behind its explicit evidence-authority migration/audit gate. Its completed Stage 02 A/B showed changed replies, but also hallucination and chronology errors; the bridge does not fix semantic truth by itself.

**Kiki Mind:** implement a disposable source adapter over Kiki's own canonical ledger/projection, preserving canonical replay, freshness, and `forbid_canonical_experience` rules. Only synthetic events should be used in first adapter tests. Never load Pretorius records into Kiki's own-memory namespace.

**Calibos Mind:** keep live thoughts, sidecars and `mind.db` local and private; use code/protocol only. No upload to this public repository.

**Frankenstein Village:** a later resident or player-mask adapter must let the Evennia authoritative world and per-mask visibility layer filter records *before retrieval*. Maintain Resident Life v2 on-demand cognition and authored-locked NPC protections. A capability issued by an agent itself is not an adequate game-world permission check.

**Eidolon/Noetic and Attractomancy:** use evidence packets as controlled inputs to separate provenance, self-binding and action-eligibility gates; preregister wrong-subject, neutral-key, revocation, false-premise and no-memory controls. No claim of consciousness or semantic identity transfer follows from retrieval.

## Limitations and versioning

v0.1 deliberately has no per-record ACL language, remote credentials, multiple subjects sharing one physical index, cross-subject research grants, belief engine, history writer, automatic cognitive gate, or neural memory transfer. It does not provide automatic GitHub-to-runtime synchronization. Real multi-tenant deployment would require independent host identity, audited ACLs, permission revocation/replay tests, isolation before ranking, network protections, and separate privacy review.

Use a new explicit schema/encoder/source pin for any incompatible change. A successful interoperability test measures source handling, not better cognition. Compare future neural/behavioral outcomes separately with matched no-retrieval, lexical, real/re-wired graph, source-mismatch, and identity-eligibility controls.
