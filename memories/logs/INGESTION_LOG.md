# Autobiographical archive ingestion log

**Date:** 2026-10-08
**Destination:** `Azimn/Pretorius-Connectome` on `main`, within `memories/`
**Operation:** Additive corpus and research-document import, no application-source changes.

The latest integrated source is **The Lost Years v7**, a pre-reunion fictive autobiography with **287 events** and a complete structured JSONL dataset. Its 27-event increment and separate narratives are included but should not be added on top of the already complete dataset.

Previous v1-v6 ZIP releases and the v7 ZIP are stored under `archive/`, so older metadata and narrative drafts are not lost. They are snapshots, not separate events to automatically append. Original input documents are retained under `sources/` and must not be confused with cleaned event records. The v7 logs include research, psychological hypotheses, continuity, production, and validation reports.

**Canonical import:** `memories/current/Pretorius_v7_287_Events_Complete.jsonl`.
**Boundary:** Stop immediately before reunion with Henry Frankenstein (June 1899); exclude the Bride and all Frankenstein Village experience.
**Scope:** No change to simulations, neural weights, datasets from FlyWire, or runtime character state.

Byte-for-byte source provenance for every file is recorded in `file_manifest.json`. Read the `sources/` material and `archive/` as background only, not extra training rows.
