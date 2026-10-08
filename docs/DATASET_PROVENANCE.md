# Connectome dataset provenance (Stage 1)

Target: FlyWire adult Drosophila brain, public Codex release v783.

The repository does **not** contain the biological connectome. The current
`flywire_import.py` only summarizes a user-provided connectivity CSV or
CSV.GZ. No full-brain simulation, synaptic imprinting, or downloaded dataset
has yet been demonstrated.

## Acquisition and verification

Obtain the v783 connectivity table from the official FlyWire Codex download
interface: https://codex.flywire.ai/ . Access and redistribution are governed
by the dataset provider's terms, which must be reviewed independently of
this repository's code license.

Keep original downloaded filenames and preserve the provider's release
metadata. Record the SHA-256 digest and file byte count before conversion:

```bash
python -c "import hashlib,sys,pathlib; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(); f=p.open('rb'); [h.update(b) for b in iter(lambda:f.read(1024*1024),b'')]; f.close(); print(p.name,p.stat().st_size,h.hexdigest())" path/to/connections_princeton.csv.gz
```

Inspect locally:

```bash
PYTHONPATH=src python -m pretorius_connectome.flywire_import path/to/connections_princeton.csv.gz
```

The importer accepts pre_root_id/post_root_id/syn_count (and documented
aliases). Confirm the *actual* downloaded file header before assuming
compatibility. This first pass counts connection rows and contacts, not
distinct edges after deduplication, and cannot count disconnected neurons.

## Research reuse

Before implementing a second simulator, inspect FlyWire Codex and existing
full-brain simulations for sparse graph representation, neuron dynamics,
synapse-sign inference, licensing, and performance. Keep original biological
edge counts separate from synthetic signed simulation weights.

## Acceptance gate for the next chunk

1. CI regression suite reports a successful run.
2. A real v783 table is acquired with provenance and checksum recorded.
3. CSV schema and row counts are verified against its release documentation.
4. Sparse import benchmark fits the available memory budget.

Do not claim these gates have passed without execution evidence.
