"""Convert FlyWire connection CSV into a compact, deterministic CSR graph.

Keeps synapse counts unsigned. Neurotransmitter sign and neural dynamics
must be modeled separately. Requires numpy; no dataset download performed.
"""
import csv
import gzip
from pathlib import Path

import numpy as np


def load_csr(path):
    """Return (root_ids, indptr, indices, counts) sorted by source and target.

    Repeated directed pairs are summed. Index arrays use int32; synapse
    counts use int64 to avoid silent overflow. Memory use scales with
    connectivity table size; streaming/out-of-core is a later milestone.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    edges = {}
    ids = set()
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        pre = next((x for x in ("pre_root_id", "pre_pt_root_id") if x in columns), None)
        post = next((x for x in ("post_root_id", "post_pt_root_id") if x in columns), None)
        count_col = next((x for x in ("syn_count", "synapse_count", "weight") if x in columns), None)
        if not all((pre, post, count_col)):
            raise ValueError("unsupported connection table columns")
        for line, row in enumerate(reader, 2):
            try:
                src, dst, count = int(row[pre]), int(row[post]), int(row[count_col])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid row {line}") from exc
            if min(src, dst) < 0 or count <= 0:
                raise ValueError(f"invalid row {line}")
            ids.update((src, dst))
            key = (src, dst)
            edges[key] = edges.get(key, 0) + count

    roots = np.asarray(sorted(ids), dtype=np.uint64)
    if len(roots) > np.iinfo(np.int32).max:
        raise ValueError("too many nodes for int32 indexing")
    mapping = {int(root): i for i, root in enumerate(roots)}
    ordered = sorted(edges.items(), key=lambda item: item[0])
    indptr = np.zeros(len(roots) + 1, dtype=np.int64)
    indices = np.empty(len(ordered), dtype=np.int32)
    counts = np.empty(len(ordered), dtype=np.int64)
    for k, ((src, dst), count) in enumerate(ordered):
        indptr[mapping[src] + 1] += 1
        indices[k] = mapping[dst]
        counts[k] = count
    np.cumsum(indptr, out=indptr)
    return roots, indptr, indices, counts
