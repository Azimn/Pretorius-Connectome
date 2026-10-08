"""Convert verified FlyWire v783 Feather connections into deterministic CSR arrays.

The original synapse counts are retained as unsigned biological evidence.
No neurotransmitter sign or neural dynamics are inferred here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


def convert(feather_path: Path, root_ids_path: Path, output_path: Path) -> dict:
    roots = np.load(root_ids_path, allow_pickle=False).astype(np.uint64)
    if roots.ndim != 1 or len(np.unique(roots)) != len(roots):
        raise ValueError("root IDs must be a unique one-dimensional array")
    roots.sort()
    table = feather.read_table(feather_path, columns=[
        "pre_pt_root_id", "post_pt_root_id", "syn_count"
    ])
    source = table["pre_pt_root_id"].to_numpy().astype(np.uint64)
    target = table["post_pt_root_id"].to_numpy().astype(np.uint64)
    counts = table["syn_count"].to_numpy().astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("synapse counts must be positive")
    src = np.searchsorted(roots, source)
    dst = np.searchsorted(roots, target)
    if np.any(src >= len(roots)) or np.any(dst >= len(roots)):
        raise ValueError("connection references unknown root ID")
    if np.any(roots[src] != source) or np.any(roots[dst] != target):
        raise ValueError("connection references unknown root ID")
    order = np.lexsort((dst, src))
    src, dst, counts = src[order], dst[order], counts[order]
    new_pair = np.empty(len(src), dtype=bool)
    if len(src):
        new_pair[0] = True
        new_pair[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
        starts = np.flatnonzero(new_pair)
        totals = np.add.reduceat(counts, starts)
        src, dst = src[starts], dst[starts]
    else:
        totals = counts
    if np.any(totals < 0):
        raise ValueError("synapse count overflow")
    indptr = np.zeros(len(roots) + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, root_ids=roots, indptr=indptr,
             indices=dst.astype(np.int32), synapse_counts=totals)
    return {
        "neurons": int(len(roots)),
        "directed_pairs": int(len(dst)),
        "synaptic_contacts": int(totals.sum()),
        "source_rows": int(table.num_rows),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feather", type=Path)
    parser.add_argument("root_ids", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.feather, args.root_ids, args.output), indent=2))


if __name__ == "__main__":
    main()
