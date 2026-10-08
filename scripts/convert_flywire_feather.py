"""Convert verified FlyWire v783 Feather connections into deterministic CSR arrays.

The original synapse counts are retained as unsigned biological evidence.
No neurotransmitter sign or neural dynamics are inferred here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pyarrow.compute as pc


def convert(feather_path: Path, root_ids_path: Path, output_path: Path, *,
            neuropil_prefix: str | None = None, prune_to_region: bool = False) -> dict:
    """Optionally isolate synapses assigned to an anatomical neuropil family.

    Prefix-selecting MB_ captures rows whose RELEASE neuropil annotation
    starts with MB_. It is not a full functional mushroom-body circuit.
    Unfiltered conversion preserves historical output unchanged.
    """
    if prune_to_region and neuropil_prefix is None:
        raise ValueError("pruning requires a neuropil prefix")
    if neuropil_prefix is not None and not neuropil_prefix.strip():
        raise ValueError("neuropil prefix must be nonempty")
    roots = np.load(root_ids_path, allow_pickle=False).astype(np.uint64)
    if roots.ndim != 1 or len(np.unique(roots)) != len(roots):
        raise ValueError("root IDs must be a unique one-dimensional array")
    roots.sort()
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    if neuropil_prefix is not None:
        columns.append("neuropil")
    table = feather.read_table(feather_path, columns=columns)
    original_rows = int(table.num_rows)
    if neuropil_prefix is not None:
        selector = pc.starts_with(
            pc.fill_null(table["neuropil"], ""), neuropil_prefix
        )
        table = table.filter(selector)
        if table.num_rows == 0:
            raise ValueError("no connectivity rows matched the neuropil prefix")
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
    release_root_count = int(len(roots))
    if prune_to_region:
        roots = np.unique(np.concatenate((source, target)))
        src = np.searchsorted(roots, source)
        dst = np.searchsorted(roots, target)
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
        "original_release_rows": original_rows,
        "unfiltered_release_neurons": release_root_count,
        "neuropil_prefix": neuropil_prefix,
        "pruned_to_incident_neurons": bool(prune_to_region),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feather", type=Path)
    parser.add_argument("root_ids", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--neuropil-prefix",
                        help="Restrict to published region labels beginning with this prefix, e.g. MB_")
    parser.add_argument("--prune-to-region", action="store_true",
                        help="Keep only neuron IDs incident to selected synapses")
    args = parser.parse_args()
    print(json.dumps(convert(
        args.feather, args.root_ids, args.output,
        neuropil_prefix=args.neuropil_prefix,
        prune_to_region=args.prune_to_region,
    ), indent=2))


if __name__ == "__main__":
    main()
