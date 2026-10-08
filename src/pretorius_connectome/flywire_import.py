"""Stream FlyWire v783 connection tables without loading them into memory.

This stage validates and summarizes real connectivity. It does not construct a
full simulation or download data requiring FlyWire authentication.
"""
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


def summarize_connections(path: str | Path) -> dict:
    """Read Codex connections_princeton.csv[.gz] and count neuron pairs/synapses."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    nodes = set()
    pairs = 0
    synapses = 0
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("connection table has no header")
        columns = set(reader.fieldnames)
        pre = next((c for c in ("pre_root_id", "pre_pt_root_id") if c in columns), None)
        post = next((c for c in ("post_root_id", "post_pt_root_id") if c in columns), None)
        weight = next((c for c in ("syn_count", "synapse_count", "weight") if c in columns), None)
        if not all((pre, post, weight)):
            raise ValueError(f"unsupported connection columns: {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            try:
                source = int(row[pre])
                target = int(row[post])
                count = int(row[weight])
                if source < 0 or target < 0 or count <= 0:
                    raise ValueError("IDs must be nonnegative and synapse count positive")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid connection at CSV line {line}") from exc
            nodes.add(source)
            nodes.add(target)
            pairs += 1
            synapses += count
    return {
        "source": str(path),
        "neuron_ids_in_connections": len(nodes),
        "connection_rows": pairs,
        "synaptic_contacts": synapses,
        "note": "Neuron count excludes isolated neurons; rows may include duplicate directed pairs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FlyWire v783 connectivity CSV")
    parser.add_argument("connections", help="Path to connections_princeton.csv.gz")
    args = parser.parse_args()
    print(json.dumps(summarize_connections(args.connections), indent=2))


if __name__ == "__main__":
    main()
