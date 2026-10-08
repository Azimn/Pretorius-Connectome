"""Version-pinned FlyWire neuron annotations for restricted lexical cue routing.

This chooses INPUT ANCHORS, not synaptic memories or biological features.
Source anatomical CSR and numerical transition weights remain untouched.
"""
from __future__ import annotations

import csv
from hashlib import blake2b, sha1
from io import StringIO
from pathlib import Path
import re

import numpy as np

# Tagged official FlyWire v783 neuron annotation release. Never fetch 'main'.
ANNOTATION_REF = "a83b2776d60d5764cef36b927f5f9679c16c47a2"  # v3.2.0
ANNOTATION_BLOB = "02e72f6c8161d3465f77fec0edf96c5d98027a9e"
ANNOTATION_PATH = "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    + ANNOTATION_REF + "/" + ANNOTATION_PATH
)


def git_blob(data: bytes) -> str:
    return sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def select_cell_class_pool(
    annotation_file: Path | str, topology, *,
    field: str = "cell_class", pattern: str = "Kenyon",
    expected_blob: str = ANNOTATION_BLOB,
) -> tuple[np.ndarray, dict]:
    """Join immutable neuron-class labels to MB graph root IDs exactly.

    The caller supplies a label pattern to make biological targeting explicit.
    Matching alone is *not* verification of a functional neuron population.
    """
    if field not in {"cell_class", "cell_type", "cell_sub_class"}:
        raise ValueError("Unsupported neuron-class field")
    if not pattern or len(pattern) > 128:
        raise ValueError("Nonempty bounded label pattern required")
    raw = Path(annotation_file).read_bytes()
    observed_blob = git_blob(raw)
    if observed_blob != expected_blob:
        raise ValueError("Neuron annotations are not the required pinned revision")
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeError as e:
        raise ValueError("Neuron annotations are not UTF-8") from e
    reader = csv.DictReader(StringIO(decoded), delimiter="\t")
    if reader.fieldnames is None or "root_id" not in reader.fieldnames or field not in reader.fieldnames:
        raise ValueError("Neuron annotation schema lacks required columns")
    ids = {int(n): i for i, n in enumerate(topology.root_ids)}
    selected = set()
    matched_labels = {}
    observed = set()
    joined_count = 0
    total_rows = 0
    regex = re.compile(re.escape(pattern), flags=re.IGNORECASE)
    for row in reader:
        total_rows += 1
        try:
            root_id = int(row["root_id"])
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Bad neuron root_id at TSV record {total_rows}") from e
        position = ids.get(root_id)
        if position is None:
            continue
        if position in observed:
            raise ValueError("Duplicate annotated FlyWire root_id inside topology")
        observed.add(position)
        joined_count += 1
        label = row[field] or ""
        if regex.search(label):
            selected.add(position)
            matched_labels[label] = matched_labels.get(label, 0) + 1
    if not selected:
        raise ValueError("No specified neuron-class labels intersect biological topology")
    pool = np.array(sorted(selected), dtype=np.int64)
    return pool, {
        "annotation_ref": ANNOTATION_REF if expected_blob == ANNOTATION_BLOB else "test-fixture",
        "annotation_git_blob": observed_blob,
        "annotation_field": field,
        "annotation_match": pattern,
        "annotations_read": total_rows,
        "annotations_joined": joined_count,
        "matched_neurons": int(len(pool)),
        "graph_neurons": int(len(topology.root_ids)),
        "matched_label_counts": dict(sorted(matched_labels.items())),
        "interpretation": "neuron-class-constrained lexical feature anchors; not learned biological sensory encoding",
    }


def degree_stratified_control(topology, actual: np.ndarray, *,
                              seed: int = 31) -> tuple[np.ndarray, dict]:
    """Sample a disjoint same-sized pool with nearest integer log2-degree bins.

    Exact edge degrees may be unmatched where candidates are exhausted.
    Quantify mismatch and do not describe the null as perfectly degree-matched.
    """
    n = len(topology.root_ids)
    selected = np.asarray(actual)
    if (selected.ndim != 1 or selected.size < 1
            or not np.issubdtype(selected.dtype, np.integer)
            or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= n)):
        raise ValueError("Invalid selected neuron indices")
    if 2 * len(selected) > n:
        raise ValueError("Not enough disjoint neurons for same-sized control")
    counts = np.diff(topology.indptr).astype(np.int64)
    counts += np.bincount(topology.indices, minlength=n).astype(np.int64)
    bins = np.floor(np.log2(counts + 1)).astype(np.int64)
    reserved = np.zeros(n, dtype=bool)
    reserved[selected] = True
    others = np.flatnonzero(~reserved)
    rng = np.random.default_rng(seed)
    stacks = {
        int(b): list(map(int, rng.permutation(others[bins[others] == b])))
        for b in np.unique(bins[others])
    }
    control = []
    distances = []
    for b in bins[selected]:
        choice = min(
            (k for k, values in stacks.items() if values),
            key=lambda k: (abs(k - int(b)), k),
        )
        control.append(stacks[choice].pop())
        distances.append(abs(choice - int(b)))
    result = np.array(sorted(control), dtype=np.int64)
    if len(result) != len(selected) or np.intersect1d(result, selected).size:
        raise AssertionError("Control pool overlaps or differs in cardinality")
    return result, {
        "null": "disjoint same-size nearest-log2-total-degree control",
        "seed": int(seed),
        "pool_size": int(len(selected)),
        "exact_degree_bin_fraction": round(
            sum(distance == 0 for distance in distances) / len(distances), 6
        ),
        "maximum_degree_bin_distance": int(max(distances)),
        "mean_degree_bin_distance": round(float(np.mean(distances)), 6),
        "preserves_exact_neuron_degree": False,
        "preserves_biological_cell_class": False,
        "changes_original_connectivity": False,
    }
