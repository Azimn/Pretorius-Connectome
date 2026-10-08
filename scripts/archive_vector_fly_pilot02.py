#!/usr/bin/env python3
"""Archive actual FlyWire class-anchor result into Git with measured summaries."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "tfidf_lexical", "unrestricted_frozen_graph",
    "cell_class_frozen_graph", "degree_pool_frozen_graph",
    "cell_class_learned_graph", "degree_pool_learned_graph",
)
METHOD_NAMES = {
    "tfidf_lexical": "Train-fitted narrative TF-IDF",
    "unrestricted_frozen_graph": "Frozen unrestricted input map",
    "cell_class_frozen_graph": "Frozen annotated neuron-class pool",
    "degree_pool_frozen_graph": "Frozen same-size degree-pool control",
    "cell_class_learned_graph": "Trained annotated neuron-class pool",
    "degree_pool_learned_graph": "Trained same-size degree-pool control",
}


def verify(r: dict) -> None:
    source = r.get("cell_annotation", {})
    if (r.get("study") != "Vector Fly neuron-class-constrained cue anchors Pilot 02"
            or r.get("source_events") != 450
            or r.get("source_l2_schema") != "pretorius.shared-features.v2"
            or r.get("annotation_git_blob") != "02e72f6c8161d3465f77fec0edf96c5d98027a9e"
            or source.get("annotation_git_blob") != r.get("annotation_git_blob")
            or source.get("annotation_field") != "cell_class"
            or source.get("annotation_match") != "Kenyon"
            or source.get("matched_neurons", 0) < 2
            or "flywire_v783_mb_csr.npz" not in r.get("topology", "")
            or r.get("neurons", 0) < 1000
            or r.get("directed_connections", 0) < 1000
            or r.get("methods") != list(METHODS)
            or [x["seed"] for x in r.get("trials", [])] != [31, 37, 43]):
        raise ValueError("Invalid, partial or not-pinned biological anchor experiment")
    for trial in r["trials"]:
        if (trial["train_events"] + trial["validation_events"]
                + trial["test_events"] != 450):
            raise ValueError("Episode split coverage changed")
        null = trial["matched_control"]
        if (null["pool_size"] != source["matched_neurons"]
                or null["null"] != "disjoint same-size nearest-log2-total-degree control"
                or null.get("changes_original_connectivity") is not False
                or not 0 <= null["exact_degree_bin_fraction"] <= 1):
            raise ValueError("Unmatched or undisclosed neuron pool")
        a, b = trial["class_learning_audit"], trial["control_learning_audit"]
        if (a["edge_parameters"] != b["edge_parameters"]
                or not a["anatomical_counts_unchanged"]
                or not b["anatomical_counts_unchanged"]
                or not a["original_edge_targets_unchanged"]
                or not b["original_edge_targets_unchanged"]):
            raise ValueError("Training changed topology or model capacity")
        if set(trial["methods"]) != set(METHODS):
            raise ValueError("Conditions changed")
        reference = None
        for name in METHODS:
            groups = trial["methods"][name]["case_results"]
            if set(groups) != {"positive", "contradiction", "absent"}:
                raise ValueError("Missing source-linked case groups")
            ids = tuple((group, item["case_id"], item["target"])
                        for group in ("positive", "contradiction", "absent")
                        for item in groups[group])
            if reference is None:
                reference = ids
            elif ids != reference:
                raise ValueError("Conditions measured different cases")
            if not ids:
                raise ValueError("No cases evaluated")


def totals(report: dict) -> dict:
    output = {}
    for method in METHODS:
        x = {"n": 0, "top": 0, "right": 0, "cn": 0, "cfalse": 0,
             "an": 0, "afalse": 0}
        for trial in report["trials"]:
            for r in trial["methods"][method]["case_results"]["positive"]:
                x["n"] += 1
                x["top"] += r["predicted"] == r["target"]
                x["right"] += r["predicted"] == r["target"] and r["accepted"]
            for r in trial["methods"][method]["case_results"]["contradiction"]:
                x["cn"] += 1
                x["cfalse"] += bool(r["accepted"])
            for r in trial["methods"][method]["case_results"]["absent"]:
                x["an"] += 1
                x["afalse"] += bool(r["accepted"])
        output[method] = x
    return output


def report_md(data: dict, checksum: str, run_id: int, filename: str) -> str:
    nums = totals(data)
    size = data["cell_annotation"]["matched_neurons"]
    lines = [
        "# Vector Fly Pilot 02: actual neuron-class-anchored memory result",
        "",
        "**Exploratory post-hoc diagnosis on already examined author-written prompts, "
        "not independent confirmation.**",
        f"**GitHub Actions real run:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/{run_id}",
        f"**Immutable original full cases:** [JSON](runs/{filename})",
        f"**Original SHA-256:** {checksum}",
        "",
        f"Data: {data['source_events']} reconstructed first-person events; original "
        f"FlyWire MB v783 topology ({data['neurons']} neurons, "
        f"{data['directed_connections']} directed aggregate connectivity pairs).",
        f"Annotation: published tagged revision {data['cell_annotation']['annotation_ref']}, "
        f"verified blob {data['annotation_git_blob']}; field cell_class contains Kenyon. "
        f"Matched input anchors: {size} neurons.",
        "All conditions use the same source, split-pinned deterministic TF-IDF L2 v2 "
        "and original connectivity. Frozen and coactivity-trained class-pool "
        "conditions are separately evaluated against same-size, disjoint "
        "nearest-log2-degree-bin pools. This is arbitrary lexical-to-cell hashing, "
        "NOT authentic sensory responses.",
        "",
        "## Pooled descriptive observations across seeds 31, 37 and 43",
        "",
        "| Input / learning condition | Correct top-1 | Correct and accepted | False contradiction acceptance | False absent memory acceptance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, name in METHOD_NAMES.items():
        x = nums[key]
        lines.append(
            f"| {name} | {x['top']}/{x['n']} | {x['right']}/{x['n']} | "
            f"{x['cfalse']}/{x['cn']} | {x['afalse']}/{x['an']} |"
        )
    class_frozen = nums["cell_class_frozen_graph"]["right"]
    control_frozen = nums["degree_pool_frozen_graph"]["right"]
    class_train = nums["cell_class_learned_graph"]["right"]
    control_train = nums["degree_pool_learned_graph"]["right"]
    lines.extend([
        "",
        "## Prespecified comparison and learning effect",
        "",
        f"Original labeled input population, frozen: **{class_frozen}** correct accepted; "
        f"matched-size degree-bin control frozen: **{control_frozen}**.",
        f"Original labeled input population, trained: **{class_train}** correct accepted; "
        f"matched-size degree-bin control trained: **{control_train}**.",
        f"Within-class training effect: {class_train - class_frozen:+d} accepted-correct; "
        f"within-control training effect: {control_train - control_frozen:+d}.",
        "",
        "## Control balance quality and caveats",
        "",
        "| Seed | Exact log2-degree-bin fraction | Mean bin shift | Max bin shift |",
        "| --- | ---: | ---: | ---: |",
    ])
    for trial in data["trials"]:
        x = trial["matched_control"]
        lines.append(
            f"| {trial['seed']} | {x['exact_degree_bin_fraction']:.1%} | "
            f"{x['mean_degree_bin_distance']} | {x['maximum_degree_bin_distance']} |"
        )
    lines.extend([
        "",
        "A degree-bin-mismatched null is explicitly **not** a fully controlled "
        "neuroanatomical null. Cell-type annotations are a later tagged "
        "revision of the 2024 data, with a fixed Git blob. The source "
        "questions were authored by the project, reused repeatedly and are "
        "not human-reviewed. No input encoding here has been validated as "
        "a semantic/odor representation. The preexisting CSR contact counts "
        "remain unchanged. Positive score movement is not proof of "
        "biological learning or Pretorius identity.",
        "",
        "Any follow-up requires independently reviewed queries and additional "
        "cell-class/degree matched controls rather than post-hoc tuning.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--run-id", type=int, required=True)
    args = p.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    verify(data)
    original_digest = sha256(raw).hexdigest()
    filename = f"vector-fly-cellclass-pilot02-run{args.run_id}.json"
    out = ROOT / "results/associative/runs" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_bytes() != raw:
        raise ValueError("Existing archived original differs, refusing overwrite")
    out.write_bytes(raw)
    name = "VECTOR_FLY_CELLCLASS_PILOT02_RESULTS.md"
    report = ROOT / "results/associative" / name
    report.write_text(report_md(data, original_digest, args.run_id, filename),
                      encoding="utf-8")
    index = ROOT / "results/associative/README.md"
    text = index.read_text(encoding="utf-8")
    marker = "## Archived Vector Fly Pilot 02 class-anchored result"
    if marker in text:
        text = text[:text.index(marker)].rstrip()
    link = (f"\n\n{marker}\n\n[Verified measured report]({name}) and "
            f"[permanent original case-level JSON](runs/{filename}) "
            f"from [real workflow {args.run_id}]"
            f"(https://github.com/Azimn/Pretorius-Connectome/actions/runs/{args.run_id}).\n")
    index.write_text(text.rstrip() + link, encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    text = handoff.read_text(encoding="utf-8")
    marker = "## Vector Fly cell-class Pilot 02 biological result archived"
    if marker in text:
        text = text[:text.index(marker)].rstrip()
    link = (f"\n\n{marker}\n\n[Successful real experiment {args.run_id}]"
            f"(https://github.com/Azimn/Pretorius-Connectome/actions/runs/{args.run_id}). "
            f"[Preserved report](../results/associative/{name}) and "
            f"[complete cases](../results/associative/runs/{filename}). "
            "The experiment is exploratory, not validated neurobiological memory.\n")
    handoff.write_text(text.rstrip() + link, encoding="utf-8")
    print(f"Archived immutable original {filename}, sha256 {original_digest}")


if __name__ == "__main__":
    main()
