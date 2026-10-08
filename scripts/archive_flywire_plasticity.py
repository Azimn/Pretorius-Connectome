#!/usr/bin/env python3
"""Archive a completed, real FlyWire MB plasticity result in permanent Git files.

Only the verified main-branch workflow_run uses this utility. Original JSON
bytes are retained intact; the report is generated from measurements rather
than model-written performance claims or expiring Actions artifacts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "narrative_tfidf",
    "real_frozen_graph",
    "real_frozen_hybrid",
    "rewired_frozen_graph",
    "rewired_frozen_hybrid",
    "real_plastic_graph",
    "real_plastic_hybrid",
    "rewired_plastic_graph",
    "rewired_plastic_hybrid",
)
METRICS = (
    "positive_top1",
    "positive_correct_and_accepted",
    "contradiction_false_acceptance",
    "absent_false_acceptance",
)
DISPLAY = {
    "narrative_tfidf": "Narrative TF-IDF",
    "real_frozen_graph": "Real frozen MB graph",
    "rewired_frozen_graph": "Rewired frozen graph",
    "real_plastic_graph": "Real learned MB graph",
    "rewired_plastic_graph": "Rewired learned graph",
    "real_plastic_hybrid": "Real learned hybrid",
    "rewired_plastic_hybrid": "Rewired learned hybrid",
}


def verify(report):
    if (report.get("study") != "FlyWire sparse trace plasticity Pilot 01 (exploratory)"
            or report.get("source_events") != 450
            or report.get("shared_l2_schema") != "pretorius.shared-features.v2"
            or report.get("shared_l2_encoder") != "sklearn-tfidf-word12-rankstable-v2"
            or "flywire_v783_mb_csr.npz" not in report.get("topology", "")
            or report.get("seeds") != [31, 37, 43]
            or len(report.get("trials", [])) != 3
            or report.get("neuron_count", 0) < 1000
            or report.get("raw_connection_entries", 0) < 1000
            or "NOT blind" not in report.get("challenge_status", "")):
        raise ValueError("Missing original biological result/provenance")
    for trial, seed in zip(report["trials"], (31, 37, 43)):
        if (trial["seed"] != seed or set(trial["methods"]) != set(REQUIRED)
                or trial.get("l2_encoder") != "sklearn-tfidf-word12-rankstable-v2"
                or not trial.get("l2_shard_sha256")):
            raise ValueError("Wrong seed, missing treatment or mismatched methods")
        episodes = [set(trial[x]) for x in (
            "training_episode_ids", "validation_episode_ids", "test_episode_ids")]
        if (not all(episodes)
                or any(episodes[i] & episodes[j] for i in range(3)
                       for j in range(i + 1, 3))
                or len(set().union(*episodes)) != 27):
            raise ValueError("Episode separation corrupted")
        if sum(trial[x] for x in ("train_events", "validation_events", "test_events")) != 450:
            raise ValueError("Event split count corrupted")
        for side in ("real_learning", "rewired_learning"):
            a = trial[side]
            if (not a["anatomical_counts_unchanged"]
                    or not a["original_edge_targets_unchanged"]
                    or a["train_narrative_count"] != trial["train_events"]
                    or a["gain"] != 2.0 or a["activity_cap"] != 256):
                raise ValueError("Learning provenance does not match protocol")
        expected_ids = None
        for method in REQUIRED:
            item = trial["methods"][method]
            groups = item["case_results"]
            if set(groups) != {"positive", "contradiction", "absent"}:
                raise ValueError("Missing case-level group")
            if any(len(groups[g]) != item[g + "_n"] for g in groups):
                raise ValueError("Case count mismatch")
            case_ids = tuple((g, x["case_id"], x["target"])
                             for g in ("positive", "contradiction", "absent")
                             for x in groups[g])
            if expected_ids is None:
                expected_ids = case_ids
            elif case_ids != expected_ids:
                raise ValueError("Case IDs differ across conditions")
            if len(case_ids) == 0:
                raise ValueError("No per-case evidence")


def ratio(count, n):
    return f"{count}/{n} ({100 * count / n:.1f}%)" if n else "not available"


def make_report(report, raw_name, checksum, run_id):
    pooled = {}
    for method in REQUIRED:
        totals = {"p": 0, "top": 0, "right": 0, "c": 0, "cfalse": 0,
                  "a": 0, "afalse": 0}
        for trial in report["trials"]:
            groups = trial["methods"][method]["case_results"]
            for row in groups["positive"]:
                totals["p"] += 1
                totals["top"] += row["predicted"] == row["target"]
                totals["right"] += (row["predicted"] == row["target"]
                                    and row["accepted"])
            for row in groups["contradiction"]:
                totals["c"] += 1
                totals["cfalse"] += bool(row["accepted"])
            for row in groups["absent"]:
                totals["a"] += 1
                totals["afalse"] += bool(row["accepted"])
        pooled[method] = totals
    lines = [
        "# FlyWire MB activity-trace plasticity Pilot 01: actual three-seed result",
        "",
        "**Exploratory post-hoc diagnostic**, not an independent confirmatory evaluation.",
        f"**GitHub Actions run:** https://github.com/Azimn/Pretorius-Connectome/actions/runs/{run_id}",
        f"**Permanent complete original case-level JSON:** [original bytes](runs/{raw_name})",
        f"**SHA-256 of original JSON:** {checksum}",
        "",
        f"Dataset: {report['source_events']} frozen reconstructed Pretorius memories across 27 episodes.",
        f"Anatomy: verified v783 MB-selected topology, {report['neuron_count']} nodes, {report['raw_connection_entries']} aggregate adjacency entries.",
        "Training: cross-process deterministic source-pinned shared TF-IDF L2 v2, gains 2.0, 256 cap, seeds 31, 37, 43. Historical uncached v1 is a separate representation, not an equivalent encoder.",
        "Biological synapse counts and original CSR edge targets: unchanged in both trained conditions.",
        "",
        "## Pooled descriptive observations (cases reused between seeds)",
        "",
        "| Condition | True top-1 | True correct and accepted | Contradictions falsely accepted | Absent episodes falsely accepted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in DISPLAY:
        x = pooled[method]
        lines.append(
            f"| {DISPLAY[method]} | {ratio(x['top'], x['p'])} | "
            f"{ratio(x['right'], x['p'])} | {ratio(x['cfalse'], x['c'])} | "
            f"{ratio(x['afalse'], x['a'])} |"
        )
    lines.extend([
        "",
        "## Prespecified plastic-vs-rewired comparison",
        "",
    ])
    real = pooled["real_plastic_graph"]
    null = pooled["rewired_plastic_graph"]
    delta = real["right"] - null["right"]
    lines.append(
        f"Real MB learned graph had {real['right']} correct-and-accepted positive cases "
        f"versus {null['right']} for the rewired learned control (difference {delta:+d} "
        "across pooled seed observations)."
    )
    lines.append(
        f"Contradiction false acceptances: real {real['cfalse']}/{real['c']}, "
        f"rewired {null['cfalse']}/{null['c']}; absent episode acceptances: "
        f"real {real['afalse']}/{real['a']}, rewired {null['afalse']}/{null['a']}."
    )
    lines.extend([
        "",
        "Per-seed paired improvements/losses (source IDs and predictions in permanent JSON):",
        "",
        "| Seed | Real minus rewired accepted-correct gains | Losses | Real updated edges | Rewired updated edges |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for trial in report["trials"]:
        paired = trial["paired_diagnostics"]["real_plastic_vs_rewired_plastic"]["positive"]
        lines.append(
            f"| {trial['seed']} | {paired['correct_and_accepted_gains']} | "
            f"{paired['correct_and_accepted_losses']} | "
            f"{trial['real_learning']['edges_with_positive_trace']} | "
            f"{trial['rewired_learning']['edges_with_positive_trace']} |"
        )
    lines.extend([
        "",
        "## Interpretation limitations",
        "",
        "The prior 68 prompt cases were assistant-authored, human-unreviewed and repeatedly examined. "
        "Pooled observations overlap across seeds; percentages are descriptive. The graph "
        "uses lexical input hashing, a train-only coactivity heuristic and a target-stub "
        "permutation, not experimentally established anatomical cell-class assignments. "
        "Accepted lexical retrieval is not entailment, and learned numerical weights are "
        "not evidence of experienced memories, consciousness or a stable Pretorius identity.",
        "",
        "A gain here would require independent human-reviewed queries, anatomical cell-type "
        "aligned input and stronger degree/class-preserving controls before claiming "
        "biological specificity. A null or negative gain is an equally valid experimental "
        "finding and must not be hidden.",
        "",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--run-id", required=True, type=int)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw)
    verify(report)
    checksum = sha256(raw).hexdigest()
    raw_name = f"flywire-mb-plasticity-pilot01-run{args.run_id}.json"
    destination = ROOT / "results/associative/runs" / raw_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError("Existing immutable original result has different bytes")
    destination.write_bytes(raw)
    result_name = "FLYWIRE_MB_PLASTICITY_PILOT01_RESULTS.md"
    result_path = ROOT / "results/associative" / result_name
    result_path.write_text(
        make_report(report, raw_name, checksum, args.run_id),
        encoding="utf-8",
    )
    index = ROOT / "results/associative/README.md"
    old = index.read_text(encoding="utf-8")
    marker = "## Archived real MB plasticity Pilot 01 result"
    section = (
        f"\n\n{marker}\n\n[Measured result report]({result_name}) and "
        f"[permanent original case-level JSON](runs/{raw_name}) "
        f"are archived from [successful verified main-branch run {args.run_id}]"
        f"(https://github.com/Azimn/Pretorius-Connectome/actions/runs/{args.run_id}). "
        "All observed outcomes remain exploratory on previously inspected prompts.\n"
    )
    if marker in old:
        old = old[:old.index(marker)].rstrip()
    index.write_text(old.rstrip() + section, encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    prior = handoff.read_text(encoding="utf-8")
    handoff_marker = "## Plasticity Pilot 01 real biological execution archived"
    next_section = (
        f"\n\n{handoff_marker}\n\nReal v783 MB experiment completed in "
        f"[run {args.run_id}](https://github.com/Azimn/Pretorius-Connectome/actions/runs/{args.run_id}). "
        f"Read [permanent result report](../results/associative/{result_name}) "
        f"and [original full-case evidence](../results/associative/runs/{raw_name}). "
        "This is an exploratory previously examined assistant-authored test, "
        "not a human-confirmed demonstration of autobiographical learning. "
        "Use these measured results, not earlier chat claims, to decide the "
        "neuron-class annotation and independent review gate.\n"
    )
    if handoff_marker in prior:
        prior = prior[:prior.index(handoff_marker)].rstrip()
    handoff.write_text(prior.rstrip() + next_section, encoding="utf-8")
    print(f"Archived verified real biological run {args.run_id}, {len(raw)} bytes, sha256 {checksum}")
    print("Permanent files:", destination.relative_to(ROOT), result_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
