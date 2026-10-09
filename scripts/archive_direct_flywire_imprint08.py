#!/usr/bin/env python3
"""Fail-closed archiver for measured real original FlyWire direct-imprint results."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("learned_real_or_synthetic_graph",
              "shuffled_content_same_graph", "zero_overlay_same_graph")
EXPECTED_SHA = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}


def verify(report: dict) -> None:
    g, tr, inp = (report[key] for key in ("graph", "training", "input"))
    if (report.get("experiment") !=
        "Direct synaptic imprinting Pilot 08 (heteroassociative)"
        or report.get("evidence_status") !=
        "Measured real FlyWire v783 original whole-brain topology"
        or "EXTERNAL oracle-assisted" not in report.get("claim_boundary", "")
        or (g.get("neurons"), g.get("raw_aggregated_directed_edges"),
            g.get("original_integer_synaptic_contacts")) !=
            (139255, 15091983, 54492922)
        or g.get("array_sha256") != EXPECTED_SHA
        or g.get("original_anatomy_unchanged") is not True):
        raise ValueError("Not a verified original biological CSR result")
    if (inp.get("canonical_source_events") != 450
        or inp.get("source_episodes") != 27
        or inp.get("representation") != "persona-net-ExperienceEncoder-sensory256-v1"
        or not inp.get("cache_artifact_sha256")
        or inp.get("episodes_disjoint") is not True
        or tr.get("seed") != 31
        or not 250 <= tr.get("imprinted_events", -1) < 450
        or tr.get("neuron_cells_per_feature") != 32
        or tr.get("full_corpus_development_mode") is not False
        or len(tr.get("checkpoint_sha256", "")) != 64):
        raise ValueError("Source version, training population or checkpoint mismatch")
    models = report.get("methods", {})
    cases = report.get("external_evaluator_case_results", {})
    if set(models) != set(CONDITIONS) or set(cases) != set(CONDITIONS):
        raise ValueError("Original treatment, shuffled and zero controls required")
    for label in CONDITIONS:
        m = models[label]["metrics_at_learned_calibrated_threshold"]
        if (not 0 <= m["positive_correct_top1"] <= 1
            or not 0 <= m["positive_correct_and_accepted"] <= 1
            or not 0 <= m["absent_false_acceptance"] <= 1
            or len(cases[label]["known"]) != m["positive_n"]
            or len(cases[label]["absent"]) != m["absent_n"]):
            raise ValueError("Aggregate and per-case results disagree")
    if (models[CONDITIONS[0]]["nonzero_overlay_edges"] <= 0
        or models[CONDITIONS[0]]["training_updates"] != tr["imprinted_events"]
        or models[CONDITIONS[1]]["training_updates"] != tr["imprinted_events"]
        or models[CONDITIONS[2]]["nonzero_overlay_edges"] != 0
        or models[CONDITIONS[2]]["training_updates"] != 0):
        raise ValueError("Expected numerical learning or frozen baseline missing")
    for kind in ("known", "absent"):
        ids = [r["event_id"] for r in cases[CONDITIONS[0]][kind]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evaluation case")
        for label in CONDITIONS[1:]:
            if ids != [r["event_id"] for r in cases[label][kind]]:
                raise ValueError("Comparison did not use matched cases")


def archive(source: Path, run_id: str) -> None:
    data_bytes = source.read_bytes()
    report = json.loads(data_bytes)
    verify(report)
    if not run_id.isdecimal():
        raise ValueError("Run ID must be numeric")
    dest = ROOT / "results/imprinting"
    runs = dest / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / ("direct-flywire-imprint-pilot08-run" + run_id + ".json")
    if destination.exists() and destination.read_bytes() != data_bytes:
        raise ValueError("Conflicting prior archived source")
    destination.write_bytes(data_bytes)
    action = ("https://github.com/Azimn/Pretorius-Connectome/actions/runs/" + run_id)
    tr, graph = report["training"], report["graph"]
    lines = [
        "# Direct FlyWire original whole-brain imprint: Pilot 08 measured result",
        "",
        "Real publisher-verified workflow: [run " + run_id + "](" + action + ").",
        "Full original case evidence: [JSON](runs/" + destination.name + ").",
        "Exact source case JSON SHA-256: " + sha256(data_bytes).hexdigest(),
        "Original checkpoint SHA-256: " + tr["checkpoint_sha256"],
        "",
        "Source: 450 pinned memories; actual episode-separated imprints: " +
        str(tr["imprinted_events"]) + ".",
        "Original graph: " + format(graph["neurons"], ",") + " neurons, " +
        format(graph["raw_aggregated_directed_edges"], ",") + " directed edges, " +
        format(graph["original_integer_synaptic_contacts"], ",") +
        " biological synaptic contacts; original arrays unchanged.",
        "",
        "| Condition | Learned edges | Correct top1 | Correct and accepted | Absent false acceptance |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = ("Real topology, content learned", "Same topology, shuffled content",
              "Same topology, untrained")
    for key, label in zip(CONDITIONS, labels):
        m = report["methods"][key]
        s = m["metrics_at_learned_calibrated_threshold"]
        lines.append("| " + label + " | " +
                     format(m["nonzero_overlay_edges"], ",") + " | " +
                     format(s["positive_correct_top1"], ".4f") + " | " +
                     format(s["positive_correct_and_accepted"], ".4f") + " | " +
                     format(s["absent_false_acceptance"], ".4f") + " |")
    lm = report["methods"][CONDITIONS[0]]["metrics_at_learned_calibrated_threshold"]
    sm = report["methods"][CONDITIONS[1]]["metrics_at_learned_calibrated_threshold"]
    lines.extend([
        "",
        "Learned-minus-shuffled correct-and-accepted difference: " +
        format(lm["positive_correct_and_accepted"] -
               sm["positive_correct_and_accepted"], "+.6f") + ".",
        "",
        "**Limitations:** These are lexical hash-vector associations constrained "
        "to directed edges of the real fly graph, not native autobiographical "
        "recollection. An **external oracle-only candidate codebook** supplies "
        "event identity during evaluation. No independent human-reviewed cues, "
        "capacity-matched rewired comparator, validated semantic encoder, "
        "sign-specific fly physiology, or generative text decoder is present.",
        "",
    ])
    (dest / "PILOT08_REAL_V783_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    index = dest / "README.md"
    content = index.read_text(encoding="utf-8") if index.exists() else "# Imprinting experiments\n"
    line = "[Pilot 08 direct real FlyWire imprint](PILOT08_REAL_V783_RESULTS.md)"
    if line not in content:
        index.write_text(content.rstrip() + "\n\n" + line + "\n",
                         encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    content = handoff.read_text(encoding="utf-8")
    title = "## Direct imprint Pilot 08 real result archived: run " + run_id
    if title not in content:
        handoff.write_text(
            content.rstrip() + "\n\n" + title + "\n\n" +
            "[Full measured results](../results/imprinting/PILOT08_REAL_V783_RESULTS.md) "
            "and original case-level JSON are now committed. The real FlyWire "
            "original CSR remained unmodified; learned signed synaptic overlays "
            "can produce 256D lexical readouts, but event identification remains "
            "external oracle-assisted. Use these measured contrasts to continue "
            "[Issue #17](https://github.com/Azimn/Pretorius-Connectome/issues/17). "
            "Do not claim autonomous recollection or topology-specific benefit.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    archive(args.input, args.run_id)
