#!/usr/bin/env python3
"""Permanently archive only a verified actual frozen FlyWire Pilot11 result.

This copies original JSON bytes, checks source/graph hashes and paired cases,
and reports both legacy and validation-calibrated heldout performance.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
SCENARIOS = (
    "calibration_trained", "calibration_absent",
    "test_trained", "test_absent",
    "paired_trained_unseen_subset", "unseen_fourth_cue"
)
CKPT = "fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a"
PRIOR_SHA = "ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8"


def verify(d):
    evidence, cal = d["evidence"], d["calibration"]
    if (d.get("study") != "FlyWire Pilot11 original learned state: unseen source cue and validation-only rejection"
        or d.get("topology_status") != "source-verified original complete FlyWire v783"
        or "EXTERNAL oracle-only" not in d.get("limitations", "")
        or evidence.get("original_pilot10_real_run") != 37869353939
        or evidence.get("checkpoint_sha256") != CKPT
        or evidence.get("original_case_json_sha256") != PRIOR_SHA
        or evidence.get("anatomical_csr_array_sha256") != EXPECTED
        or evidence.get("original_anatomy_unchanged") is not True
        or (evidence.get("events"), evidence.get("train_events"),
            evidence.get("test_positive_events"),
            evidence.get("validation_unknown_events"),
            evidence.get("test_unknown_events")) != (450, 317, 159, 62, 71)):
        raise ValueError("Not original verified real FlyWire v783 frozen checkpoint")
    eligible = evidence.get("eligible_unseen_source_events")
    if type(eligible) is not int or not 0 < eligible <= 159:
        raise ValueError("No verified eligible unseen source cue test events")
    if (cal.get("calibration_known_events") != 158
        or cal.get("calibration_absent_events") != 62
        or cal.get("calibration_false_acceptances", 100) > 6
        or cal.get("calibration_false_acceptance_rate", 1) > .10
        or not isinstance(cal.get("threshold"), (int, float))):
        raise ValueError("Calibration validation-only rejection gate altered")
    cases, summaries = d["case_results"], d["comparisons"]
    if set(cases) != set(SCENARIOS) or set(summaries) != set(SCENARIOS):
        raise ValueError("Original calibration and test cases missing")
    expected_sizes = (158, 62, 159, 71, eligible, eligible)
    for key, n in zip(SCENARIOS, expected_sizes):
        rows = cases[key]
        if (len(rows) != n or len({r["event_id"] for r in rows}) != n
            or any(r["condition"] != key for r in rows)
            or set(summaries[key]) != {
                "original_pilot08_threshold", "validation_constrained_threshold"
            }):
            raise ValueError("Malformed source-linked per-case rows")
        for gate, stat in summaries[key].items():
            if stat["n"] != n or not 0 <= stat["acceptance_rate"] <= 1:
                raise ValueError("Numerical summary and source case count diverged")
    if ([r["event_id"] for r in cases["paired_trained_unseen_subset"]]
        != [r["event_id"] for r in cases["unseen_fourth_cue"]]):
        raise ValueError("Paired unseen literal source cue comparisons are not paired")
    known = set(r["event_id"] for r in cases["test_trained"])
    absent = set(r["event_id"] for r in cases["test_absent"])
    if known & absent:
        raise ValueError("Test positives overlap truly absent episode source events")
    if not set(r["event_id"] for r in cases["unseen_fourth_cue"]) <= known:
        raise ValueError("Unseen case set differs from original 159 test positives")
    if any(r["unseen_shared_feature_count"] < 0 for r in cases["unseen_fourth_cue"]):
        raise ValueError("Invalid frozen encoding overlap statistic")


def archive(source: Path, run_id: str):
    original = source.read_bytes()
    doc = json.loads(original)
    verify(doc)
    if not run_id.isdecimal():
        raise ValueError("Invalid GitHub Actions source run")
    dest = ROOT / "results/imprinting"
    history = dest / "runs"
    history.mkdir(parents=True, exist_ok=True)
    result = history / ("direct-flywire-imprint-pilot11-run" + run_id + ".json")
    if result.exists() and result.read_bytes() != original:
        raise ValueError("Conflicting original Pilot11 case evidence")
    result.write_bytes(original)
    url = "https://github.com/Azimn/Pretorius-Connectome/actions/runs/" + run_id
    methods = (
        ("test_trained", "159 original trained last cues"),
        ("paired_trained_unseen_subset", "Trained cues on unseen-eligible subset"),
        ("unseen_fourth_cue", "Genuinely untrained literal source cue"),
        ("test_absent", "71 truly untrained-episode absent events")
    )
    lines = [
        "# Pilot11: frozen original biological FlyWire novel literal cue and rejection",
        "",
        "Original publisher-verified biological workflow: [run " + run_id + "](" + url + ").",
        "Full unmodified original per-case JSON: [source](runs/" + result.name + ").",
        "Raw case JSON SHA-256: " + sha256(original).hexdigest(),
        "Original learned source-bound checkpoint SHA-256: " + CKPT,
        "Original synaptic state unchanged: 139,255 FlyWire neurons, 15,091,983 aggregate directed edges and 54,492,922 integer biological synaptic contacts.",
        "",
        "Genuinely untrained source literal cue eligible test memories: " +
            str(doc["evidence"]["eligible_unseen_source_events"]) + " out of 159.",
        "Original acceptance threshold: 0.0082783. New threshold: " +
            str(doc["calibration"]["threshold"]) +
            ", selected using only 158 non-test training positives and 62 absent validation events; allowed <=6 false accepts in validation.",
        "",
        "| Test condition | N | Original correct top1 | Original correctly accepted | "
        "Original absent false acceptance | Validation-gated correctly accepted | "
        "Validation-gated absent false acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in methods:
        old = doc["comparisons"][key]["original_pilot08_threshold"]
        new = doc["comparisons"][key]["validation_constrained_threshold"]
        lines.append(
            "| " + label + " | " + str(old["n"]) + " | " +
            (format(old["correct_top1"], ".6f") if "correct_top1" in old else "n/a") +
            " | " + (format(old["correct_and_accepted"], ".6f")
                      if "correct_and_accepted" in old else "n/a") +
            " | " + (format(old["absent_false_acceptance"], ".6f")
                      if "absent_false_acceptance" in old else "n/a") +
            " | " + (format(new["correct_and_accepted"], ".6f")
                      if "correct_and_accepted" in new else "n/a") +
            " | " + (format(new["absent_false_acceptance"], ".6f")
                      if "absent_false_acceptance" in new else "n/a") + " |"
        )
    novel = doc["comparisons"]["unseen_fourth_cue"]["original_pilot08_threshold"]
    lines += [
        "",
        "Untrained source cues with no overlap with any of the three trained lexical cue coordinates: " +
            str(novel["no_shared_source_feature_count"]) + "/" + str(novel["n"]) + ".",
        "",
        "**Scientific boundary:** Neural inference receives the cue only and "
        "returns a numerical BC01 lexical content vector. Event IDs and "
        "accept/reject similarities are EXTERNAL oracle-only diagnostics. "
        "A fourth source cue is not an independently human-reviewed semantic "
        "paraphrase, and 10% validation false acceptance is not a guarantee "
        "for independent test episodes. The original fly synaptic anatomy and "
        "saved learned overlay were not changed. No fly cognition, autonomous "
        "recollection or uniquely biological advantage has been established.",
        "",
    ]
    (dest / "PILOT11_REAL_UNSEEN_CUE_REJECTION_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    index = dest / "README.md"
    prior_index = index.read_text(encoding="utf-8") if index.exists() else "# Imprinting experiments\n"
    entry = "[Pilot11 frozen real FlyWire unseen cue and rejection](PILOT11_REAL_UNSEEN_CUE_REJECTION_RESULTS.md)"
    if entry not in prior_index:
        index.write_text(prior_index.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    earlier = handoff.read_text(encoding="utf-8")
    marker = "## Frozen original FlyWire Pilot11 unseen cue and rejection run " + run_id
    if marker not in earlier:
        handoff.write_text(
            earlier.rstrip() + "\n\n" + marker + "\n\n" +
            "[Full real Pilot11 paired case evidence]"
            "(../results/imprinting/PILOT11_REAL_UNSEEN_CUE_REJECTION_RESULTS.md) "
            "uses the exact permanently archived Pilot10 real synaptic state "
            "without any new learning. It measures first originally untrained "
            "source cue among cases with four or more literal cue fields, "
            "paired to that same event's trained last cue, and validates a "
            "rejection threshold fit only to separate calibration episodes. "
            "Interpret the measured unseen-cue and test false acceptance "
            "together, not as autonomous semantic recall.\n",
            encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    archive(args.input, args.run_id)
