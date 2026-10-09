#!/usr/bin/env python3
"""Fail-closed permanent original FlyWire Pilot10 experimental data archiver."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARRAY_SHA = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
METHODS = ("multicue_source", "repeated_concat_control",
           "multicue_shuffled_targets", "zero_overlay")
SCENARIOS = ("first_trained_cue", "last_trained_cue",
             "last_trained_cue_drop_token", "absent_episode_last_cue")


def verify(r: dict) -> None:
    src, split = r["source"], r["split"]
    if (r.get("study") != "Pretorius direct FlyWire Pilot10 multi-cue binding"
        or r.get("topology_status") != "verified original full FlyWire v783"
        or "EXTERNAL oracle-assisted" not in r.get("scientific_boundary", "")
        or (src["canonical_memories"], src["source_episodes"],
            src["neurons"], src["aggregate_directed_edges"],
            src["integer_synapse_contacts"]) != (450, 27, 139255, 15091983, 54492922)
        or src["original_anatomy_array_sha256"] != ARRAY_SHA
        or src["original_anatomy_unchanged"] is not True
        or src["encoder"] != "persona-net-ExperienceEncoder-sensory256-v1"
        or src["bc01_artifact_sha256"] !=
        "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"):
        raise ValueError("Not source-verified real whole-brain original FlyWire")
    if (tuple(split.get(k) for k in (
        "seed", "imprinted_train", "validation_unknown", "test_unknown",
        "selected_train_test", "cue_exposures_per_training_memory"))
        != (31, 317, 62, 71, 159, 3)
        or split["fixed_acceptance_threshold"] != 0.0082783
        or split["primary_positive_scenario"] !=
            "last_trained_cue is explicitly trained, not held out"
        or not split["checkpoints"]["multicue_source_sha256"]):
        raise ValueError("Source split or learning budget changed")
    summaries, cases = r["comparisons"], r["oracle_only_case_results"]
    if set(summaries) != set(METHODS) or set(cases) != set(METHODS):
        raise ValueError("Missing any of four required controls")
    for method in METHODS:
        if set(cases[method]) != set(SCENARIOS):
            raise ValueError("Incomplete source-paired evaluation")
        for scenario in SCENARIOS:
            rows = cases[method][scenario]
            stats = summaries[method][scenario]
            n = 71 if scenario == SCENARIOS[-1] else 159
            if (len(rows) != n or stats["cases"] != n
                or len({row["event_id"] for row in rows}) != n
                or not 0 <= stats["acceptance_rate"] <= 1):
                raise ValueError("Source-linked per-case trial malformed")
            if scenario == SCENARIOS[-1]:
                if (any(row["known"] for row in rows)
                    or not 0 <= stats["absent_false_acceptance"] <= 1):
                    raise ValueError("Absent-only evaluation invalid")
            elif (any(not row["known"] for row in rows)
                  or not 0 <= stats["correct_top1"] <= 1):
                raise ValueError("Positive evaluation invalid")
    for scenario in SCENARIOS:
        ids = [x["event_id"] for x in cases[METHODS[0]][scenario]]
        for method in METHODS[1:]:
            if ids != [x["event_id"] for x in cases[method][scenario]]:
                raise ValueError("Unequal cases between experimental methods")
    if (any(summaries[m]["imprint_presentations"] != 951 for m in METHODS[:-1])
        or summaries[METHODS[-1]]["imprint_presentations"] != 0
        or summaries[METHODS[-1]]["modified_original_synapses"] != 0
        or summaries[METHODS[0]]["modified_original_synapses"] <= 0):
        raise ValueError("Expected real synaptic training absent")


def archive(original: Path, run_id: str) -> None:
    b = original.read_bytes()
    r = json.loads(b)
    verify(r)
    if not run_id.isdecimal():
        raise ValueError("Actions run id invalid")
    folder = ROOT / "results/imprinting"
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    saved = runs / ("direct-flywire-imprint-pilot10-run" + run_id + ".json")
    if saved.exists() and saved.read_bytes() != b:
        raise ValueError("Existing archived data conflict")
    saved.write_bytes(b)
    url = "https://github.com/Azimn/Pretorius-Connectome/actions/runs/" + run_id
    lines = [
        "# Pilot10: real original FlyWire v783 multiple-cue imprint measured results",
        "",
        "Real publisher-verified workflow: [run " + run_id + "](" + url + ").",
        "Original full per-case JSON: [source evidence](runs/" + saved.name + ").",
        "SHA-256 of unmodified original result: " + sha256(b).hexdigest(),
        "Learned synaptic checkpoint SHA-256: " +
        r["split"]["checkpoints"]["multicue_source_sha256"],
        "",
        "Original graph unchanged: 139,255 neurons, 15,091,983 directed pairs, "
        "54,492,922 anatomical integer contacts. Original canonical 450 memories, "
        "317 training events, 159 positive and 71 absent-episode evaluation cases.",
        "",
        "| Condition | Imprints | Changed source synaptic edges | First cue correct | "
        "Last cue correct | Deletion correct | Absent false acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = (
        "Explicit three source-literal cue bindings",
        "Repeated concatenated-cue equal exposure",
        "Multiple source cues with deranged content",
        "Unmodified zero-delta baseline",
    )
    for condition, label in zip(METHODS, labels):
        t = r["comparisons"][condition]
        lines.append("| " + label + " | " +
                     str(t["imprint_presentations"]) + " | " +
                     str(t["modified_original_synapses"]) + " | " +
                     format(t["first_trained_cue"]["correct_top1"], ".6f") + " | " +
                     format(t["last_trained_cue"]["correct_top1"], ".6f") + " | " +
                     format(t["last_trained_cue_drop_token"]["correct_top1"], ".6f") + " | " +
                     format(t["absent_episode_last_cue"]["absent_false_acceptance"], ".6f") + " |")
    lines.extend([
        "",
        "**Interpretation:** The last source literal cue was explicitly trained "
        "in the multi-cue condition and is NOT novel semantic paraphrase recall. "
        "Deletion-cue probes are mechanical lexical variations, not independent "
        "human semantic evaluation. Inference returned a 256D lexical vector; "
        "event identity was assigned using EXTERNAL evaluator-only content "
        "codebooks. The training exposures and nominal rate budgets match "
        "but effective edge counts and neural capacity do not. No rewired "
        "biological control, fly physiology, autonomous narrative recollection "
        "or cognitive subject is demonstrated.",
        "",
    ])
    (folder / "PILOT10_REAL_MULTICUE_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    index = folder / "README.md"
    text = index.read_text(encoding="utf-8") if index.exists() else "# Pretorius imprinting\n"
    entry = "[Pilot10 real v783 multi-cue results](PILOT10_REAL_MULTICUE_RESULTS.md)"
    if entry not in text:
        index.write_text(text.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    text = handoff.read_text(encoding="utf-8")
    marker = "## Direct real FlyWire multi-cue Pilot10 archived " + run_id
    if marker not in text:
        handoff.write_text(
            text.rstrip() + "\n\n" + marker + "\n\n" +
            "[Original full per-case direct synaptic imprint results]"
            "(../results/imprinting/PILOT10_REAL_MULTICUE_RESULTS.md) "
            "were archived from successful real-v783 source-verified Actions. "
            "Compare multiple literal-source-cue bindings against equal-exposure "
            "old concatenation, deranged narrative association and zero controls. "
            "This is oracle-assisted numerical lexical association, not semantic "
            "or autonomous autobiographical memory.\n",
            encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    archive(args.input, args.run_id)
