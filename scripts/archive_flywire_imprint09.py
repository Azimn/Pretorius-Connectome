#!/usr/bin/env python3
"""Permanently preserve exact real Pilot09 diagnostics after successful main run.

The archiver verifies real original v783 provenance, exact prior learned
checkpoint, paired event sets and full original source-case JSON.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
METHODS = ("trained_source", "shuffled_content", "frozen_untrained")
GROUPS = ("trained_cue", "withheld_last_cue", "absent_episode_last_cue")
EXPECTED_CKPT = "e6edf8dd68e84540140612dbcbc42f827e00ce707d9f2947adf3342e215fcb6d"


def validate(r: dict):
    d = r["data_source"]
    if (r.get("study") != "Direct FlyWire Pilot09: frozen Pilot08 exact-cue versus cross-cue audit"
        or r.get("status") != "real v783 diagnostic"
        or d.get("original_workflow_run_id") != 37863818653
        or d.get("original_checkpoint_sha256") != EXPECTED_CKPT
        or d.get("topology_array_sha256") != EXPECTED_SHA
        or d.get("anatomical_synapse_counts_intact") is not True
        or (d.get("canonical_memory_count"), d.get("train_events"),
            d.get("evaluation_trained_events"), d.get("evaluation_absent_events"))
            != (450, 317, 159, 71)
        or d.get("bc01_artifact_sha256") !=
           "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"):
        raise ValueError("Not original source-matched whole-brain experiment")
    summaries = r.get("comparisons", {})
    all_cases = r.get("case_results", {})
    if set(summaries) != set(METHODS) or set(all_cases) != set(METHODS):
        raise ValueError("Missing three original comparison conditions")
    for model in METHODS:
        if not set(GROUPS).issubset(summaries[model]) or set(all_cases[model]) != set(GROUPS):
            raise ValueError("Incomplete comparison scenarios")
        for scenario, n in zip(GROUPS, (159, 159, 71)):
            rows = all_cases[model][scenario]
            score = summaries[model][scenario]
            if (len(rows) != n or score.get("count") != n
                or len({x["event_id"] for x in rows}) != n
                or not 0 <= score.get("fraction_accepted", -1) <= 1):
                raise ValueError("Malformed original case evidence")
            if scenario != GROUPS[2] and not 0 <= score.get("correct_top1", -1) <= 1:
                raise ValueError("Known-event recall score absent")
            if scenario == GROUPS[2] and any(x["known"] for x in rows):
                raise ValueError("Absent events contaminated with train cases")
    for scenario in GROUPS:
        identity = [x["event_id"] for x in all_cases[METHODS[0]][scenario]]
        if any(identity != [x["event_id"] for x in all_cases[m][scenario]]
               for m in METHODS[1:]):
            raise ValueError("Methods evaluated on mismatched cases")
    if (summaries[METHODS[0]]["modified_synaptic_edges"] != 25780
        or summaries[METHODS[2]]["modified_synaptic_edges"] != 0
        or summaries[METHODS[0]][GROUPS[1]]["correct_top1"] != 0.0):
        raise ValueError("Pilot08 original negative result did not replay")


def archive(raw: Path, run_id: str) -> None:
    content = raw.read_bytes()
    data = json.loads(content)
    validate(data)
    if not run_id.isdecimal():
        raise ValueError("Invalid run ID")
    folder = ROOT / "results/imprinting"
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    dest = runs / ("direct-flywire-imprint-pilot09-run" + run_id + ".json")
    if dest.exists() and dest.read_bytes() != content:
        raise ValueError("Existing result differs")
    dest.write_bytes(content)
    link = "https://github.com/Azimn/Pretorius-Connectome/actions/runs/" + run_id
    out = [
        "# Pilot 09: exact-cue vs heldout-cue retrieval on frozen real FlyWire",
        "",
        "Original saved learned synapses from Pilot08 run 37863818653.",
        "Real diagnostic workflow: [run " + run_id + "](" + link + ").",
        "Complete exact original per-case JSON: [permanent result](runs/" + dest.name + ").",
        "Raw SHA256: " + sha256(content).hexdigest(),
        "Frozen learned checkpoint SHA256: " + EXPECTED_CKPT,
        "",
        "Original real 139,255-neuron, 15,091,983-edge biological synapse arrays were unchanged.",
        "Seed 31 uses same 317 imprinted memories, 159 paired trained-memory probes, "
        "and 71 absent-episode test probes. External codebook used only after inference.",
        "",
        "| Synaptic condition | Correct on trained cue | Correct on different literal cue | "
        "Correct and accepted on different cue | False acceptance of absent episode | "
        "Mean cross-cue feature cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, name in (
        ("trained_source", "Original learned overlay"),
        ("shuffled_content", "Same-graph shuffled content"),
        ("frozen_untrained", "Original zero overlay"),
    ):
        trial = data["comparisons"][method]
        a, b, c = [trial[x] for x in GROUPS]
        out.append(
            "| " + name + " | " + format(a["correct_top1"], ".6f") +
            " | " + format(b["correct_top1"], ".6f") +
            " | " + format(b["correct_and_accepted"], ".6f") +
            " | " + format(c["fraction_accepted"], ".6f") +
            " | " + format(b["mean_cross_cue_input_cosine"], ".6f") + " |"
        )
    nooverlap = data["comparisons"]["trained_source"]["withheld_last_cue"][
        "fraction_cross_cue_no_shared_feature"
    ]
    out.extend([
        "",
        "Fraction of trained-memory cues whose training versus distinct literal "
        "test input had no shared hashed feature coordinates: " +
        format(nooverlap, ".6f") + ".",
        "",
        "**Boundary:** This is retrospective diagnostic evaluation of a frozen "
        "source-bound biological-connectome weight overlay, not new learning "
        "or independent confirmation of semantic generalization. The system "
        "cannot generate autobiographical prose, and event identities require "
        "an external oracle-only candidate-content dictionary. Original source "
        "recall cues are not independently human-reviewed semantic paraphrases.",
        "",
    ])
    (folder / "PILOT09_REAL_CUE_TRANSFER_RESULTS.md").write_text(
        "\n".join(out), encoding="utf-8"
    )
    index = folder / "README.md"
    previous = index.read_text(encoding="utf-8") if index.exists() else "# Pretorius imprinting\n"
    entry = "[Pilot09 frozen real-FlyWire cue transfer](PILOT09_REAL_CUE_TRANSFER_RESULTS.md)"
    if entry not in previous:
        index.write_text(previous.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    previous = handoff.read_text(encoding="utf-8")
    marker = "## Frozen Pilot09 real synaptic recall diagnostic run " + run_id
    if marker not in previous:
        handoff.write_text(
            previous.rstrip() + "\n\n" + marker + "\n\n" +
            "[Pilot09 real per-case results](../results/imprinting/PILOT09_REAL_CUE_TRANSFER_RESULTS.md) "
            "compare original source-trained 317-event synaptic overlay and same-graph "
            "shuffled/zero controls using 159 exact-trained-cue and separate literal "
            "cue tests plus 71 absent-episode tests. Learned checkpoint, original "
            "topology and calibration were frozen from Pilot08. See actual measured "
            "contrast before designing new feature encoding or neural plasticity.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    archive(a.input, a.run_id)
