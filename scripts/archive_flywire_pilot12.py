#!/usr/bin/env python3
"""Fail-closed postmerge archival of actual original FlyWire Pilot12 cases.

An archived numerical result without its exact encoder/rewired null provenance
or both original-source-bound semantic overlay checkpoints is INCOMPLETE.
Only successful real-v783 main-branch Actions results can be written.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GRAPH = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
METHODS = (
    "real_bc01", "degree_rewired_bc01", "real_minilm",
    "degree_rewired_minilm", "real_minilm_shuffled",
    "degree_rewired_minilm_shuffled"
)
TESTS = (
    "familiar_trained_last", "controlled_last_cue_token_deletion",
    "paired_familiar_for_unseen", "genuinely_untrained_fourth_source_cue",
    "absent_heldout_episode"
)
MODEL_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def verify_original_source(report: dict) -> None:
    source = report["source"]
    graph = report["rewired_degree_matched_control"]
    model = report["frozen_pretrained_cue_encoder"]
    if (report.get("study") !=
        "Pretorius Pilot12 frozen semantic cue encoder vs degree-rewired real FlyWire"
        or report.get("status") != "original source-verified full FlyWire v783"
        or "EXTERNAL oracle-only" not in report.get("scientific_limitations", "")
        or (source.get("source_memories"), source.get("episode_count"),
            source.get("train"), source.get("validation_absent"),
            source.get("test_absent"), source.get("test_known"),
            source.get("calibration_known"), source.get(
                "eligible_never_seen_cue_events")) != (450, 27, 317, 62, 71, 159, 158, 31)
        or source.get("original_graph_array_sha256") != EXPECTED_GRAPH
        or (source.get("original_graph_neurons"),source.get("original_directed_edges"),
            source.get("original_integer_synaptic_contacts")) !=
            (139255, 15091983, 54492922)
        or source.get("original_graph_unchanged") is not True
        or source.get("BC01_artifact_sha256") !=
            "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
        or model.get("revision") != MODEL_REV
        or model.get("backend") != "onnx/model.onnx"
        or len(model.get("onnx_sha256", "")) != 64
        or len(model.get("project_384_to_256_sha256", "")) != 64
        or graph.get("original_csr_array_sha256") != EXPECTED_GRAPH
        or graph.get("source_graph_unchanged") is not True
        or graph.get("binary_indegrees_equal") is not True
        or graph.get("binary_outdegrees_equal") is not True
        or graph.get("same_number_of_synaptic_slots") is not True
        or graph.get("same_original_total_integer_synapses") is not True
        or graph.get("originally_eligible_edges") != graph.get("rewired_eligible_edges")
        or graph.get("changed_edge_destinations", 0) <= 0):
        raise ValueError("Not verified original full-v783, source-bound semantic/null evidence")
    cases = report.get("per_case_external_oracle_evidence", {})
    conditions = report.get("conditions", {})
    if set(cases) != set(METHODS) or set(conditions) != set(METHODS):
        raise ValueError("All six independent matched conditions required")
    for m in METHODS:
        if set(cases[m]) != set(TESTS):
            raise ValueError("Incomplete train, true unseen or unknown-episode probes")
        info = conditions[m]
        if (info["presentations"] != 951
            or info["trainable_original_support_slots"] !=
                graph["originally_eligible_edges"]
            or info["learned_edge_positions"] <= 0
            or not 0 <= info["gate"]["calibration_false_acceptance_rate"] <= .1):
            raise ValueError("Source training exposure, matched slots or validation leakage")
        for kind, n in zip(TESTS,(159,159,31,31,71)):
            rows = cases[m][kind]
            if (len(rows) != n
                or len({r["event_id"] for r in rows}) != n
                or conditions[m]["summaries"][kind]["original_pilot08_threshold"]["n"] != n):
                raise ValueError("Case-level source/sample count or outcome mismatch")
        if (m.endswith("_shuffled") and info["target_labels_shuffled"] is not True
            or not m.endswith("_shuffled") and info["target_labels_shuffled"] is not False):
            raise ValueError("Control target permutations not reproducibly labeled")
    for kind in TESTS:
        reference = [r["event_id"] for r in cases[METHODS[0]][kind]]
        for method in METHODS[1:]:
            if reference != [r["event_id"] for r in cases[method][kind]]:
                raise ValueError("Unpaired source-test cases across six conditions")
    manifest = report.get("model_checkpoint_metadata",{})
    if set(manifest) != {"real_minilm", "degree_rewired_minilm"}:
        raise ValueError("Two original source-bound semantic overlay checkpoints required")


def archive(raw: Path, weights: Path, run_id: str) -> None:
    original = raw.read_bytes()
    result = json.loads(original)
    verify_original_source(result)
    if not run_id.isdecimal():
        raise ValueError("Invalid Actions run ID")
    base = ROOT / "results/imprinting"
    runs = base / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    copy = runs / ("flywire-pilot12-real-v783-run" + run_id + ".json")
    if copy.exists() and copy.read_bytes() != original:
        raise ValueError("Existing same-run evidence has conflicting original bytes")
    copy.write_bytes(original)

    permanent = ROOT / "artifacts/imprinting/checkpoints"
    permanent.mkdir(parents=True, exist_ok=True)
    for name, metadata in result["model_checkpoint_metadata"].items():
        external = weights / (name + ".npz")
        b = external.read_bytes()
        if sha256(b).hexdigest() != metadata["sha256"]:
            raise ValueError(name + ": original learned synapse SHA-256 mismatch")
        if not 0 < len(b) < 10 * 1024 * 1024:
            raise ValueError("Expected compact learned weight overlay")
        target = permanent / ("pilot12-" + name + "-v783-run" + run_id + ".npz")
        if target.exists() and target.read_bytes() != b:
            raise ValueError("Existing learned source-specific original differs")
        target.write_bytes(b)

    url = "https://github.com/Azimn/Pretorius-Connectome/actions/runs/" + run_id
    lines = [
        "# Pilot12 actual original FlyWire: frozen semantic cue × rewired null",
        "",
        "Original [publisher-verified real whole-v783 run " + run_id + "](" + url + ").",
        "Complete unchanged [source case JSON](runs/" + copy.name + ").",
        "Original raw JSON SHA-256: " + sha256(original).hexdigest(),
        "Pinned external pretrained MiniLM revision: " + MODEL_REV,
        "ONNX weight file SHA-256: " +
            result["frozen_pretrained_cue_encoder"]["onnx_sha256"],
        "Projected 384→256 feature matrix SHA-256: " +
            result["frozen_pretrained_cue_encoder"]["project_384_to_256_sha256"],
        "Control switch seed: 73; changed target destinations: " +
            str(result["rewired_degree_matched_control"]["changed_edge_destinations"]),
        "",
        "Same 317 train source events, 951 first/middle/last cue presentations, "
        "159 familiar test positives, 31 matched truly untrained source fourth "
        "cue positives and 71 episode-absent negatives across SIX source-paired "
        "conditions. Every original v783 root ID, CSR index and integer contact "
        "was unchanged. Null edges were constructed in an isolated in-memory graph.",
        "",
        "| Condition | Changed learned edges | Familiar-cue top1 | "
        "Genuinely unseen fourth-cue top1 | Heldout absent false accept (validated gate) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in METHODS:
        data = result["conditions"][name]
        known = data["summaries"]["familiar_trained_last"]["validation_only_threshold"]
        novel = data["summaries"]["genuinely_untrained_fourth_source_cue"][
            "validation_only_threshold"]
        negative = data["summaries"]["absent_heldout_episode"][
            "validation_only_threshold"]
        lines.append("| " + name + " | " +
                     str(data["learned_edge_positions"]) + " | " +
                     format(known["correct_top1"], ".6f") + " | " +
                     format(novel["correct_top1"], ".6f") + " | " +
                     format(negative["absent_false_acceptance"], ".6f") + " |")
    lines += [
        "",
        "**Interpretation boundary:** The MiniLM input is a pretrained external "
        "language-model prior; the fly model trains ONLY sparse signed "
        "source-edge synaptic deltas while preserving original BC01 narrative "
        "CONTENT coordinates. All memory/event rank decisions and accept/reject "
        "gates are external oracle diagnostics. The null is matched in binary "
        "degree and number of eligible writable directed edges, not "
        "contact-weighted target input strength or actual nonzero post-training "
        "weights. Original 31 fourth cues are unreviewed lexical source fields, "
        "not independent human-written semantic paraphrases. No biological "
        "plasticity, native-language autobiographical retrieval, independent "
        "subject or topology-specific cognitive function is claimed.",
        "",
    ]
    (base / "PILOT12_REAL_SEMANTIC_REWIRED_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    index = base / "README.md"
    existing = index.read_text(encoding="utf-8") if index.exists() else "# Imprinting experiments\n"
    row = "[Pilot12 actual FlyWire semantic cue × degree-rewired null](PILOT12_REAL_SEMANTIC_REWIRED_RESULTS.md)"
    if row not in existing:
        index.write_text(existing.rstrip() + "\n\n" + row + "\n",
                         encoding="utf-8")
    handoff = ROOT / "docs/RESEARCH_HANDOFF.md"
    current = handoff.read_text(encoding="utf-8")
    marker = "## Original real FlyWire Pilot12 semantic versus rewired null run " + run_id
    if marker not in current:
        handoff.write_text(
            current.rstrip() + "\n\n" + marker + "\n\n" +
            "[Full source-linked measured Pilot12 real result]"
            "(../results/imprinting/PILOT12_REAL_SEMANTIC_REWIRED_RESULTS.md) "
            "compares original source FlyWire vs exact binary degree-switched "
            "trainable synaptic slots, frozen BC01 vs pretrained externally "
            "pinned local ONNX MiniLM input cues, source-matched and deranged "
            "content, same 317 train events and exact source test partitions. "
            "All six case-level JSON arrays, original anatomy SHA and two "
            "actual source-bound learned checkpoints are permanently committed. "
            "Only numerical cue associations are shown; the MiniLM contributes "
            "external semantics and the score decoder is an external oracle.\n",
            encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    archive(a.input, a.weights, a.run_id)
