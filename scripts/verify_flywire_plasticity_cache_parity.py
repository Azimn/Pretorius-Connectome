#!/usr/bin/env python3
"""Fail-closed numerical equivalence check for original versus shared L2 runs.

CSR floating-point operation order need not produce byte-identical overlay
checksums. No event IDs, outputs, threshold choices or non-numeric attributes
may differ. All numerical scores must agree within a strict tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(a, b, where="root"):
    if type(a) is not type(b):
        raise AssertionError(f"{where}: types {type(a).__name__} != {type(b).__name__}")
    if isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError(f"{where}: dictionary keys differ")
        for key in a:
            if key == "learned_operator_sha256" and where.endswith("_learning"):
                # Exact in-run bytes remain archived in each input report.
                # They are not reliable cross-fit equality evidence by themselves.
                continue
            compare(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"{where}: lengths differ")
        for i, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f"{where}[{i}]")
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"{where}: numerical values {a} != {b}")
    elif a != b:
        raise AssertionError(f"{where}: values {a!r} != {b!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("cached", type=Path)
    args = parser.parse_args()
    old = json.loads(args.original.read_text(encoding="utf-8"))
    new = json.loads(args.cached.read_text(encoding="utf-8"))
    if old["shared_l2"] != "unshared train-only TF-IDF" or new["shared_l2"] != "train-only TF-IDF":
        raise AssertionError("not a cached-versus-uncached comparison")
    a, b = dict(old), dict(new)
    a.pop("shared_l2")
    b.pop("shared_l2")
    compare(a, b)
    for left, right in zip(old["trials"], new["trials"]):
        for key in ("real_learning", "rewired_learning"):
            assert left[key]["baseline_operator_sha256"] == right[key]["baseline_operator_sha256"]
    print("PASS: source IDs, split, all decisions, all predictions, edge counts, "
          "calibration, source-linked cases and numerical scores agree "
          "(floating tolerance 1e-8 abs / 1e-10 rel); "
          "raw learned-weight SHA values retained separately for provenance")


if __name__ == "__main__":
    main()
