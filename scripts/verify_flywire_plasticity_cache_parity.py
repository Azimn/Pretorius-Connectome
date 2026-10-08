#!/usr/bin/env python3
"""Fail-closed comparison of independently built deterministic L2 v2 runs.

Both runners must consume separately fitted, source-validated v2 caches
constructed in different Python processes. The historical uncached
TfidfVectorizer(max_features=8192) is a different representation and cannot
be treated as an equivalent v2 control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(left, right, where="root"):
    if type(left) is not type(right):
        raise AssertionError(f"{where}: types differ")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{where}: keys differ")
        for key in left:
            compare(left[key], right[key], where + "." + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"{where}: lengths differ")
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{where}[{i}]")
    elif left != right:
        raise AssertionError(f"{where}: {left!r} != {right!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2_process_a", type=Path)
    parser.add_argument("v2_process_b", type=Path)
    args = parser.parse_args()
    first = json.loads(args.v2_process_a.read_text(encoding="utf-8"))
    second = json.loads(args.v2_process_b.read_text(encoding="utf-8"))
    for result in (first, second):
        if (result.get("shared_l2_schema") != "pretorius.shared-features.v2"
                or result.get("shared_l2_encoder") != "sklearn-tfidf-word12-rankstable-v2"
                or result.get("shared_l2") != "deterministic train-only TF-IDF v2"):
            raise AssertionError("Not a deterministic shared L2 v2 experiment")
        if any(t.get("l2_shard_sha256") is None for t in result["trials"]):
            raise AssertionError("Missing independently fitted L2 shard fingerprints")
    compare(first, second)
    print("PASS: two independent v2 processes have identical feature shards, "
          "floating edge operator hashes, every raw source case, predictions, "
          "accepted decisions, calibrated scores and learning audits.")


if __name__ == "__main__":
    main()
