#!/usr/bin/env python3
"""Acquire and verify exact official v3.2.0 FlyWire v783 annotations."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.cell_anchors import (
    ANNOTATION_BLOB, ANNOTATION_REF, ANNOTATION_URL, git_blob
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        content = args.output.read_bytes()
    else:
        with urlopen(ANNOTATION_URL, timeout=90) as response:
            content = response.read()
    observed = git_blob(content)
    if observed != ANNOTATION_BLOB:
        raise ValueError(
            f"Official FlyWire annotation mismatch at {ANNOTATION_REF}: {observed}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(
        f"Verified official tag-commit {ANNOTATION_REF}; "
        f"source Git blob {observed}; {len(content)} bytes; path {args.output}"
    )


if __name__ == "__main__":
    main()
