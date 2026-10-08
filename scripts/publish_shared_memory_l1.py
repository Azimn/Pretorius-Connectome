#!/usr/bin/env python3
"""Commit the deterministic L1 artifact and manifest using GitHub's content API.

Run only after exporter tests have passed. Existing files are immutable:
identical bytes are accepted, differing bytes fail instead of overwriting.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.shared_memory import git_blob_sha, read_l1


def gh(*args: str, input_bytes: bytes | None = None) -> str:
    completed = subprocess.run(
        ["gh", "api", *args],
        input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8")


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    if repository != "Azimn/Pretorius-Connectome":
        raise ValueError("This archive may be published only to canonical source repository")
    target = ROOT / "artifacts/shared_memory/v1"
    archive = target / "pretorius_l1_v1.jsonl.gz"
    manifest = target / "manifest.json"
    rows = read_l1(archive, manifest)
    if len(rows) != 450:
        raise ValueError("Incomplete shared memory cannot be published")
    for path in (archive, manifest):
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        expected_git_sha = git_blob_sha(raw)
        exists = subprocess.run(
            ["gh", "api", f"repos/{repository}/contents/{relative}", "--jq", ".sha"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if exists.returncode == 0:
            actual = exists.stdout.strip()
            if actual != expected_git_sha:
                raise ValueError(f"Existing published L1 file differs: {relative}")
            print(f"Verified existing immutable artifact: {relative}, SHA {actual}")
            continue
        if "404" not in exists.stderr:
            raise RuntimeError(f"Cannot check existing artifact: {exists.stderr}")
        payload = {
            "message": f"Freeze portable shared Pretorius L1 artifact: {relative}",
            "branch": "main",
            "content": base64.b64encode(raw).decode("ascii"),
        }
        response = json.loads(gh(
            "--method", "PUT", f"repos/{repository}/contents/{relative}",
            "--input", "-", input_bytes=json.dumps(payload).encode("utf-8")
        ))
        actual = response["content"]["sha"]
        if actual != expected_git_sha:
            raise ValueError("Published Git blob differs from source bytes")
        print(f"Committed immutable artifact: {relative}, SHA {actual}")


if __name__ == "__main__":
    main()
