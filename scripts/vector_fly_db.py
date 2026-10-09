#!/usr/bin/env python3
"""Local, persistent, exact sparse vector lookup for Pretorius autobiography.

No API fees; no model download; no semantic or identity claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.vector_store import build_store, VectorFlyStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for mode in ("build", "query", "get", "verify", "info"):
        cmd = sub.add_parser(mode)
        cmd.add_argument("--database", type=Path, required=True)
        cmd.add_argument("--cache-dir", type=Path, required=True,
                         help="Already-validated and split-fitted L2 v2 directory")
        if mode == "build":
            cmd.add_argument("--scope", choices=("train", "all"), default="train",
                             help="train for experiments; all for everyday 450-memory browsing")
        if mode == "query":
            cmd.add_argument("--text", required=True)
            cmd.add_argument("--top-k", type=int, default=5)
            cmd.add_argument("--episode-id")
            cmd.add_argument("--provenance")
        if mode == "get":
            cmd.add_argument("--event-id", required=True)
    args = parser.parse_args()
    if args.action == "build":
        response = build_store(args.database, args.cache_dir, scope=args.scope)
    else:
        with VectorFlyStore(args.database, args.cache_dir) as db:
            if args.action == "query":
                response = {
                    "query": args.text,
                    "source": "canonical pinned Pretorius L1",
                    "vector_kind": "train-fitted TF-IDF lexical, not semantic",
                    "scope": db.info()["scope"],
                    "results": db.search(args.text, top_k=args.top_k,
                                         episode_id=args.episode_id,
                                         provenance=args.provenance),
                }
            elif args.action == "get":
                response = {"record": db.get(args.event_id),
                            "event_id": args.event_id}
            elif args.action == "verify":
                response = db.verify_vectors()
            else:
                response = db.info()
    print(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
