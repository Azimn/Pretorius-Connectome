#!/usr/bin/env python3
"""Serve the validated Vector Fly lexical database to local AI agents.

No cloud fees, third-party framework or database mutation. Bind defaults to
127.0.0.1; binding a network-facing address requires a bearer token via env.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.vector_api import make_server
from pretorius_connectome.vector_store import VectorFlyStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Default localhost; remote access requires --token-env")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default=None,
                        help="Environment variable containing 24+ character bearer token")
    args = parser.parse_args()
    token = os.environ.get(args.token_env) if args.token_env else None
    if args.token_env and not token:
        parser.error("--token-env is set but the token variable is unavailable")
    with VectorFlyStore(args.database, args.cache_dir, check_vectors=True) as store:
        server = make_server(store, host=args.host, port=args.port, token=token)
        print(
            f"Vector Fly read-only {store.info()['scope']} retrieval listening on "
            f"{args.host}:{server.server_address[1]} (lexical; no memory writes)",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
