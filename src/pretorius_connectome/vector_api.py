"""Read-only, local-first JSON/HTTP API for the verified Vector Fly database.

No extra framework, cloud server, neural-weight mutation, or paid LLM API.
A single-threaded server deliberately shares one validated read-only SQLite
connection; this is suitable for small local bot populations, not public SaaS.
"""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import re
from urllib.parse import unquote, urlsplit

from pretorius_connectome.vector_store import VectorFlyStore

API_VERSION = "vector-fly-retrieval/1"
MAX_REQUEST_BYTES = 8192
MAX_QUERY_CHARS = 2000
MAX_TOP_K = 20
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


class InputError(ValueError):
    """Invalid request, not an internal service error."""


def _query_body(data: object) -> tuple[str, int, str | None, str | None]:
    if not isinstance(data, dict):
        raise InputError("JSON request must be an object")
    if set(data) - {"query", "top_k", "episode_id", "provenance"}:
        raise InputError("Unsupported search parameter")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise InputError("query must be a nonempty string of at most 2000 characters")
    top_k = data.get("top_k", 5)
    if type(top_k) is not int or not (1 <= top_k <= MAX_TOP_K):
        raise InputError("top_k must be an integer from 1 through 20")
    for key in ("episode_id", "provenance"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str)
                                  or not value or len(value) > 96):
            raise InputError(key + " must be a nonempty string when supplied")
    return query, top_k, data.get("episode_id"), data.get("provenance")


def make_handler(store: VectorFlyStore, *, token: str | None = None):
    """Bind a validated read-only index to an auditable JSON API."""
    if token is not None and (not isinstance(token, str) or len(token) < 24):
        raise ValueError("Remote access token must contain at least 24 characters")

    class AgentHandler(BaseHTTPRequestHandler):
        server_version = "VectorFly/1"
        sys_version = ""
        protocol_version = "HTTP/1.0"

        def log_message(self, format, *args):
            # Queries and source texts must not accidentally enter access logs.
            return

        def respond(self, status: int, payload: dict) -> None:
            raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                              allow_nan=False) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def fail(self, code: int, error: str) -> None:
            self.respond(code, {"api_version": API_VERSION, "error": error})

        def authorized(self) -> bool:
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, "Bearer " + token):
                self.fail(401, "Authorization required")
                return False
            return True

        def route(self):
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment:
                raise InputError("Query strings are not part of this API")
            return parsed.path

        def do_GET(self):
            if not self.authorized():
                return
            try:
                path = self.route()
                info = store.info()
                if path == "/v1/health":
                    self.respond(200, {
                        "api_version": API_VERSION, "status": "ok",
                        "scope": info["scope"], "read_only": True,
                    })
                elif path == "/v1/info":
                    self.respond(200, {
                        "api_version": API_VERSION, "database": info,
                        "limitations": [
                            "lexical feature similarity, not semantic entailment",
                            "all-scope search is not held-out evaluation",
                            "reconstructed fictional memories, not lived records",
                        ],
                    })
                elif path.startswith("/v1/memories/"):
                    requested = unquote(path[len("/v1/memories/"):])
                    if not EVENT_ID_PATTERN.fullmatch(requested):
                        raise InputError("Invalid event ID")
                    memory = store.get(requested)
                    if memory is None:
                        self.fail(404, "No source memory with that ID in this index")
                    else:
                        self.respond(200, {
                            "api_version": API_VERSION,
                            "scope": info["scope"],
                            "memory": memory,
                            "evidence_status": "authored source record, not verified entailment",
                        })
                else:
                    self.fail(404, "Unknown read-only endpoint")
            except InputError as exc:
                self.fail(400, str(exc))

        def do_POST(self):
            if not self.authorized():
                return
            try:
                path = self.route()
                if path != "/v1/search":
                    self.fail(405, "Only POST /v1/search is supported")
                    return
                typ = self.headers.get("Content-Type", "")
                if typ.split(";")[0].strip().lower() != "application/json":
                    self.fail(415, "Content-Type must be application/json")
                    return
                length = self.headers.get("Content-Length")
                if length is None or not length.isascii() or not length.isdecimal():
                    self.fail(411, "Valid Content-Length required")
                    return
                size = int(length)
                if size > MAX_REQUEST_BYTES:
                    self.fail(413, "JSON body too large")
                    return
                if size < 1:
                    raise InputError("Empty JSON body")
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                query, top_k, episode, provenance = _query_body(data)
                found = store.search(
                    query, top_k=top_k, episode_id=episode, provenance=provenance
                )
                info = store.info()
                self.respond(200, {
                    "api_version": API_VERSION,
                    "scope": info["scope"],
                    "source_git_blob": info["source_git_blob"],
                    "vector_kind": info["vector_kind"],
                    "query": query,
                    "count": len(found),
                    "results": found,
                    "evidence_status": "retrieval only; no factual entailment check",
                })
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.fail(400, "Invalid UTF-8 JSON")
            except (ValueError, InputError) as exc:
                self.fail(400, str(exc))

        def do_PUT(self):
            self.fail(405, "Memory database is read-only")

        do_PATCH = do_PUT
        do_DELETE = do_PUT

    return AgentHandler


def make_server(store: VectorFlyStore, *, host: str = "127.0.0.1",
                port: int = 8765, token: str | None = None) -> HTTPServer:
    """Refuse accidental unauthenticated remote exposure."""
    if not isinstance(host, str) or not host:
        raise ValueError("Valid listening host required")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not token:
        raise ValueError("Non-loopback binding requires an explicit bearer token")
    if type(port) is not int or port < 0 or port > 65535:
        raise ValueError("Invalid listening port")
    return HTTPServer((host, port), make_handler(store, token=token))
