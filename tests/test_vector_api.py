"""HTTP contract tests against the actual frozen 450-memory Vector Fly source."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import threading
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.shared_memory_l2 import build_cache
from pretorius_connectome.vector_store import build_store, VectorFlyStore
from pretorius_connectome.vector_api import make_server, MAX_REQUEST_BYTES


class VectorFlyAgentAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.cache = root / "l2-seed31"
        build_cache(cls.cache, seed=31)
        cls.db_path = root / "db-all.sqlite"
        cls.train_db_path = root / "db-train.sqlite"
        build_store(cls.db_path, cls.cache, scope="all")
        build_store(cls.train_db_path, cls.cache, scope="train")
        cls.index = VectorFlyStore(cls.db_path, cls.cache)
        cls.server = make_server(cls.index, host="127.0.0.1", port=0)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()
        cls.index.close()
        cls.temp.cleanup()

    def request(self, method, path, payload=None, *, content_type="application/json",
                token=None, port=None):
        conn = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=5)
        headers = {}
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = content_type
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), json.loads(raw)
        finally:
            conn.close()

    def test_health_info_and_source_capabilities(self):
        status, headers, result = self.request("GET", "/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["scope"], "all")
        self.assertTrue(result["read_only"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, _, info = self.request("GET", "/v1/info")
        self.assertEqual(status, 200)
        self.assertEqual(info["database"]["indexed_documents"], 450)
        self.assertEqual(info["database"]["vector_dim"], 8192)
        self.assertIn("lexical", info["limitations"][0])

    def test_search_uses_exact_source_ids_and_score_calibration(self):
        query = "camphor beetle brass key"
        status, _, data = self.request("POST", "/v1/search",
                                       {"query": query, "top_k": 5})
        self.assertEqual(status, 200)
        self.assertEqual(data["api_version"], "vector-fly-retrieval/1")
        self.assertEqual(data["results"], self.index.search(query, top_k=5))
        self.assertEqual(data["results"][0]["event_id"], "E01-001")
        self.assertEqual(data["results"][0]["provenance"], "reconstructed")
        self.assertIn("retrieval only", data["evidence_status"])
        status, _, unknown = self.request("POST", "/v1/search", {
            "query": "zzzzqzxzy unrecoverableword12345"
        })
        self.assertEqual(status, 200)
        self.assertEqual(unknown["results"], [])
        self.assertEqual(unknown["count"], 0)

    def test_record_lookup_and_episode_provenance_filtering(self):
        original = self.index.get("E01-001")
        status, _, response = self.request("GET", "/v1/memories/E01-001")
        self.assertEqual(status, 200)
        self.assertEqual(response["memory"], original)
        self.assertEqual(response["memory"]["provenance"], "reconstructed")
        status, _, missing = self.request("GET", "/v1/memories/E99-999")
        self.assertEqual(status, 404)
        self.assertIn("No source memory", missing["error"])
        status, _, filtered = self.request("POST", "/v1/search", {
            "query": original["memory_text"][:150], "episode_id": "E01",
            "provenance": "reconstructed", "top_k": 5
        })
        self.assertEqual(status, 200)
        self.assertTrue(filtered["results"])
        self.assertTrue(all(x["episode_id"] == "E01"
                            for x in filtered["results"]))
        status, _, impossible = self.request("POST", "/v1/search", {
            "query": "camphor key", "provenance": "verified-lived"
        })
        self.assertEqual(status, 200)
        self.assertEqual(impossible["results"], [])

    def test_bad_and_unsafe_requests_fail_closed(self):
        for payload in (
            {"query": ""},
            {"query": "x" * 2001},
            {"query": "laboratory", "top_k": True},
            {"query": "laboratory", "top_k": 21},
            {"query": "laboratory", "unexpected": "ignore me"},
            {"query": "laboratory", "episode_id": []},
            ["laboratory"],
        ):
            status, _, body = self.request("POST", "/v1/search", payload)
            self.assertEqual(status, 400, payload)
            self.assertIn("error", body)
        status, _, _ = self.request(
            "POST", "/v1/search", {"query": "laboratory"},
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        status, _, _ = self.request("POST", "/v1/nonexistent", {"query": "x"})
        self.assertEqual(status, 405)
        status, _, _ = self.request("GET", "/v1/memories/../../etc/passwd")
        self.assertIn(status, (400, 404))
        status, _, _ = self.request("GET", "/v1/health?leak=true")
        self.assertEqual(status, 400)
        for method in ("DELETE", "PUT", "PATCH"):
            status, _, response = self.request(method, "/v1/memories/E01-001")
            self.assertEqual(status, 405)
            self.assertIn("read-only", response["error"])

    def test_oversized_body_rejected_before_json_decode(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/v1/search", body=b"{}",
                         headers={"Content-Type": "application/json",
                                  "Content-Length": str(MAX_REQUEST_BYTES + 1)})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            self.assertEqual(resp.status, 413)
            self.assertIn("too large", data["error"])
        finally:
            conn.close()

    def test_remote_bind_requires_auth_and_token_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit bearer"):
            make_server(self.index, host="0.0.0.0", port=0)
        with self.assertRaisesRegex(ValueError, "24 characters"):
            make_server(self.index, host="127.0.0.1", port=0, token="weak")
        secret = "test-an-example-very-long-local-token-1234"
        auth_server = make_server(
            self.index, host="127.0.0.1", port=0, token=secret
        )
        worker = threading.Thread(
            target=auth_server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        worker.start()
        port = auth_server.server_address[1]
        try:
            status, _, _ = self.request("GET", "/v1/info", port=port)
            self.assertEqual(status, 401)
            status, _, _ = self.request("GET", "/v1/info", port=port,
                                        token="incorrect-local-token")
            self.assertEqual(status, 401)
            status, _, result = self.request("GET", "/v1/info", port=port,
                                             token=secret)
            self.assertEqual(status, 200)
            self.assertEqual(result["database"]["indexed_documents"], 450)
        finally:
            auth_server.shutdown()
            worker.join(timeout=5)
            auth_server.server_close()

    def test_restricted_database_does_not_expose_held_out_records(self):
        with VectorFlyStore(self.train_db_path, self.cache) as source:
            held_out = source.cache.manifest["split_event_ids"]["test"][0]
            training = source.cache.manifest["fit_event_ids"][0]
            restricted_server = make_server(source, port=0)
            worker = threading.Thread(
                target=restricted_server.serve_forever,
                kwargs={"poll_interval": 0.01}, daemon=True
            )
            worker.start()
            port = restricted_server.server_address[1]
            try:
                status, _, absent = self.request(
                    "GET", "/v1/memories/" + held_out, port=port
                )
                self.assertEqual(status, 404)
                status, _, known = self.request(
                    "GET", "/v1/memories/" + training, port=port
                )
                self.assertEqual(status, 200)
                self.assertEqual(known["scope"], "train")
                status, _, r = self.request("GET", "/v1/info", port=port)
                self.assertEqual(status, 200)
                self.assertEqual(r["database"]["indexed_documents"], 317)
            finally:
                restricted_server.shutdown()
                worker.join(timeout=5)
                restricted_server.server_close()


if __name__ == "__main__":
    unittest.main()
