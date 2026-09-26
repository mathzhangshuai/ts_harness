import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml
from fixtures import PROJECT

from tszoo.config import load_config
from tszoo.data import download as resources


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = b"local-test-payload"
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"{}" if "config.json" in self.path else cls.payload
                requested = self.headers.get("Range")
                cls.requests.append((self.path, requested))
                offset = int(requested.split("=")[1].split("-")[0]) if requested else 0
                if requested and not self.path.startswith("/ignore"):
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {offset}-{len(body) - 1}/{len(body)}"
                    )
                    body = body[offset:]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expected = hashlib.sha256(self.payload).hexdigest()
        self.requests.clear()

    def test_download_then_offline_reuse_and_corruption_rejection(self):
        path = self.root / "weights"
        self.assertTrue(
            resources.download_file(self.url + "/file", path, self.expected)
        )
        with patch.object(
            resources.requests, "get", side_effect=AssertionError("network")
        ):
            self.assertFalse(
                resources.download_file(self.url + "/file", path, self.expected)
            )
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                resources.download_file(self.url + "/file", path, self.expected)
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_resume_and_server_ignoring_range(self):
        for route in ("/resume", "/ignore"):
            path = self.root / route[1:]
            path.with_name(path.name + ".part").write_bytes(self.payload[:5])
            resources.download_file(
                self.url + route, path, self.expected, len(self.payload)
            )
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertEqual(self.requests[-1][1], "bytes=5-")
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_invalid_payload_is_never_published(self):
        path = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "failed validation"):
            resources.download_file(self.url + "/bad", path, "0" * 64)
        self.assertFalse(path.exists())
        self.assertTrue(path.with_name(path.name + ".part").exists())

    def test_m5_manifest_and_only_selected_checkpoint(self):
        (self.root / "data").mkdir()
        manifest = {
            "files": [
                {
                    "name": "calendar.csv",
                    "url": self.url + "/calendar",
                    "sha256": self.expected,
                    "bytes": len(self.payload),
                }
            ]
        }
        (self.root / "data/manifest.json").write_text(json.dumps(manifest))
        spec = {
            "repo_id": "example/selected",
            "revision": "a" * 40,
            "files": {
                "config.json": hashlib.sha256(b"{}").hexdigest(),
                "model.safetensors": self.expected,
            },
        }
        config = {
            "dataset": {"name": "m5", "path": str(self.root / "raw")},
            "models": {"selected": str(self.root / "selected")},
            "model_sources": {"selected": spec, "not_selected": spec},
            "store": str(self.root / "store"),
        }
        with (
            patch.object(resources, "ROOT", self.root),
            patch.object(resources, "MIRROR", self.url),
        ):
            result = resources.ensure_resources(config, prepare=False)
            self.assertEqual(result["dataset_files_downloaded"], 1)
            self.assertEqual(result["model_files_downloaded"], {"selected": 2})
            self.assertEqual(len(self.requests), 3)
            self.assertEqual(
                resources.ensure_resources(config, prepare=False)[
                    "dataset_files_downloaded"
                ],
                0,
            )
            self.assertEqual(len(self.requests), 3)
        self.assertFalse((self.root / "not_selected").exists())

    def test_strict_yaml_sources_and_paths(self):
        path = self.root / "run.yaml"
        raw = yaml.safe_load((PROJECT / "configs/baseline.yaml").read_text())
        path.write_text(yaml.safe_dump(raw))
        result = load_config(path)
        self.assertEqual(
            result["dataset"]["path"], str((self.root / "../datasets/m5").resolve())
        )
        self.assertEqual(
            result["model_sources"]["chronos2"]["repo_id"], "amazon/chronos-2"
        )
        for key, value in (("revision", "main"), ("repo_id", "../escape")):
            invalid = yaml.safe_load(path.read_text())
            invalid["models"]["chronos2"][key] = value
            path.write_text(yaml.safe_dump(invalid))
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(yaml.safe_dump(raw))
        raw["dataset"]["name"] = "other"
        path.write_text(yaml.safe_dump(raw))
        with self.assertRaises(ValueError):
            load_config(path)

    def test_target_store_is_prepared_without_static_features(self):
        import pandas as pd

        from tszoo.data import MemmapWindows

        source = self.root / "raw"
        source.mkdir()
        pd.DataFrame(
            {
                "d": [f"d_{i}" for i in range(1, 15)],
                "date": pd.date_range("2020-01-01", periods=14),
            }
        ).to_csv(source / "calendar.csv", index=False)
        row = {
            "id": "a",
            "item_id": "a",
            "dept_id": "d",
            "cat_id": "c",
            "store_id": "s",
            "state_id": "x",
            **{f"d_{i}": i for i in range(1, 15)},
        }
        pd.DataFrame([row]).to_csv(source / "sales_train_evaluation.csv", index=False)
        store = self.root / "store"
        self.assertTrue(resources.prepare_store(source, store, reuse=True))
        self.assertFalse(resources.prepare_store(source, store, reuse=True))
        manifest = json.loads((store / "manifest.json").read_text())
        self.assertEqual(set(manifest["fields"]), {"target"})
        data = MemmapWindows(store, 7, 3, mode="predict", end=10)
        self.assertEqual(data[0]["target"].shape, (1, 7))
        data.close()


if __name__ == "__main__":
    unittest.main()
