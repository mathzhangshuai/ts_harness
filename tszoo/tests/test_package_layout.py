import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import safetensors
import torch
import yaml
from fixtures import DUMMY, PROJECT
from tszoo.data import write_store
from tszoo.layers import MLP, Attention, ResidualBlock, RMSNorm


class PackageLayoutTests(unittest.TestCase):
    def test_standalone_copy_runs_without_parent_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "independent-project"
            root.mkdir()
            for folder in ("config", "data", "layers", "models", "utils"):
                shutil.copytree(
                    PROJECT / folder,
                    root / folder,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            for name in ("run.py", "__init__.py"):
                shutil.copy2(PROJECT / name, root / name)
            shutil.copytree(
                Path(safetensors.__file__).parent,
                root / ".experiment-deps/safetensors",
            )
            shutil.copytree(DUMMY, root / "weights")
            write_store(
                [{"item_id": "a", "start": "2020-01-01", "target": np.arange(14)}],
                root / "store",
                "D",
            )
            config = {
                "store": "../store",
                "output": "../out",
                "models": {"tiny": "../weights"},
                "context": 7,
                "origin": 10,
                "horizon": 3,
                "batch_size": 1,
                "device": "cpu",
                "quantiles": [0.1, 0.5, 0.9],
                "variate_attention": "grouped",
                "variate_attention_policy": "bidirectional",
            }
            (root / "config/test.yaml").write_text(yaml.safe_dump(config))
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "run.py",
                    "evaluate",
                    "--config",
                    "config/test.yaml",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((root / "out/metrics.json").read_text())
            self.assertEqual(metrics["models"]["tiny"]["observations"], 3)
            for name in ("chronos2", "evaluate_m5.py", "train.py", "fields.py"):
                self.assertFalse((PROJECT / name).exists(), name)

    def test_metrics_and_configuration_import_without_torch(self):
        code = (
            f"import runpy, sys; runpy.run_path({str(PROJECT / 'run.py')!r}); "
            "import tszoo.utils.metrics; import tszoo.utils.io; "
            "import tszoo.config.m5; assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_common_layers_need_no_model_config(self):
        x = torch.randn(2, 5, 8, requires_grad=True)
        norm = RMSNorm(8)
        mlp = MLP(8, 16)
        projection = ResidualBlock(8, 8, 16)
        for rotary in (False, True):
            attention = Attention(8, 4, 2, rotary=rotary)
            y = attention(projection(mlp(norm(x))), None)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.isfinite(y).all())
            y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
