import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import PROJECT


class PackageTests(unittest.TestCase):
    def test_cli_runs_from_an_independent_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "standalone"
            root.mkdir()
            for folder in ("data", "models", "training", "evaluation", "configs"):
                shutil.copytree(
                    PROJECT / folder,
                    root / folder,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            for name in ("run.py", "config.py", "__init__.py"):
                shutil.copy2(PROJECT / name, root / name)
            for command in ("download", "prepare", "train", "evaluate"):
                result = subprocess.run(
                    [sys.executable, "-I", "run.py", command, "--help"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import runpy; runpy.run_path('run.py'); from tszoo.config import load_training_config; c=load_training_config('configs/train_small.yaml'); assert c['train_end']==1913",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
