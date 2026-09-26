import tempfile
import unittest
from pathlib import Path

import yaml
from fixtures import PROJECT

from tszoo.config import load_config, load_training_config


class ConfigurationTests(unittest.TestCase):
    def test_yaml_paths_and_finetuned_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = yaml.safe_load((PROJECT / "configs/baseline.yaml").read_text())
            raw["models"] = {"trained": {"path": "checkpoint", "format": "finetuned"}}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(raw))
            config = load_config(path)
            self.assertEqual(config["models"]["trained"], str(root / "checkpoint"))
            self.assertEqual(config["model_formats"]["trained"], "finetuned")
            self.assertEqual(config["model_sources"], {})

    def test_duplicate_unknown_and_invalid_training_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text("origin: 1913\norigin: 1941\n")
            with self.assertRaises(ValueError):
                load_config(path)
            raw = yaml.safe_load((PROJECT / "configs/train_small.yaml").read_text())
            for section, key, value in (
                ("data", "fields", ["target", "feat_dynamic_real"]),
                ("training", "epochs", True),
                ("data", "train_days", True),
                ("data", "train_days", 0),
                ("training", "epochs", 0),
                ("training", "steps", 1000),
                ("training", "devices", 0),
                ("training", "devices", True),
                ("training", "shuffle_block_size", 0),
                ("training", "shuffle_block_size", True),
                ("training", "strategy", "invalid"),
                ("training", "precision", "invalid"),
                ("model", "attention", {"variate_attention_policy": "invalid"}),
                ("training", "lr", float("nan")),
            ):
                invalid = yaml.safe_load(yaml.safe_dump(raw))
                invalid[section][key] = value
                path.write_text(yaml.safe_dump(invalid))
                with self.assertRaises(ValueError):
                    load_training_config(path)
