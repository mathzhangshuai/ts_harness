"""Run tszoo directly from this standalone directory, without installation."""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
local_dependencies = ROOT / ".experiment-deps"
if local_dependencies.is_dir():
    sys.path.insert(0, str(local_dependencies))
spec = importlib.util.spec_from_file_location(
    "tszoo", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules["tszoo"] = package
spec.loader.exec_module(package)

COMMANDS = {
    "baseline": "utils.baseline_m5",
    "download": "utils.resources",
    "prepare": "data.prepare",
    "evaluate": "utils.evaluate_m5",
    "reference": "utils.evaluate_m5_reference",
    "wrmsse": "utils.score_m5_wrmsse",
    "train": "utils.train",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "test"])
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    if args.command == "test":
        import unittest

        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())
    sys.argv = [f"run.py {args.command}", *remaining]
    importlib.import_module("tszoo." + COMMANDS[args.command]).main()


if __name__ == "__main__":
    main()
