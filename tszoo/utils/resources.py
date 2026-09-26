"""Download and verify M5 and the local checkpoints selected by YAML."""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MIRROR = "https://hf-mirror.com"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, path, expected, size=None):
    """Reuse verified files; resume partial transfers and publish only valid bytes."""
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("A lowercase SHA-256 is required")
    if path.exists():
        if (size is not None and path.stat().st_size != size) or sha256(
            path
        ) != expected:
            raise ValueError(
                f"Existing file failed validation; refusing overwrite: {path}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if (
        partial.exists()
        and sha256(partial) == expected
        and (size is None or partial.stat().st_size == size)
    ):
        partial.rename(path)
        return True
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if size is not None and offset >= size:
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=(20, 90)
            ) as response:
                if response.status_code == 416:
                    partial.write_bytes(b"")
                    continue
                response.raise_for_status()
                append = response.status_code == 206 and offset > 0
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected Content-Range in download")
                with partial.open("ab" if append else "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if (size is not None and partial.stat().st_size != size) or sha256(
                partial
            ) != expected:
                raise ValueError(f"Downloaded file failed validation: {path.name}")
            partial.rename(path)
            return True
        except requests.RequestException:
            if attempt == 2:
                raise
    raise RuntimeError(f"Could not download {path}")


def ensure_m5(directory):
    manifest = json.loads((ROOT / "data/m5_manifest.json").read_text(encoding="utf-8"))
    downloaded = 0
    for entry in manifest["files"]:
        downloaded += download_file(
            entry["url"],
            Path(directory) / entry["name"],
            entry["sha256"],
            entry["bytes"],
        )
    return downloaded


def ensure_checkpoint(path, source=None):
    path = Path(path)
    if source is None:
        if not (path / "config.json").is_file() or not (
            (path / "model.safetensors").is_file()
            or (path / "model.safetensors.index.json").is_file()
        ):
            raise FileNotFoundError(
                f"Local checkpoint absent; specify its download source in YAML: {path}"
            )
        return 0
    downloaded = 0
    for name, expected in source["files"].items():
        url = f"{MIRROR}/{source['repo_id']}/resolve/{source['revision']}/{name}?download=true"
        downloaded += download_file(url, path / name, expected)
    return downloaded


def prepare_target_store(source, output):
    from ..data.prepare import Categories, m5_entries
    from ..data.storage import write_store

    output = Path(output)
    if output.exists():
        if not (output / "manifest.json").is_file():
            raise ValueError(f"Incomplete processed store: {output}")
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and publish the complete store atomically.
    with tempfile.TemporaryDirectory(
        prefix=".m5-build-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "store"
        entries = (
            {key: entry[key] for key in ("item_id", "start", "target")}
            for entry in m5_entries(source, Categories())
        )
        write_store(entries, staging, "D")
        staging.rename(output)
    return True


def ensure_resources(config, prepare=True):
    dataset = config.get("dataset")
    dataset_downloads = ensure_m5(dataset["path"]) if dataset else 0
    model_downloads = {}
    for name, path in config["models"].items():
        model_downloads[name] = ensure_checkpoint(
            path, config.get("model_sources", {}).get(name)
        )
    prepared = False
    if prepare and not Path(config["store"]).exists():
        if not dataset:
            raise FileNotFoundError(
                "Processed store absent and no M5 dataset configured"
            )
        if config.get("fields", ["target"]) != ["target"]:
            raise ValueError(
                "Automatic preparation supports target-only M5; prepare covariates explicitly"
            )
        prepared = prepare_target_store(dataset["path"], config["store"])
    result = {
        "dataset_files_downloaded": dataset_downloads,
        "model_files_downloaded": model_downloads,
        "store_prepared": prepared,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    from ..config.m5 import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/m5-zero-shot.yaml"))
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.models:
        unknown = set(args.models) - set(config["models"])
        if unknown:
            parser.error(f"Unknown models: {sorted(unknown)}")
        config["models"] = {name: config["models"][name] for name in args.models}
    ensure_resources(config, prepare=False)


if __name__ == "__main__":
    main()
