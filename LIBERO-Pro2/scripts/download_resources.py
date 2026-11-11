#!/usr/bin/env python3
"""Download LIBERO-Pro2 metadata and the official pi05 LIBERO checkpoint.

Run with .hf-env/bin/python scripts/download_resources.py [metadata|model|all].
Re-running resumes interrupted downloads. All caches stay in this workspace.
Full trajectories require the explicit dataset --include-trajectories command.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
DATASET = "zhouxueyang/LIBERO-Pro2.0-Train"
REVISION = "6c4c5524676815a88880a7025badd17fb724f7f4"
PATTERNS = ["README.md", "data/**", "docs/**", "manifests/**", "release.json", "scripts/**"]
METADATA_PATTERNS = ["README.md", "release.json", "docs/**", "manifests/**", "scripts/**",
                     "data/**/metadata.json", "data/**/source_index.json"]


def report(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h


def dataset(metadata_only=False):
    import fnmatch
    from huggingface_hub import HfApi, snapshot_download

    destination = ROOT / "LIBERO-Pro2.0-Train"
    patterns = METADATA_PATTERNS if metadata_only else PATTERNS
    label = "Metadata" if metadata_only else "Dataset"
    manifest_name = "metadata" if metadata_only else "dataset"
    cached = ROOT / "resource-manifests/dataset.json"
    if cached.exists() and json.loads(cached.read_text())["revision"] == REVISION:
        candidates = json.loads(cached.read_text())["files"]
    else:
        info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
        candidates = [{"path": f.rfilename, "size": f.size,
                       "sha256": f.lfs.sha256 if f.lfs else None} for f in info.siblings]
    inventory = [f for f in candidates if any(fnmatch.fnmatch(f["path"], p) for p in patterns)]
    save_json(ROOT / f"resource-manifests/{manifest_name}.json",
              {"repository": DATASET, "revision": REVISION, "files": inventory})
    report(f"{label}: downloading {len(inventory)} files, {sum(f['size'] for f in inventory):,} bytes")
    snapshot_download(DATASET, repo_type="dataset", revision=REVISION,
                      allow_patterns=patterns, local_dir=destination, max_workers=8 if metadata_only else 4)
    report(f"{label}: checking file sizes and SHA-256 hashes")
    for entry in inventory:
        path = destination / entry["path"]
        if path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Dataset size mismatch: {path}")
        if entry["sha256"] and digest(path, "sha256").hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Dataset SHA-256 mismatch: {path}")
    save_json(ROOT / f"resource-manifests/{manifest_name}-complete.json",
              {"revision": REVISION, "verified_files": len(inventory)})
    report(f"{label}: download and verification complete")


def model():
    import google_crc32c

    prefix = "checkpoints/pi05_libero/"
    manifest = ROOT / "resource-manifests/pi05_libero.json"
    if manifest.exists():
        inventory = json.loads(manifest.read_text())["files"]
    else:
        url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(
            {"prefix": prefix, "maxResults": 1000})
        with urllib.request.urlopen(url, timeout=60) as response:
            listing = json.load(response)
        if listing.get("nextPageToken"):
            raise RuntimeError("Unexpected paginated model inventory")
        inventory = listing["items"]
        if not inventory:
            raise RuntimeError("Empty checkpoint inventory")
        save_json(manifest, {"source": "gs://openpi-assets/" + prefix, "files": inventory})
    destination = ROOT / "models/pi05_libero"
    report(f"Model: downloading {len(inventory)} files, {sum(int(f['size']) for f in inventory):,} bytes")

    def fetch(entry):
        relative = entry["name"].removeprefix(prefix)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        size = int(entry["size"])
        expected = entry["crc32c"]

        def valid(candidate):
            if not candidate.exists() or candidate.stat().st_size != size:
                return False
            checksum = google_crc32c.Checksum()
            with candidate.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    checksum.update(chunk)
            return base64.b64encode(checksum.digest()).decode() == expected

        if valid(path):
            report(f"Model: already verified {relative}")
            return
        partial = path.with_name(path.name + ".partial")
        url = ("https://storage.googleapis.com/openpi-assets/"
               + urllib.parse.quote(entry["name"], safe="/")
               + "?generation=" + entry["generation"])
        if not valid(partial):
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "10", "--retry-delay", "5", "--connect-timeout", "30",
                            "--speed-limit", "1024", "--speed-time", "120",
                            "--continue-at", "-", "--output", str(partial), url], check=True)
        if not valid(partial):
            raise RuntimeError(f"Model checksum mismatch: {partial}")
        partial.replace(path)
        report(f"Model: verified {relative} ({size:,} bytes)")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, inventory))
    save_json(ROOT / "resource-manifests/model-complete.json",
              {"source": "gs://openpi-assets/checkpoints/pi05_libero", "verified_files": len(inventory)})
    report("Model: download and verification complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", choices=["all", "dataset", "metadata", "model"], default="metadata", nargs="?")
    parser.add_argument("--include-trajectories", action="store_true",
                        help="Explicitly authorize the 234 GB full training download")
    args = parser.parse_args()
    if args.resource == "all":
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(dataset, metadata_only=True), pool.submit(model)]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as error:
                    report(f"Download failed: {type(error).__name__}: {error}")
                    raise
    elif args.resource == "dataset":
        if not args.include_trajectories:
            parser.error("Full training downloads are disabled; use metadata, or explicitly pass --include-trajectories")
        dataset()
    elif args.resource == "metadata":
        dataset(metadata_only=True)
    else:
        model()
