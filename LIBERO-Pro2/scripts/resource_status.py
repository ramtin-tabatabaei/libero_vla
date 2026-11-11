#!/usr/bin/env python3
"""Show local download progress without contacting remote services."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def show(label, directory, manifest_name, marker_name):
    manifest = ROOT / "resource-manifests" / manifest_name
    marker = ROOT / "resource-manifests" / marker_name
    if not manifest.exists():
        print(f"{label}: inventory not fetched yet")
        return
    entries = json.loads(manifest.read_text())["files"]
    expected = sum(int(e["size"]) for e in entries)
    present = 0
    finished = 0
    for entry in entries:
        relative = entry.get("path") or entry["name"].removeprefix("checkpoints/pi05_libero/")
        path = ROOT / directory / relative
        if path.exists():
            size = path.stat().st_size
            present += min(size, int(entry["size"]))
            finished += size == int(entry["size"])
        elif path.with_name(path.name + ".partial").exists():
            present += min(path.with_name(path.name + ".partial").stat().st_size, int(entry["size"]))
    status = "COMPLETE, checksums verified" if marker.exists() else "INCOMPLETE"
    if manifest_name == "dataset.json" and (ROOT / "resource-manifests/training-download-stopped.txt").exists():
        status = "STOPPED; partial files retained"
    print(f"{label}: {status}; {present / 1e9:.2f}/{expected / 1e9:.2f} GB, "
          f"{finished}/{len(entries)} files at expected size")


show("LIBERO-Pro2 metadata", "LIBERO-Pro2.0-Train", "metadata.json", "metadata-complete.json")
show("LIBERO-Pro2 training", "LIBERO-Pro2.0-Train", "dataset.json", "dataset-complete.json")
show("pi05 LIBERO", "models/pi05_libero", "pi05_libero.json", "model-complete.json")
print("Log:", ROOT / "download-resources.log")
