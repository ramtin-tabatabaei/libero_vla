#!/usr/bin/env python3
"""Prepare a private MAX source overlay and extract its existing simulator archive."""

import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from libero_gui_catalog import ROOT, MAX_ROOT


def prepare():
    runtime = ROOT / ".max-runtime"
    assets = runtime / "plus/libero/assets"
    assets.mkdir(parents=True, exist_ok=True)
    source = MAX_ROOT / ".deps/LIBERO-plus/libero/libero"
    destination = assets.parent
    for item in source.iterdir():
        target = destination / item.name
        if item.name in ("__pycache__", "assets", ".DS_Store"):
            continue
        if item.is_file():
            shutil.copy2(item, target)
        elif item.name in ("envs", "benchmark"):
            shutil.copytree(item, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        elif not target.exists():
            target.symlink_to(item, target_is_directory=True)
    marker = runtime / "assets-ready.json"
    if marker.exists():
        print("LIBERO-Max simulator assets are ready.", flush=True)
        return
    archive = MAX_ROOT / ".deps/libero-plus-assets/assets.zip"
    with zipfile.ZipFile(archive) as z:
        entries = []
        for info in z.infolist():
            if "/assets/" not in info.filename or info.is_dir():
                continue
            relative = PurePosixPath(info.filename.split("/assets/", 1)[1])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe asset path")
            if relative.parts[0] == "new_objects" and "/usd/MJCF/" not in str(relative):
                continue  # Other CAD formats and authoring files are not used by MuJoCo.
            if relative.name == ".DS_Store" or "__MACOSX" in relative.parts:
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected symlink in simulation archive")
            entries.append((info, relative))
        print(f"Preparing {len(entries):,} simulator files ({sum(i.file_size for i, _ in entries)/1e9:.2f} GB) from the existing local archive.", flush=True)
        for n, (info, relative) in enumerate(entries, 1):
            target = assets.joinpath(*relative.parts)
            if not target.exists() or target.stat().st_size != info.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            if n % 5000 == 0:
                print(f"Prepared {n:,}/{len(entries):,} files", flush=True)
    marker.write_text(json.dumps({"archive": str(archive), "files": len(entries)}, indent=2))
    print("LIBERO-Max simulator assets are ready.", flush=True)


if __name__ == "__main__":
    prepare()
