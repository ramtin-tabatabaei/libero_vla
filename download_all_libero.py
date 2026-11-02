"""Resume the official raw LIBERO demonstrations and verify their inventory."""
from pathlib import Path
import shutil
from collections import Counter
from huggingface_hub import HfApi, snapshot_download
import h5py

repo = "yifengzhu-hf/LIBERO-datasets"
destination = Path(__file__).resolve().parent / "LIBERO/libero/datasets"
expected = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
info = HfApi().dataset_info(repo, files_metadata=True)
files = [f for f in info.siblings if f.rfilename.endswith(".hdf5")]
counts = Counter(f.rfilename.split("/")[0] for f in files)
if dict(counts) != expected:
    raise RuntimeError(f"Unexpected upstream inventory: {counts}")
remaining = sum(f.size for f in files if not (destination / f.rfilename).exists() or (destination / f.rfilename).stat().st_size != f.size)
free = shutil.disk_usage(destination.parent).free
print(f"Revision: {info.sha}; files: {counts}; download upper bound: {remaining / 2**30:.1f} GiB; free: {free / 2**30:.1f} GiB", flush=True)
if remaining + 5 * 2**30 > free:
    raise RuntimeError("Insufficient free space, including 5 GiB reserve")
snapshot_download(repo, repo_type="dataset", revision=info.sha, local_dir=destination,
                  allow_patterns=[f"{suite}/*.hdf5" for suite in expected] + ["README.md"], max_workers=3)
episodes = Counter()
for entry in files:
    path = destination / entry.rfilename
    if path.stat().st_size != entry.size:
        raise RuntimeError(f"Size mismatch: {path}")
    with h5py.File(path, "r") as data:
        demos = data["data"]
        episodes[entry.rfilename.split("/")[0]] += len(demos)
        for demo in demos.values():
            if demo["actions"].shape[0] == 0:
                raise RuntimeError(f"Empty trajectory: {path}/{demo.name}")
print(f"VERIFIED {len(files)} HDF5 files; episodes by suite: {dict(episodes)}", flush=True)
print(f"Dataset directory: {destination}", flush=True)
