"""Check 4: raw observation -> openpi transform -> model input.

Builds the exact inference transform stack pi05_libero uses (minus the weights)
and pushes two observations through it that differ only in end-effector
position. If the raw state changes but the model input does not, the state is
being dropped somewhere in the pipeline.

    JAX_PLATFORMS=cpu /home/ramtin/openpi/.venv/bin/python check_transform.py
"""

import pathlib

import numpy as np

import openpi.transforms as transforms
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero"
LABELS = ["x", "y", "z", "ax", "ay", "az", "grip_l", "grip_r"]


def main():
    cfg = _config.get_config("pi05_libero")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(CKPT / "assets", data_config.asset_id)

    print(f"asset_id: {data_config.asset_id}   use_quantile_norm: {data_config.use_quantile_norm}")
    print(f"model: pi05={cfg.model.pi05} discrete_state_input={cfg.model.discrete_state_input}\n")

    ns = norm_stats["state"]
    print("state norm stats")
    for i, l in enumerate(LABELS):
        extra = ""
        if data_config.use_quantile_norm:
            lo, hi = float(ns.q01[i]), float(ns.q99[i])
            extra = f"  q01={lo:+.4f} q99={hi:+.4f} width={hi-lo:.5f}"
            if hi - lo < 1e-3:
                extra += "   <-- DEGENERATE"
        else:
            extra = f"  mean={float(ns.mean[i]):+.4f} std={float(ns.std[i]):.5f}"
            if float(ns.std[i]) < 1e-3:
                extra += "   <-- DEGENERATE"
        print(f"  [{i}] {l:7s}{extra}")

    chain = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    print("\ntransform chain:")
    for t in chain:
        print(f"  {type(t).__name__}")

    rng = np.random.default_rng(0)
    img = rng.integers(256, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(256, size=(224, 224, 3), dtype=np.uint8)

    def build(state):
        return {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": np.asarray(state, dtype=np.float64),
            "prompt": "pick up the butter and place it in the basket",
        }

    raw_a = np.array([-0.1526, -0.0063, 0.2484, 3.1406, -0.0015, -0.0914, 0.0387, -0.0387])
    raw_b = raw_a.copy()
    raw_b[:3] += 0.10  # move the end-effector 10 cm in every axis

    def push(data):
        for t in chain:
            data = t(data)
        return data

    out_a, out_b = push(build(raw_a)), push(build(raw_b))

    print("\n" + "=" * 72)
    print("raw state       A:", np.round(raw_a, 4))
    print("raw state       B:", np.round(raw_b, 4))
    print("raw delta        :", np.round(raw_b - raw_a, 4))
    print()
    sa, sb = np.asarray(out_a["state"]), np.asarray(out_b["state"])
    print(f"model input 'state' shape {sa.shape} dtype {sa.dtype}")
    print("model input     A:", np.round(sa, 4))
    print("model input     B:", np.round(sb, 4))
    print("model delta      :", np.round(sb - sa, 4))
    print()
    changed = np.abs(sb - sa).max()
    print(f"max |model input delta| = {changed:.6f}")
    print("VERDICT:", "state reaches the model" if changed > 1e-6 else "STATE IS BEING DROPPED")

    keys_a = sorted(out_a)
    print(f"\nmodel input keys: {keys_a}")
    for k in keys_a:
        v = out_a[k]
        if isinstance(v, np.ndarray):
            print(f"  {k:24s} {v.shape} {v.dtype}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                shape = getattr(vv, "shape", type(vv).__name__)
                print(f"  {k}/{kk:18s} {shape}")
        else:
            print(f"  {k:24s} {type(v).__name__}")


if __name__ == "__main__":
    main()
