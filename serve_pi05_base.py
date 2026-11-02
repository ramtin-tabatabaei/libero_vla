"""Serve frozen pi05_base weights through an experimental LIBERO interface.

No fine-tuned weights are loaded. LIBERO normalization is explicitly reused;
this is not a claim of dataset-free or verified zero-shot performance.
"""
import argparse
import dataclasses
import logging
from pathlib import Path

from openpi.models.pi0_config import Pi0Config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.checkpoints import load_norm_stats
from openpi.training.config import get_config

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "checkpoints/openpi/openpi-assets/checkpoints/pi05_base"
STATS = Path("/home/ramtin/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets")


def configuration():
    return dataclasses.replace(
        get_config("pi05_libero"),
        name="pi05_base_libero_interface",
        # Preserve the base model's native discrete state input and horizon.
        model=Pi0Config(pi05=True),
        policy_metadata={
            "weights": "pi05_base",
            "libero_finetuned_weights": False,
            "normalization": "pi05_libero/assets/physical-intelligence/libero",
            "experimental_base_transfer": True,
            "action_horizon": 50,
            "discrete_state_input": True,
        },
    )


def load_policy(cfg, stats):
    """Use bounded Orbax reads instead of the unbounded default restore."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import traverse_util
    from openpi.policies.policy import Policy
    from openpi import transforms

    devices = jax.devices("gpu")
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    # The largest source tensor exceeds 2 GB; smaller limits cannot restore it.
    logging.info("Restoring BF16 weights on %s with 4 GB concurrent reads", devices[0])
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler(restore_concurrent_gb=4)) as checkpointer:
        metadata = checkpointer.metadata(BASE / "params")
        item = {"params": metadata["params"]}
        restored = checkpointer.restore(
            BASE / "params", args=ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=jax.Array, dtype=jnp.bfloat16), item),
            ),
        )["params"]
    flat = traverse_util.flatten_dict(restored)
    if all(key[-1] == "value" for key in flat):
        restored = traverse_util.unflatten_dict({key[:-1]: value for key, value in flat.items()})
    model = cfg.model.load(restored)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    return Policy(model,
                  transforms=[*data.data_transforms.inputs,
                              transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
                              *data.model_transforms.inputs],
                  output_transforms=[*data.model_transforms.outputs,
                                     transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
                                     *data.data_transforms.outputs],
                  metadata=cfg.policy_metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Load and infer once, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    cfg = configuration()
    stats = load_norm_stats(STATS, "physical-intelligence/libero")
    if stats is None or not {"state", "actions"} <= stats.keys():
        raise RuntimeError("LIBERO normalization statistics are missing")
    if not (BASE / "params").is_dir():
        raise FileNotFoundError(f"Base checkpoint not downloaded: {BASE}")
    logging.info("Checkpoint: %s; native base configuration: %s", BASE, cfg.model)
    logging.warning("EXPERIMENTAL: base weights with LIBERO normalization; not a validated zero-shot policy")
    if args.check_config:
        return
    policy = load_policy(cfg, stats)
    if args.smoke_test:
        import numpy as np
        from openpi.policies.libero_policy import make_libero_example
        result = policy.infer(make_libero_example())
        actions = np.asarray(result["actions"])
        assert actions.shape == (50, 7) and np.isfinite(actions).all(), actions.shape
        print("Base checkpoint loaded; finite (50, 7) action output verified.")
        return
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
