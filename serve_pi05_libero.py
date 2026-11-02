"""Serve the standard OpenPI Pi 0.5 LIBERO checkpoint with explicit identity."""
import argparse
import dataclasses
import json
import logging
from pathlib import Path

MODEL_ID = "pi05_libero"
CHECKPOINT = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero"
ASSET_ID = "physical-intelligence/libero"
ACTION_HORIZON = 10
NUM_STEPS = 10
REPLAN_STEPS = 5


def metadata():
    """The contract shared by the server and the recovery experiment."""
    return dict(model_id=MODEL_ID, weights=MODEL_ID, training="supervised",
                rl_finetuned=False, action_horizon=ACTION_HORIZON,
                num_steps=NUM_STEPS, replan_steps=REPLAN_STEPS,
                discrete_state_input=False)


def configuration(checkpoint=CHECKPOINT):
    from openpi.training.config import get_config

    checkpoint = Path(checkpoint).resolve()
    # This launcher loads the cached standard JAX checkpoint, never the RLinf
    # PyTorch snapshot used by the separate RL server.
    if not (checkpoint / "params").is_dir() or (checkpoint / "model.safetensors").exists():
        raise ValueError("Expected the standard OpenPI pi05_libero JAX checkpoint")
    if not (checkpoint / "assets" / ASSET_ID / "norm_stats.json").is_file():
        raise FileNotFoundError("Missing pi05_libero normalization in " + str(checkpoint))
    cfg = get_config(MODEL_ID)
    identity = metadata()
    identity["checkpoint"] = str(checkpoint)
    return dataclasses.replace(cfg, policy_metadata=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate the cached checkpoint without loading model weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    cfg = configuration()
    print(json.dumps(cfg.policy_metadata, indent=2), flush=True)
    if args.check_config:
        return
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = create_trained_policy(cfg, CHECKPOINT, sample_kwargs={"num_steps": NUM_STEPS})
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
