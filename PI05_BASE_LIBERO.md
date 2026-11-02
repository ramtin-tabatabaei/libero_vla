# Experimental pi05_base weights in LIBERO

The server loads the official `gs://openpi-assets/checkpoints/pi05_base`
checkpoint, not pi05_libero weights. No training is performed. It retains
`Pi0Config(pi05=True)` defaults: 32 action dimensions, 50-step horizon and
discrete state input. The client replans every five actions.

This is an embodiment-transfer experiment, not a validated zero-shot policy.
It uses OpenPI's LIBERO input/output adapter and normalization from
`pi05_libero/assets/physical-intelligence/libero`. Therefore “without LIBERO”
means without LIBERO-fine-tuned weights, NOT without any LIBERO-specific data
or processing. Absence of LIBERO from pretraining has not been established.
Arbitrary text instructions do not guarantee compliant behavior.

Finish other GPU policy servers before starting this one. On this 31 GiB RAM
machine with no swap, checkpoint loading may still encounter host-memory limits.

Terminal 1:
```bash
cd /home/ramtin/libero_vla
bash serve_pi05_base.sh
```

Terminal 2, once the server is listening:
```bash
cd /home/ramtin/libero_vla
bash run_pi05_base_libero.sh \
  --prompt "Grasp the wine bottle. Put it on the bowl." \
  --shuffle_wine --seed 7
```

The client defaults to goal task 2 (wine on cabinet); its reported success
still measures the original cabinet goal unless `--stop_on_grasp` is enabled.
Client flags may override the defaults. Metadata records the use of base
weights and LIBERO normalization.

## Force a drop and watch recovery

With the policy server running:
```bash
bash run_pi05_base_libero.sh --drop_object \
  --drop_delay_steps 10 --drop_open_steps 10
```

Once per episode, two consecutive control steps of two-finger contact with
any movable object arm the intervention. After 10 more normal control steps,
the gripper is forced open for 10 steps. Arm motion and policy inference
continue throughout; afterward the policy controls the gripper again.
Increase the delay to allow more time to lift the object. Contact alone does
not guarantee the object is lifted, and a drop is not guaranteed if it is
still supported. If no grasp occurs, no intervention occurs.

The existing live view and saved video show the outcome. The terminal logs
`forced drop` events for opening and resuming, and the results JSON records
the settings. Normal task success and time limits still apply. This option
cannot be combined with `--stop_on_grasp`. Omit `--drop_object` for normal runs.

Check metadata/configuration without loading weights:
```bash
bash serve_pi05_base.sh --check-config
```

Explicit model load and synthetic inference test:
```bash
bash serve_pi05_base.sh --smoke-test
```
