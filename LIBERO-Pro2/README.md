# LIBERO-Pro2.0 and π0.5-LIBERO workspace

To push the source and continue on another computer, follow [MIGRATION.md](MIGRATION.md).

This workspace contains the LIBERO-Pro2.0 evaluation release, training metadata,
and the official OpenPI π0.5 checkpoint fine-tuned on LIBERO.
**The full training download was stopped at the user's request.** The selected
dataset download includes only metadata and documentation (about 36 MB), excluding
saved trajectories, preview images and videos. Model weights download separately.
Check download status before using either resource.

| Resource | Local directory | Source |
| --- | --- | --- |
| Evaluation release archives and manifests | `LIBERO-Pro2.0-Eval/` | [LIBERO-Pro2.0-Eval](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Eval) |
| Extracted simulator and evaluation service | `libero-pro-install/LIBERO-Pro-evaluation/` | Evaluation release |
| Training metadata and source indices, about 36 MB | `LIBERO-Pro2.0-Train/` | Training repository, metadata only |
| Training demonstrations, STOPPED with partial files retained | `LIBERO-Pro2.0-Train/` | [LIBERO-Pro2.0-Train](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Train) |
| π0.5-LIBERO JAX checkpoint, about 12.4 GB | `models/pi05_libero/` | `gs://openpi-assets/checkpoints/pi05_libero` |
| Official model implementation | `openpi/` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |

Check download progress from this directory:

```bash
.hf-env/bin/python scripts/resource_status.py
tail -f download-resources.log
tail -f download-metadata.log
```

The status is complete only after the downloader has checked every expected file
and its checksum. Dataset hashes and the pinned Hugging Face revision are recorded
in `resource-manifests/dataset.json`. The selected metadata inventory is recorded
in `resource-manifests/metadata.json`. The model manifest records Google Cloud
Storage object generations and CRC32C hashes. Metadata downloads exclude all HDF5
demonstrations, review videos, web viewer previews, card images, and the duplicate
simulator archive. Previously downloaded partial trajectories are retained;
they are no longer being downloaded. Existing simulator meshes and textures are
part of the evaluation installation.

If the download process has stopped, resume it with:

```bash
.hf-env/bin/python scripts/download_resources.py model
.hf-env/bin/python scripts/download_resources.py metadata
```

The default command downloads metadata; `all` downloads metadata plus model
weights. Full training downloads require an explicit `dataset --include-trajectories`
command and must not be resumed without a new user request. Avoid starting
another downloader while one is already running. The current model-only process
ID is recorded in `download-resources.pid`. Downloads survive closing this chat,
but need to be resumed if the machine shuts down or the process fails. The
metadata-only process ID is recorded in `download-metadata.pid`.

The local `.hf-env` environment is for downloads, not model inference. It includes
`huggingface_hub` and `google-crc32c`. Download caches remain under `.cache/` here.

## Using the model after the download

The checkpoint is π0.5 trained on original LIBERO, not a model fine-tuned on
LIBERO-Pro2.0. Its OpenPI environment and the separate simulator environment are
installed in this workspace, and real single-task episodes have succeeded.
For a fresh installation, follow
the [OpenPI installation instructions](https://github.com/Physical-Intelligence/openpi#installation).
For example, from `openpi/`:

```bash
UV_CACHE_DIR="$PWD/../.cache/uv" \
UV_PYTHON_INSTALL_DIR="$PWD/../.cache/uv-python" \
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11

UV_CACHE_DIR="$PWD/../.cache/uv" \
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

OPENPI_DATA_HOME="$PWD/../.cache/openpi" \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero --policy.dir=../models/pi05_libero
```

## Desktop task GUI: LIBERO-Pro2 and LIBERO-Max

Keep the OpenPI server on port 8001. Pro2 also uses its simulator service on port
8000; Max uses its own isolated simulator process.
Open the desktop GUI from this workspace:

```bash
bash scripts/launch_gui.sh
```

The redesigned GUI uses a warm white background, dark text, and larger fonts.
Use **A− / A+** in the header to adjust the text size. The GUI provides:

- A suite selector, searchable task selector, task slider, and previous/next buttons.
- A benchmark selector for Pro2, Max Lite (800 cases), and Max (8,000 cases).
- Expandable condition groups, with **Expand all / Collapse** controls and search
  by name, meaning or Pro2 code. Pro2 has 43 condition types in eight groups,
  including Original Task, Objects & Geometry, Obstacles & Clutter, Cameras &
  Images, Lighting & Appearance, Robot & Calibration, Action Delivery and Language.
  Its **At start / During task** filters distinguish when changes take effect.
- All eight Max event types remain visible in four groups, independently of the
  selected task: move target / destination; camera shift / sensor noise;
  lighting / visual theme; distractors / obstacles. Choose a condition first;
  the task selector and slider then contain compatible tasks. A task count of
  zero means the manifest has no case for that event in the current suite.
- Max includes every published setup from the selected manifest (800 Lite or
  8,000 full cases). The task selector identifies source scene categories;
  **Setup** selects the published sample and severity for that task and event.
- Plain-language condition names and a visible explanation of what changes,
  when it happens, and which objects or cameras it affects. Published condition
  IDs and exact parameters remain available in Condition Details.
- Seed, initial state, episode step limit and action replan settings.
- Preview Scene, Run Task and Stop controls, with both cameras in the same window.
- Current action values, progress, elapsed time, service health and result history.
- A copyable terminal command and session logs.

Preview Scene loads the selected starting state without calling the model.
Runtime interventions become visible when their trigger fires during a run.
For Max, the condition card also lets you choose **With selected change** or
**Without mid-task change**. The viewer reports when the event is applied and
whether the episode finishes before its trigger is reached.
Stop cancels policy waiting and releases the simulator session after the current
simulator request finishes. Closing the app also cleans up the active session.
Selections are locked during a run; stop it before switching to another task.
The latest camera frame replaces stale frames to keep the UI responsive.
Images and videos are not saved; episode results are written as JSON under
`results/`. The GUI reads credentials from `client.env` without displaying them.

The GUI has passed 31 automated checks, including reachability of all 1,688 Pro2,
800 Max Lite and 8,000 Max full cases through grouped navigation. Desktop checks
cover condition search, timing filters, unsupported task combinations, all eight
Max event selectors in every suite, task slider and setup changes, and both
cameras. A Max camera-shift episode through the grouped controls applied the
event and succeeded in 86 steps. Earlier live checks also cover both Max source
types (PRO and Plus) and stopping a worker. The local Tk package uses its Xft build for readable
desktop fonts. An application-only font configuration keeps the layout at 96 DPI
without changing desktop settings.

Max is read from the existing `../LIBERO-MAX` checkout (override with
`LIBERO_MAX_ROOT` if necessary). Its simulator additions are installed under
`.max-runtime/python`, using the existing OpenPI Python 3.11 environment and
model weights. Source modules and simulator assets are prepared inside this
workspace; the adjacent checkout is read without modification. To recreate that
runtime from the existing checkout and its local asset archive:

```bash
bash scripts/setup_max_gui.sh
```

The GUI executes individual Max Base or Dynamic demonstrations using the
upstream event engine. Official Max scores require matched Base/Dynamic pairs
and controlled policy randomness; these single-arm GUI runs do not provide that
paired evaluation protocol. See the [MAX specification](https://github.com/liberomax/LIBERO-MAX/blob/main/docs/BENCHMARK_SPEC.md).

## Run one LIBERO-Pro2 task from a terminal

Leave the OpenPI server running on port 8001. From the workspace root, install
and start the simulator once (this downloads simulator software dependencies,
not demonstrations or model weights):

```bash
bash scripts/setup_libero_sim.sh
```

The setup script stops on errors, preserves existing simulator credentials, and
runs the official simulator smoke test. Then run one baseline task:

```bash
cd /home/ramtin/robotics/LIBERO-Pro2
set -a
source libero-pro-install/LIBERO-Pro-evaluation/client.env
set +a

libero-pro-install/LIBERO-Pro-evaluation/.envs/libero-pro/bin/python \
  scripts/run_pi05_task.py \
  --suite libero_spatial --task-id 0 --variant base
```

The bridge uses the official 180-degree camera rotation, 224-pixel padded resize,
eight-dimensional position/axis-angle/gripper state, and replans every five
actions. The simulator performs its own ten settling steps. The episode uses
the suite's published step limit and writes a result JSON under `results/`.
No frames or videos are recorded. Change `--suite`, `--task-id`, `--variant`,
`--init-state-id` and `--seed` to select a different valid Pro2 instance.

To watch a task in real time, add `--live` to the command above. A desktop window
shows the agent and wrist cameras side by side, updating on each simulator step.
Press Q or Esc, or close the window, to stop early. At the end, the final frame
stays open with the success/failure status until you close it. No video is saved.
Run this from a desktop terminal with `DISPLAY` or `WAYLAND_DISPLAY` configured.

This is a single-episode check. The standard OpenPI websocket server does not
reset its model RNG per episode, so this runner does not claim deterministic
policy replay or reproduction of published benchmark scores. The separate
resumable scheduler and its reproducibility contract need further integration
for a full scored benchmark. The bundled LeRobot `pi0.py` adapter does not load
this OpenPI π0.5 checkpoint.

Eleven bridge tests pass against the vendored OpenPI preprocessing, action queue
behavior, and live viewer orientation and stop controls. A real baseline
LIBERO-Spatial task 0 episode completed successfully through this bridge in
144 steps; this is a single-episode check, not a benchmark aggregate.

The host GPU was checked outside the restricted sandbox: NVIDIA RTX 3080 Ti Laptop
GPU, 16 GB VRAM. GPU inference and simulator rendering have both been exercised
successfully in the single-task runs described above.
