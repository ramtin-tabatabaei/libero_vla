# Continue on another computer

Git stores the workspace scripts, tests, documentation, and resource manifests.
Downloaded datasets, model weights, simulator installations, environments,
credentials, results, and upstream checkouts stay outside Git. The existing
OpenPI checkout is clean, and every file in the Pro2 evaluation source archive
matches its extracted copy, so these dependencies can be restored from upstream.

The runtime instructions below target Linux with Conda, uv, curl, Git, and
compatible NVIDIA/EGL drivers. The current computer has a 16 GB NVIDIA GPU;
the recorded successful episodes are described in README.md.

## 1. Clone your project

On the new computer, clone the combined repository and enter this workspace:

```bash
git clone https://github.com/ramtin-tabatabaei/libero_vla.git
cd libero_vla/LIBERO-Pro2
```

## 2. Restore OpenPI and the download tools

Run from the new workspace root. OpenPI is pinned to the commit used here.

```bash
git clone https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git -C openpi submodule update --init --recursive

uv venv --python 3.11 .hf-env
uv pip install --python .hf-env/bin/python huggingface_hub google-crc32c
```

## 3. Restore the simulator release and weights

Download the exact evaluation release used on the original computer. This
downloads the simulator assets, not training demonstrations.

```bash
.hf-env/bin/hf download zhouxueyang/LIBERO-Pro2.0-Eval \
  --repo-type dataset --revision d1933228f8be59f5d8fa1d8e2e251b463dfb64bc \
  --include 'evaluation.zip' 'evaluation.zip.sha256' \
            'assets.zip' 'assets.zip.sha256' 'manifests/**' 'scripts/**' \
            'docs/**' 'README.md' 'LICENSE_NOTICE.md' 'release.json' \
  --local-dir LIBERO-Pro2.0-Eval

.hf-env/bin/python LIBERO-Pro2.0-Eval/scripts/prepare_evaluation.py \
  --download-dir LIBERO-Pro2.0-Eval --output-dir libero-pro-install

.hf-env/bin/python scripts/download_resources.py model
```

The preparation script requires a new `libero-pro-install` directory and
verifies archive and member checksums before extraction. If you already copied
an installation in the optional transfer step below, skip its extraction.
The checkpoint is approximately 12.4 GB. The downloader resumes partial files
and verifies checksums. Optional training metadata can be restored with:

```bash
.hf-env/bin/python scripts/download_resources.py metadata
```

Full training trajectories were intentionally stopped on the original machine;
they are unnecessary for inference and are not part of these steps.

## 4. Recreate the environments and run

From the workspace root, install OpenPI in a subshell:

```bash
(
  cd openpi
  GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
)

bash scripts/setup_libero_sim.sh
```

The simulator setup generates fresh local `client.env` and `server.env`
credentials, starts the service, and runs its smoke test. Do not commit these
files. Start the policy server in a separate terminal from the workspace root:

```bash
(
  cd openpi
  OPENPI_DATA_HOME="$PWD/../.cache/openpi" \
  uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero --policy.dir=../models/pi05_libero
)
```

Then launch the GUI from the workspace root:

```bash
bash scripts/launch_gui.sh
```

Pro2 uses service port 8000 and policy port 8001. A graphical desktop is needed
for the GUI. See README.md for terminal episode commands and runtime controls.
Replace the old absolute `cd /home/ramtin/robotics/LIBERO-Pro2` example there
with your new workspace path.

## Optional: copy downloads instead of downloading again

With SSH access from the new computer to the old one, run these commands on
the new computer from inside the cloned project. Replace `OLD_USER@OLD_HOST`.

```bash
rsync -avh --partial --info=progress2 \
  --exclude='.cache/' \
  OLD_USER@OLD_HOST:/home/ramtin/robotics/LIBERO-Pro2/models/ models/

rsync -avh --partial --info=progress2 \
  --exclude='.cache/' --exclude='__pycache__/' \
  OLD_USER@OLD_HOST:/home/ramtin/robotics/LIBERO-Pro2/LIBERO-Pro2.0-Eval/ \
  LIBERO-Pro2.0-Eval/
```

You can then use the preparation command in step 3 to extract the copied
archives and run `scripts/download_resources.py model` to verify the copied
weights. Environments must be recreated on the new computer.
Copy `results/` separately with rsync if you want the old episode history.

## Optional: restore LIBERO-Max

Max additionally depends on the adjacent `../LIBERO-MAX` checkout, its
`.deps/LIBERO-plus` and `.deps/LIBERO-PRO` sources, task data, and local asset
archive. These are separate from the Pro2 repository. The original Max commit
is `01625e290b1aa07511bdc820781e635e851353cc`.

The direct way to retain the prepared dependencies is to copy that checkout
from the old computer, including its `.deps` directory:

```bash
rsync -avh --partial --info=progress2 \
  --exclude='.venv/' --exclude='__pycache__/' --exclude='.cache/' \
  OLD_USER@OLD_HOST:/home/ramtin/robotics/LIBERO-MAX/ ../LIBERO-MAX/

bash scripts/setup_max_gui.sh
```

The setup recreates this workspace's `.max-runtime` using the new OpenPI
environment. If copying is unavailable, clone
`https://github.com/liberomax/LIBERO-MAX.git`, check out the pinned commit,
and follow its `docs/RUNTIME_INTEGRATION.md` to restore the `.deps` sources
and assets before running `setup_max_gui.sh`. Set `LIBERO_MAX_ROOT` if you keep
Max somewhere other than the adjacent directory.

## Continue development

Commit and push changes from the clone on either computer:

```bash
git add scripts tests README.md MIGRATION.md .gitignore resource-manifests
git commit -m "Describe your change"
git push
```

On the other computer, commit any local changes first, then use
`git pull --ff-only`. Git synchronizes source; the large ignored resources
remain on each computer.
