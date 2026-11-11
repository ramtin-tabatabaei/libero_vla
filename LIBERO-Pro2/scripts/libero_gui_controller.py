"""Task/condition selection and the isolated LIBERO-Max simulator worker."""
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from libero_gui_catalog import ROOT, RunSettings

class ExplorerController:

    def settings(self):
        if self.case is None:
            raise ValueError("Select a task and a condition first")
        return RunSettings(self.suite, self.task_id, self.case, int(self.seed_var.get()), int(self.init_var.get()),
            int(self.limit_var.get()), int(self.replan_var.get()), benchmark=self.catalog.benchmark_id,
            arm="control" if self.arm_var.get() == "Without mid-task change" else "intervention").validate(self.catalog)

    def update_command(self):
        if not hasattr(self, "command_text"):
            return
        try:
            settings = self.settings()
            command = settings.command()
            if settings.benchmark == "pro2":
                if self.policy_host != "127.0.0.1" or self.policy_port != 8001 or self.base_url != "http://127.0.0.1:8000":
                    command += " --base-url " + shlex.quote(self.base_url) + " --policy-host " + shlex.quote(self.policy_host) + " --policy-port " + str(self.policy_port)
            else:
                command += " --host " + shlex.quote(self.policy_host) + " --port " + str(self.policy_port)
        except (ValueError, IndexError, KeyError) as error:
            command = "Complete the settings to generate a command.\n" + str(error)
        self.set_text(self.command_text, command)

    def run_max_worker(self, mode, settings, connection):
        _, _, host, port = connection
        command = [str(ROOT / "openpi/.venv/bin/python"), str(ROOT / "scripts/run_max_task.py"),
            "--profile", "lite" if settings.benchmark == "max-lite" else "full", "--case-id", settings.case,
            "--arm", settings.arm, "--mode", mode, "--seed", str(settings.seed),
            "--init-state-id", str(settings.init_state_id), "--max-steps", str(settings.max_steps),
            "--replan-steps", str(settings.replan_steps), "--host", host, "--port", str(port), "--ipc"]
        process, completed, recent = None, False, []
        try:
            if self.stop_event.is_set():
                self.events.put(("stopped", None)); return
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                       NUMBA_CACHE_DIR=str(ROOT / ".cache/numba-max"))
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            def cancel():
                while process.poll() is None:
                    if self.stop_event.wait(0.1):
                        try:
                            process.stdin.write("STOP\n");process.stdin.flush()
                        except (OSError, ValueError):
                            pass
                        try: process.wait(timeout=8)
                        except subprocess.TimeoutExpired: process.terminate()
                        return
            threading.Thread(target=cancel, daemon=True).start()
            for line in process.stdout:
                try: data = json.loads(line)
                except (ValueError, TypeError):
                    if line.strip():
                        recent.append(line.strip()[:700]);recent = recent[-10:]
                        self.events.put(("log", line.strip()[:700]))
                    continue
                if not isinstance(data, dict) or "type" not in data:
                    continue
                kind = data.get("type")
                if kind == "frame":
                    pair = tuple(np.array(Image.open(BytesIO(base64.b64decode(value))).convert("RGB")) for value in data["images"])
                    try: self.frames.put_nowait(pair)
                    except queue.Full:
                        try: self.frames.get_nowait()
                        except queue.Empty: pass
                        self.frames.put_nowait(pair)
                    self.events.put(("step", data["step"]))
                elif kind == "result":
                    completed = True
                    self.events.put(("result", (data["result"], Path(data["path"]))))
                elif kind == "instruction": self.events.put(("instruction", data["text"]))
                elif kind == "action": self.events.put(("action", data["values"]))
                elif kind == "status": self.events.put(("status", data["text"]))
                elif kind in ("preview", "stopped"):
                    completed = True; self.events.put((kind, None))
                elif kind == "intervention": self.events.put(("event_applied", data["text"]))
            code = process.wait()
            if self.stop_event.is_set() and not completed:
                self.events.put(("stopped", None))
            elif code != 0 or not completed:
                raise RuntimeError("LIBERO-Max worker could not complete. " + "\n".join(recent[-5:]))
        except Exception as error:
            self.events.put(("error", str(error)))
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream: stream.close()
            self.events.put(("idle", None))
