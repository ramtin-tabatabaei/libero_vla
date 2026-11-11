#!/usr/bin/env python3
"""Desktop task browser and live controller for the local LIBERO-Pro2 servers."""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from libero_gui_catalog import ROOT, BUNDLE, SUITES, TaskCatalog, RunSettings, read_client_config, readable, expression
from run_pi05_task import OpenPiAgent

from libero_gui_view import ExplorerView, BG, PANEL, CARD, TEXT, MUTED, ACCENT, RED, EDGE
from libero_gui_controller import ExplorerController
from libero_condition_selection import GroupedConditionSelection


class EpisodeStopped(Exception):
    pass


class GuiPolicy:
    """OpenPI's messagepack protocol with interruptible inference waiting."""

    def __init__(self, host, port, stop):
        self.stop = stop
        self._ws = connect(f"ws://{host}:{port}", compression=None, max_size=None,
                           open_timeout=5, close_timeout=1)
        try:
            self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=5))
        except BaseException:
            self._ws.close()
            raise

    def reset(self):
        pass  # The stock OpenPI server has no per-episode RNG reset endpoint.

    def infer(self, observation):
        self._ws.send(msgpack_numpy.packb(observation))
        deadline = time.monotonic() + 900
        while not self.stop.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("The policy did not return an action within 15 minutes")
            try:
                response = self._ws.recv(timeout=0.2)
            except TimeoutError:
                continue
            if isinstance(response, str):
                raise RuntimeError(response)
            return msgpack_numpy.unpackb(response)
        raise EpisodeStopped()


class FrameObserver:
    def __init__(self, frames, events, stop):
        self.frames, self.events, self.stop = frames, events, stop
        self.step = 0

    def write(self, observation):
        if self.stop.is_set():
            raise EpisodeStopped()
        pair = tuple(np.array(observation[k], copy=True) for k in
                     ("agentview_image", "robot0_eye_in_hand_image"))
        try:
            self.frames.put_nowait(pair)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(pair)
        self.events.put(("step", self.step))
        self.step += 1


class GuiAgent(OpenPiAgent):
    def __init__(self, policy, replan_steps, events, stop):
        super().__init__(policy, replan_steps)
        self.events, self.stop = events, stop

    def reset(self, context):
        super().reset(context)
        self.events.put(("instruction", context.instruction))

    def act(self, observation):
        if self.stop.is_set():
            raise EpisodeStopped()
        previous_instruction = self.instruction
        action = super().act(observation)
        if self.stop.is_set():
            raise EpisodeStopped()
        self.events.put(("action", action.tolist()))
        if observation["instruction"] != previous_instruction:
            self.events.put(("instruction", observation["instruction"]))
        return action


def get_health(url, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with urlopen(Request(url, headers=headers), timeout=4) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return response.read()


def condition_text(details):
    entry, fields = details["entry"], details["condition"]
    timing = {"base": "Original environment", "static": "Applied at reset",
              "dynamic": "Applied during execution"}[entry["case_kind"]]
    lines = [f"{entry['case_id']}  ·  {timing}", "", details["instruction"], ""]
    if fields:
        for name in ("operator", "variant", "targets", "references", "trigger", "constraints"):
            if name in fields:
                lines.extend([name.replace("_", " ").title(),
                              readable(" ".join(str(x) for x in fields[name])), ""])
        if fields.get("params"):
            lines.append("Parameters declared by this task")
            for item in fields["params"]:
                if isinstance(item, list) and item:
                    value = " ".join(expression(x) if isinstance(x, list) else str(x) for x in item[1:])
                    lines.append(f"  {readable(item[0].lstrip(':'))}: {readable(value)}")
            lines.append("")
    else:
        lines.extend(["This is the original task without an additional perturbation.", ""])
    lines.extend(["Success condition", details["goal"], "", "Task definition", str(details["path"])])
    return "\n".join(lines)


class LiberoApp(GroupedConditionSelection, ExplorerController, ExplorerView):
    def __init__(self, root):
        self.root = root
        self.catalog = TaskCatalog()
        self.catalogs = {"LIBERO-Pro2": self.catalog}
        self.config = read_client_config()
        self.base_url = self.config.get("LIBERO_REMOTE_URL", "http://127.0.0.1:8000")
        self.api_key = self.config.get("LIBERO_REMOTE_API_KEY", "")
        self.policy_host, self.policy_port = "127.0.0.1", 8001
        self.events, self.frames = queue.Queue(), queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = None
        self.busy = self.closing = self.updating = False
        self.active_index = -1
        self.task_ids, self.case_entries = [], []
        self.last_frames = self.photos = None
        self.started = 0
        self.last_result_path = None
        self.controls = []
        self.variables()
        self.style()
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.change_benchmark()
        self.load_history()
        self.root.after(40, self.drain)
        self.root.after(100, self.tick)

    def variables(self):
        self.benchmark_var = tk.StringVar(value="LIBERO-Pro2")
        self.condition_query_var = tk.StringVar()
        self.condition_timing_var = tk.StringVar(value="All")
        self.condition_help_var = tk.StringVar()
        self.task_var = tk.StringVar()
        self.scenario_var = tk.StringVar()
        self.condition_query_var.trace_add("write", self.refresh_condition_tree)
        self.search_var = tk.StringVar()
        self.arm_var = tk.StringVar(value="With selected change")
        self.condition_count_var = tk.StringVar()
        self.change_title_var = tk.StringVar()
        self.change_description_var = tk.StringVar()
        self.change_when_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_tasks)
        self.suite_var = tk.StringVar(value="Spatial")
        self.seed_var = tk.StringVar(value="142")
        self.init_var = tk.StringVar(value="0")
        self.limit_var = tk.StringVar(value="220")
        self.replan_var = tk.StringVar(value="5")
        self.task_count_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.instruction_var = tk.StringVar()
        self.condition_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready to explore")
        self.step_var = tk.StringVar(value="— / —")
        self.elapsed_var = tk.StringVar(value="0.0 s")
        self.sim_health = tk.StringVar(value="Simulator · checking")
        self.model_health = tk.StringVar(value="π0.5 · checking")
        self.action_vars = [tk.StringVar(value="—") for _ in range(7)]
        self.result_var = tk.StringVar(value="Choose a task, preview its scene, then press Run task.")
        for variable in (self.seed_var, self.init_var, self.limit_var, self.replan_var):
            variable.trace_add("write", lambda *_: self.update_command())


    def register(self, widget, state="normal"):
        self.controls.append((widget, state))
        return widget


    def text_box(self, parent, height=12, expand=True):
        box = ttk.Frame(parent)
        box.pack(fill="both" if expand else "x", expand=expand)
        text = tk.Text(box, height=height, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", borderwidth=0,
                       wrap="word", padx=14, pady=12, font=self.fonts["code"], state="disabled")
        text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(box, command=text.yview)
        scroll.pack(side="right", fill="y")
        text.configure(yscrollcommand=scroll.set)
        return text

    def set_text(self, widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", value)
        widget.configure(state="disabled")

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", datetime.now().strftime("%H:%M:%S") + "  " + message + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 250:
            self.log_text.delete("1.0", "40.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @property
    def suite(self):
        return next(key for key, value in SUITES.items() if value == self.suite_var.get())

    @property
    def task_id(self):
        return self.task_ids[self.active_index]


    def slide_changed(self, value):
        self.set_task(int(round(float(value))))

    def roll_task(self, delta):
        self.set_task(self.active_index + delta)
        return "break"


    def copy_command(self):
        try:
            self.settings()
        except (ValueError, KeyError) as error:
            messagebox.showerror("Run settings", str(error), parent=self.root)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.command_text.get("1.0", "end-1c"))
        self.log("Command copied to clipboard.")

    def set_controls(self, busy):
        for widget, idle_state in self.controls:
            if isinstance(widget, ttk.Treeview):
                widget.state(["disabled"] if busy else ["!disabled"])
            else:
                widget.configure(state="disabled" if busy else idle_state)
        if self.catalog.benchmark_id == "pro2":
            self.arm_picker.configure(state="disabled")
            self.arm_row.pack_forget()
        else:
            self.arm_row.pack(anchor="w", fill="x", pady=(12, 0))
        enabled = not busy and self.case is not None
        self.run_button.configure(state="normal" if enabled else "disabled")
        self.preview_button.configure(state="normal" if enabled else "disabled")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def start(self, mode):
        if self.busy:
            return
        try:
            settings = self.settings()
            if self.catalog.benchmark_id == "pro2" and not self.api_key:
                raise ValueError("Simulator credentials are missing. Run bash scripts/setup_libero_sim.sh first.")
        except (ValueError, KeyError) as error:
            messagebox.showerror("Cannot start", str(error), parent=self.root)
            return
        self.busy = True
        self.stop_event.clear()
        self.started = time.monotonic()
        self.set_controls(True)
        self.tabs.select(0)
        self.progress.configure(maximum=settings.max_steps, value=0)
        self.step_var.set(f"0 / {settings.max_steps}")
        self.status_var.set("Preparing scene…")
        self.result_var.set("Loading the selected task and condition.")
        self.last_frames = None
        self.clear_frames()
        self.draw_frames()
        self.log(f"{'Run' if mode == 'run' else 'Preview'}: {settings.suite} / task {settings.task_id} / {settings.case}, seed {settings.seed}.")
        connection = (self.base_url, self.api_key, self.policy_host, self.policy_port)
        self.worker = threading.Thread(target=self.run_worker, args=(mode, settings, connection), daemon=True)
        self.worker.start()

    def run_worker(self, mode, settings, connection):
        if settings.benchmark != "pro2":
            self.run_max_worker(mode, settings, connection)
            return
        base_url, api_key, host, port = connection
        agent = None
        try:
            from libero_remote import RemoteLiberoEnv
            from libero_remote.runner import AgentEpisodeSpec, run_remote_agent_episode
            get_health(base_url.rstrip("/") + "/v1/health", api_key)
            observer = FrameObserver(self.frames, self.events, self.stop_event)
            with RemoteLiberoEnv(base_url, api_key=api_key, timeout=120) as env:
                if mode == "preview":
                    observation, info = env.reset(suite=settings.suite, task_id=settings.task_id,
                        variant=settings.variant, seed=settings.seed, init_state_id=settings.init_state_id,
                        max_steps=settings.max_steps, control_mode="relative", camera_height=256, camera_width=256)
                    self.events.put(("instruction", info["language_instruction"]))
                    observer.write(observation)
                    self.events.put(("preview", None))
                    return
                get_health(f"http://{host}:{port}/healthz")
                if self.stop_event.is_set():
                    raise EpisodeStopped()
                policy = GuiPolicy(host, port, self.stop_event)
                agent = GuiAgent(policy, settings.replan_steps, self.events, self.stop_event)
                spec = AgentEpisodeSpec(settings.suite, settings.task_id, settings.init_state_id,
                    settings.seed, settings.max_steps, control_mode="relative", variant=settings.variant)
                self.events.put(("status", "Running π0.5"))
                result = run_remote_agent_episode(env, agent, spec, recorder=observer).to_dict()
                result["server_result"] = env.result()
            result.update({"variant": settings.variant, "replan_steps": settings.replan_steps,
                "max_steps": settings.max_steps, "purpose": "single-episode check", "policy_rng_reset": False,
                "live_preview": True})
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = ROOT / "results" / f"gui_{settings.suite}_task{settings.task_id}_{settings.variant}_{stamp}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
            self.events.put(("result", (result, path)))
        except EpisodeStopped:
            self.events.put(("stopped", None))
        except Exception as error:
            self.events.put(("error", f"{type(error).__name__}: {error}"))
            self.events.put(("log", traceback.format_exc()))
        finally:
            try:
                if agent is not None:
                    agent.close()
            except Exception as error:
                self.events.put(("log", f"Policy connection cleanup: {error}"))
            self.events.put(("idle", None))

    def stop(self):
        if self.busy:
            self.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.status_var.set("Stopping…")
            self.result_var.set("Finishing the current simulator request and releasing the session.")

    def drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "step":
                    self.step_var.set(f"{payload} / {self.limit_var.get()}")
                    self.progress["value"] = payload
                elif kind == "action":
                    for variable, value in zip(self.action_vars, payload):
                        variable.set(f"{value:+.3f}")
                elif kind == "instruction":
                    # MAX filename-derived instructions can contain scene variant
                    # suffixes. Show its declared BDDL goal while preserving the
                    # actual policy prompt in the log and upstream trace.
                    display = (self.catalog.details(self.suite, self.task_id, self.case)["instruction"]
                               if self.catalog.benchmark_id != "pro2" else payload)
                    self.instruction_var.set(display)
                    self.log("Instruction: " + payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "event_applied":
                    self.change_when_var.set("APPLIED NOW · " + payload)
                    self.log(payload)
                elif kind == "preview":
                    self.status_var.set("Scene preview")
                    self.result_var.set("Starting scene loaded. Runtime conditions take effect during a run.")
                    self.log("Starting scene preview ready.")
                elif kind == "result":
                    result, path = payload
                    self.last_result_path = path
                    self.status_var.set("Success ✓" if result["success"] else "Goal not reached")
                    if "event_applied" in result:
                        self.change_when_var.set("Base episode: no mid-task change applied." if result.get("arm") == "control" else
                            "The selected change was applied during this run." if result["event_applied"] else
                            "This run ended before the condition's trigger was reached.")
                    self.result_var.set(f"{'SUCCESS' if result['success'] else 'FINISHED'} · {result['steps']} steps · {result['duration_seconds']:.1f} s · {path.name}")
                    self.log(f"Episode finished: success={result['success']}; result saved to {path}")
                    self.load_history()
                elif kind == "stopped":
                    self.status_var.set("Stopped")
                    self.result_var.set("Episode stopped. Select another task or condition to continue.")
                    self.log("Episode stopped; simulator session closed.")
                elif kind == "error":
                    self.status_var.set("Could not complete")
                    self.result_var.set(payload[:300])
                    self.log(payload)
                elif kind == "log":
                    self.log(payload)
                elif kind == "health":
                    for name, ok, text in payload:
                        variable, badge = (self.sim_health, self.sim_badge) if name == "Simulator" else (self.model_health, self.model_badge)
                        variable.set(f"{name} · {'ready' if ok else 'offline'}")
                        badge.configure(foreground=ACCENT if ok else RED)
                        if not ok:
                            self.log(f"{name}: {text}")
                elif kind == "idle":
                    self.busy = False
                    self.set_controls(False)
        except queue.Empty:
            pass
        try:
            self.last_frames = self.frames.get_nowait()
            self.draw_frames()
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(40, self.drain)

    def draw_frames(self):
        if not hasattr(self, "canvases"):
            return
        photos = []
        for i, canvas in enumerate(self.canvases):
            width, height = max(100, canvas.winfo_width()), max(100, canvas.winfo_height())
            canvas.delete("all")
            if self.last_frames is None:
                canvas.create_text(width / 2, height / 2 - 12, text="LIVE CAMERA", fill=EDGE, font=self.fonts["instruction"])
                canvas.create_text(width / 2, height / 2 + 18, text="Preview a scene or run a task", fill="#b7bab1", font=self.fonts["small"])
            else:
                image = Image.fromarray(np.ascontiguousarray(self.last_frames[i][::-1, ::-1]))
                size = min(width - 8, height - 8)
                image = image.resize((size, size), Image.Resampling.BILINEAR)
                photo = ImageTk.PhotoImage(image)
                photos.append(photo)
                canvas.create_image(width / 2, height / 2, image=photo)
        self.photos = photos

    def clear_frames(self):
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

    def tick(self):
        if self.busy:
            self.elapsed_var.set(f"{time.monotonic() - self.started:.1f} s")
        self.root.after(100, self.tick)

    def check_health(self):
        base_url, api_key, host, port = self.base_url, self.api_key, self.policy_host, self.policy_port
        is_max = self.catalog.benchmark_id != "pro2"
        def check():
            results = []
            for name, url, key in [("Simulator", base_url.rstrip("/") + "/v1/health", api_key),
                                   ("π0.5", f"http://{host}:{port}/healthz", None)]:
                try:
                    if name == "Simulator" and is_max:
                        required = [ROOT / ".max-runtime/assets-ready.json",
                                    ROOT / ".max-runtime/python/robosuite/__init__.py",
                                    ROOT / "openpi/.venv/bin/python"]
                        if not all(p.exists() for p in required):
                            raise RuntimeError("Prepare the Max simulator with bash scripts/setup_max_gui.sh")
                    else:
                        get_health(url, key)
                    results.append((name, True, "Ready"))
                except Exception as error:
                    results.append((name, False, str(error)))
            self.events.put(("health", results))
        threading.Thread(target=check, daemon=True).start()

    def connections(self):
        if self.busy:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Local connections")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        fields = []
        for i, (title, value) in enumerate([("Simulator URL", self.base_url), ("Policy host", self.policy_host), ("Policy port", self.policy_port)]):
            ttk.Label(dialog, text=title).grid(row=i, column=0, sticky="w", padx=18, pady=12)
            variable = tk.StringVar(value=str(value))
            entry = ttk.Entry(dialog, textvariable=variable, width=36)
            entry.grid(row=i, column=1, padx=(0, 18), pady=12)
            fields.append(variable)
        ttk.Label(dialog, text="Simulator credentials are loaded from client.env.", style="Muted.TLabel").grid(row=3, column=0, columnspan=2, padx=18, pady=12)
        def apply():
            try:
                port = int(fields[2].get())
                base, host = fields[0].get().strip(), fields[1].get().strip()
                if not 1 <= port <= 65535 or not host or not base.startswith(("http://", "https://")):
                    raise ValueError("Enter an HTTP simulator URL, policy host, and port between 1 and 65535")
            except ValueError as error:
                messagebox.showerror("Connections", str(error), parent=dialog)
                return
            self.base_url, self.policy_host, self.policy_port = base.rstrip("/"), host, port
            dialog.destroy()
            self.update_command()
            self.check_health()
        ttk.Button(dialog, text="Apply & check", command=apply).grid(row=4, column=1, sticky="e", padx=18, pady=(0, 18))

    def load_history(self):
        self.history_paths = []
        self.history_list.delete(0, "end")
        for path in sorted((ROOT / "results").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text())
                if "success" not in data or "suite" not in data:
                    continue
                self.history_paths.append(path)
                self.history_list.insert("end", f"{'✓' if data['success'] else '○'}  {data.get('benchmark', 'LIBERO-Pro2')} · {SUITES.get(data['suite'], data['suite'])} / {data['task_id']:02d} · {data['steps']} steps")
            except (OSError, ValueError, KeyError):
                continue
            if len(self.history_paths) >= 30:
                break
        if self.history_paths:
            self.history_list.selection_set(0)
            self.show_history()

    def show_history(self, event=None):
        selected = self.history_list.curselection()
        if selected:
            path = self.history_paths[selected[0]]
            self.set_text(self.history_text, str(path) + "\n\n" + path.read_text())

    def on_close(self):
        self.closing = True
        if self.busy:
            self.stop()
            self.status_var.set("Closing session…")
        elif self.worker is None or not self.worker.is_alive():
            self.root.destroy()


def main():
    logging.basicConfig(level=logging.WARNING)
    os.environ["FONTCONFIG_FILE"] = str(ROOT / "scripts/gui-fonts.conf")
    root = tk.Tk()
    LiberoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
