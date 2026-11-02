"""Live, nonmodal waypoint coverage report for the studio's VLA runs."""
import time
import tkinter as tk
from tkinter import filedialog, ttk

from waypoint_coverage import RolloutCoverage
from waypoint_task_events import FailureConfig


class CoverageView:
    def __init__(self, panel, parent):
        self.panel = panel
        self.monitor = self.window = None
        self.last_update = 0.
        self.enabled = tk.BooleanVar(value=True)
        self.watch_failures = tk.BooleanVar(value=True)
        self.expected_object = tk.StringVar(value="Automatic")
        self.force_n = tk.StringVar(value="10")
        self.off_cm = tk.StringVar(value="30")
        self.off_seconds = tk.StringVar(value="1")
        self.lift_cm = tk.StringVar(value="2")
        self.pickup_seconds = tk.StringVar(value="0.15")
        self.last_event_count = 0
        self.centimetres = tk.StringVar(value="5")
        self.check_angle = tk.BooleanVar(value=False)
        self.degrees = tk.StringVar(value="30")
        self.summary = tk.StringVar(value="Pose coverage is measured during the next VLA run.")
        bar = ttk.Frame(parent, style="Card.TFrame")
        bar.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        bar.columnconfigure(0, weight=1)
        checks = ttk.Frame(bar, style="Card.TFrame")
        checks.grid(row=0, column=0, sticky="w")
        self.check = ttk.Checkbutton(checks, text="Waypoints", variable=self.enabled)
        self.check.pack(side="left")
        self.failure_check = ttk.Checkbutton(checks, text="Failure times", variable=self.watch_failures)
        self.failure_check.pack(side="left", padx=(5, 0))
        self.button = panel.ui_button(bar, "Settings / report…", self.show)
        self.button.grid(row=0, column=1, padx=(6, 0))
        ttk.Label(parent, textvariable=self.summary, style="Muted.TLabel",
                  wraplength=410).grid(row=5, column=0, sticky="w", pady=(4, 0))
        panel.rows.tag_configure("pose_visited", background="#d9f3e9", foreground="#096650")
        panel.rows.tag_configure("pose_missed", background="#ffe5e7", foreground="#9c2938")

    def prepare(self, session, chain, backend):
        if not self.enabled.get() and not self.watch_failures.get():
            self.summary.set("Waypoint checking is off.")
            return None
        if not chain.waypoints and not self.watch_failures.get():
            self.summary.set("Load or save a waypoint plan to check the VLA's poses.")
            return None
        try:
            distance = float(self.centimetres.get()) / 100 if self.enabled.get() else .05
            angle = float(self.degrees.get()) if self.enabled.get() and self.check_angle.get() else None
        except ValueError:
            raise ValueError("Waypoint tolerances must be numbers.")
        # Validate before opening the report or starting the policy.
        import numpy as np
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Waypoint distance tolerance must be greater than zero.")
        if angle is not None and (not np.isfinite(angle) or not 0 < angle <= 180):
            raise ValueError("Wrist angle tolerance must be greater than 0 and at most 180 degrees.")
        failure_config = None
        if self.watch_failures.get():
            try:
                choice = self.expected_object.get()
                expected = "automatic" if choice == "Automatic" else "none" if choice == "No pickup expected" else choice
                failure_config = FailureConfig(float(self.force_n.get()), float(self.off_cm.get())/100,
                    float(self.off_seconds.get()), float(self.lift_cm.get())/100,
                    float(self.pickup_seconds.get()), expected).validate()
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid failure timing settings: " + str(exc))
        self.last_event_count = 0
        self.monitor = RolloutCoverage(session, chain, backend, distance, angle, self.update,
            failure_config=failure_config, check_waypoints=self.enabled.get())
        self.show()
        return self.monitor

    def show(self):
        if self.window is not None and self.window.winfo_exists():
            self.expected_box.configure(values=self.object_choices())
            self.window.lift()
            self.update(self.monitor, force=True)
            return
        window = self.window = tk.Toplevel(self.panel.root)
        window.title("VLA · waypoint coverage and failure times")
        window.geometry("1080x700")
        window.minsize(790, 450)
        window.transient(self.panel.root)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(3, weight=1)
        self.caption = tk.StringVar(value="Choose tolerances, then run the selected VLA in the main window.")
        ttk.Label(window, textvariable=self.caption, wraplength=940).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        settings = ttk.Frame(window, padding=(16, 0))
        settings.grid(row=1, column=0, sticky="ew")
        ttk.Label(settings, text="Distance (cm)").pack(side="left")
        distance = ttk.Entry(settings, textvariable=self.centimetres, width=6)
        distance.pack(side="left", padx=(6, 20))
        angle_check = ttk.Checkbutton(settings, text="Also check wrist angle", variable=self.check_angle)
        angle_check.pack(side="left")
        angle = ttk.Entry(settings, textvariable=self.degrees, width=6)
        angle.pack(side="left", padx=6)
        ttk.Label(settings, text="degrees · applies to the next run").pack(side="left")
        self.settings = (distance, angle_check, angle)
        self.details = tk.StringVar(value="No rollout has been checked yet.")
        ttk.Label(window, textvariable=self.details, wraplength=940).grid(row=2, column=0, sticky="ew", padx=16, pady=12)
        self.tabs = ttk.Notebook(window)
        self.tabs.grid(row=3, column=0, sticky="nsew", padx=16)
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text="Waypoints")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("waypoint", "status", "distance", "angle", "visit", "ordered")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for key, label, width in zip(columns,
                ("Waypoint", "Pose match", "Closest (cm)", "Angle there (°)", "First visit (s)", "In order (s)"),
                (220, 175, 100, 110, 110, 100)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=70, stretch=key in ("waypoint", "status"))
        self.table.grid(row=0, column=0, sticky="nsew")
        self.table.tag_configure("visited", background="#d9f3e9")
        self.table.tag_configure("missed", background="#ffe5e7")
        scroll = ttk.Scrollbar(frame, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.build_failure_tabs()
        ttk.Label(window, text="Targets are fixed at rollout start. The order count stops at the first waypoint not yet reached.\n"
                              "Pose matches alone do not verify actions. Failure times use separate contact, lift and distance measurements.",
                  wraplength=940).grid(row=4, column=0, sticky="ew", padx=16, pady=12)
        buttons = ttk.Frame(window, padding=(16, 0, 16, 16))
        buttons.grid(row=5, column=0, sticky="ew")
        ttk.Button(buttons, text="Save report CSV…", command=self.save).pack(side="left")
        ttk.Button(buttons, text="Save failure CSV…", command=self.save_events).pack(side="left", padx=6)
        ttk.Button(buttons, text="Save full JSON…", command=self.save_json).pack(side="left")
        ttk.Button(buttons, text="Stop VLA", command=self.panel.stop).pack(side="right")
        self.update(self.monitor, force=True)

    def object_choices(self):
        env = self.panel.session.env
        return ["Automatic", "No pickup expected"] + (sorted(env.env.objects_dict) if env is not None else [])

    def build_failure_tabs(self):
        self.event_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.event_tab, text="Failure timeline")
        self.event_tab.columnconfigure(0, weight=1)
        self.event_tab.rowconfigure(1, weight=1)
        self.event_status = tk.StringVar(value="Enable Failure times, then run the VLA.")
        ttk.Label(self.event_tab, textvariable=self.event_status, wraplength=960).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.event_table = ttk.Treeview(self.event_tab,
            columns=("event", "onset", "confirmed", "wall", "step", "detail"), show="headings", height=8)
        for key, label, width in zip(("event", "onset", "confirmed", "wall", "step", "detail"),
                ("Event", "Onset (sim s)", "Confirmed (s)", "Wall (s)", "Action", "Evidence"), (210, 120, 120, 95, 65, 350)):
            self.event_table.heading(key, text=label)
            self.event_table.column(key, width=width, minwidth=50, stretch=key == "detail")
        self.event_table.grid(row=1, column=0, sticky="nsew")
        self.event_table.tag_configure("alert", background="#ffe5e7")
        scroll = ttk.Scrollbar(self.event_tab, command=self.event_table.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.event_table.configure(yscrollcommand=scroll.set)
        self.event_evidence = tk.StringVar(value="Select an event to read its full evidence.")
        def select_event(_):
            selected = self.event_table.selection()
            if selected:
                self.event_evidence.set(self.event_table.item(selected[0], "values")[-1])
        self.event_table.bind("<<TreeviewSelect>>", select_event)
        ttk.Label(self.event_tab, textvariable=self.event_evidence, wraplength=960).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.failure_rules = tk.StringVar(value="")
        ttk.Label(self.event_tab, textvariable=self.failure_rules, wraplength=960).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        settings = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(settings, text="Failure settings")
        settings.columnconfigure(1, weight=1)
        enabled = ttk.Checkbutton(settings, text="Time task failures during VLA runs", variable=self.watch_failures)
        enabled.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(settings, text="Expected pickup object").grid(row=1, column=0, sticky="w", padx=(0, 14))
        self.expected_box = ttk.Combobox(settings, textvariable=self.expected_object,
                                         values=self.object_choices(), state="readonly", width=44)
        self.expected_box.grid(row=1, column=1, sticky="w")
        controls = [enabled, self.expected_box]
        fields = (("Collision force (N)", self.force_n),
                  ("Far from plan (cm)", self.off_cm), ("Stay far for (sim s)", self.off_seconds),
                  ("Wrong-object lift (cm)", self.lift_cm), ("Keep grasp + lift for (sim s)", self.pickup_seconds))
        for row, (label, variable) in enumerate(fields, 2):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(settings, textvariable=variable, width=9)
            entry.grid(row=row, column=1, sticky="w", pady=4)
            controls.append(entry)
        ttk.Label(settings, text="Automatic pickup identity uses task goals, then saved grasp targets. You can override it above.\n"
                  "Collision excludes intended finger contacts with task/plan targets; forces are read at every physics step.\n"
                  "Far from plan is a distance warning, allowing a different route within the chosen corridor.\n"
                  "Task not finished is assessed at the policy step limit. A manual Stop or policy error remains unassessed.\n"
                  "The VLA continues after alerts. Settings apply to the next run.", wraplength=960).grid(
                      row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.settings += tuple(controls)

    def update(self, monitor, force=False):
        if monitor is None:
            return
        now = time.monotonic()
        if not force and monitor.state == "running" and now-self.last_update < .1:
            return
        self.last_update = now
        report = monitor.report()
        summary = (f"Visited {report['visited']}/{report['total']} · In order {report['ordered_prefix']}/{report['total']}"
                   if report['checking_waypoints'] and report['total'] else "Waypoint checking off / no saved targets")
        timeline = report['failure_timeline']
        if timeline is not None:
            events = timeline['events']
            summary += f" · Alerts {len(events)}"
            if events:
                first = events[0]
                seconds = first['confirmed_sim_seconds'] if first['onset_sim_seconds'] is None else first['onset_sim_seconds']
                summary += f" · first {seconds:.2f} s"
        self.summary.set(summary)
        if self.panel.busy:
            for row in report["rows"]:
                key = str(row["waypoint"]-1)
                if self.panel.rows.exists(key):
                    tag = "pose_visited" if row["first_visit_sim_seconds"] is not None else "pose_missed" if report["run_state"] == "finished" else ""
                    self.panel.rows.item(key, tags=(tag,) if tag else ())
        if self.window is None or not self.window.winfo_exists():
            return
        name = {"pi05": "Pi0.5"}.get(report['backend'], report['backend'])
        self.caption.set(f"{name} · {report['suite']} task {report['task_id']+1} · {report['language']}")
        angle = report["orientation_tolerance_degrees"]
        rule = f"Within {report['position_tolerance_cm']:g} cm" + (f" and {angle:g}°" if angle is not None else " · position only")
        outcome = "" if report["task_success"] is None else f" · Task success: {'yes' if report['task_success'] else 'no'}"
        self.details.set(summary + f" · {report['steps']} actions · {report['simulation_seconds']:.2f} sim s · {report['run_state']}"
                         + "\n" + rule + outcome)
        for control in self.settings:
            control.state(["disabled"] if monitor.state in ("preparing", "running") else ["!disabled"])
        existing = set(self.table.get_children())
        for row in report["rows"]:
            key = str(row["waypoint"])
            def number(value):
                return "—" if value is None else f"{value:.2f}"
            values = (f"{row['waypoint']:02d} {row['name']}", row["status"],
                      number(row["closest_position_cm"]), number(row["angle_at_closest_deg"]),
                      number(row["first_visit_sim_seconds"]), number(row["ordered_visit_sim_seconds"]))
            tags = ("visited",) if row["first_visit_sim_seconds"] is not None else ("missed",) if row["status"] == "Not visited" else ()
            if key in existing:
                self.table.item(key, values=values, tags=tags)
                existing.remove(key)
            else:
                self.table.insert("", "end", iid=key, values=values, tags=tags)
        for key in existing:
            self.table.delete(key)
        self.event_table.delete(*self.event_table.get_children())
        if timeline is None:
            self.event_status.set("Failure timing is off." if monitor.failure_config is None else
                "Failure timing could not start; see the main-window error." if monitor.state == "error" else
                "Failure timing is being prepared.")
            self.failure_rules.set("")
            return
        events = timeline['events']
        self.event_status.set(f"{len(events)} alert episodes · {timeline['assessment']}. Times are simulation seconds since VLA start.")
        expected = timeline['expected_objects']
        identity = "Unavailable: select an expected object" if expected is None else ", ".join(expected) or "No pickup expected"
        cfg = timeline['config']
        off = (f"Off-plan: >{100*cfg['off_plan_metres']:g} cm for {cfg['off_plan_seconds']:g} s."
               if timeline['off_plan_available'] else "Off-plan check unavailable: load a saved waypoint plan.")
        self.failure_rules.set(f"Expected pickup: {identity} ({timeline['expected_source']}).\n"
            f"Pickup: lift ≥{100*cfg['pickup_lift_metres']:g} cm for {cfg['pickup_hold_seconds']:g} s. "
            f"Collision: ≥{cfg['collision_force_n']:g} N, excluding intended finger contacts. {off}\n"
            "Onset is the first qualifying sample; confirmation can follow later. “Before start” means contact was already present. "
            f"Collisions rearm after {cfg['collision_quiet_seconds']:g} s below threshold; far-from-plan is a proxy.")
        for event in events:
            onset = "Before start" if event['onset_sim_seconds'] is None else f"{event['onset_sim_seconds']:.3f}"
            wall = "—" if event['confirmed_wall_seconds'] is None else f"{event['confirmed_wall_seconds']:.3f}"
            self.event_table.insert("", "end", values=(event['event'], onset,
                f"{event['confirmed_sim_seconds']:.3f}", wall, event['control_step'], event['detail']), tags=('alert',))
        if events and self.last_event_count == 0:
            self.event_evidence.set(events[0]['detail'])
            self.tabs.select(self.event_tab)
        self.last_event_count = len(events)

    def save(self):
        if self.monitor is None or self.monitor.tracker is None:
            return
        report = self.monitor.report()
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile=f"{report['suite']}_task_{report['task_id']:03d}_waypoint_coverage.csv",
            filetypes=[("CSV report", "*.csv")])
        if path:
            self.monitor.save_csv(path)

    def save_events(self):
        if self.monitor is None:
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile="vla_failure_times.csv", filetypes=[("Failure timeline", "*.csv")])
        if path:
            self.monitor.save_failures_csv(path)

    def save_json(self):
        if self.monitor is None:
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json",
            initialfile="vla_waypoints_and_failures.json", filetypes=[("Full report", "*.json")])
        if path:
            self.monitor.save_json(path)
