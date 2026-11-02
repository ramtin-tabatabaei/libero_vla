"""Presentation for the waypoint studio: readable controls and focused pickers."""
import tkinter as tk
import textwrap
from tkinter import font as tkfont, ttk

from waypoint_chain import WORLD, PREVIOUS, MAX_WAYPOINTS

BG = "#eef2f7"
PAPER = "#ffffff"
INK = "#172b45"
MUTED = "#586c85"
BLUE = "#245edb"
GREEN = "#087e70"
LINE = "#d9e2ed"
SUITE_LABELS = {"libero_spatial": "Spatial tasks", "libero_object": "Object tasks",
                "libero_goal": "Goal tasks", "libero_10": "Long tasks · 10", "libero_90": "LIBERO-90 · all 90 tasks"}
MOTIONS = {"Avoid obstacles": "reach", "Straight · allows contact": "line", "Joint arc": "arc"}
POLICY_LABELS = {"pi05": "Pi0.5"}


def friendly_name(name):
    if name == WORLD:
        return "Fixed position in the scene"
    if name == PREVIOUS:
        return "From the previous waypoint"
    text = name.replace("akita_black_bowl", "black_bowl").replace("_", " ")
    return text[:1].upper() + text[1:]


class StudioLayout:
    def build_workspace(self):
        root = self.host
        if not self.embedded:
            root.title("LIBERO | Waypoint Studio")
            root.geometry("1500x930")
            root.minsize(1240, 800)
        root.configure(style="TFrame") if self.embedded else root.configure(bg=BG)
        self.font_family = "DejaVu Sans"
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=self.font_family, size=11)
        root.option_add("*TCombobox*Listbox.font", (self.font_family, 11))
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=(self.font_family, 11), background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PAPER)
        style.configure("TLabel", background=PAPER, foreground=INK)
        style.configure("Muted.TLabel", foreground=MUTED, font=(self.font_family, 10))
        style.configure("Title.TLabel", font=(self.font_family, 19, "bold"))
        style.configure("Heading.TLabel", font=(self.font_family, 13, "bold"))
        style.configure("TButton", background="#f2f5fa", foreground=INK, bordercolor=LINE,
                        lightcolor="#f2f5fa", darkcolor="#f2f5fa", borderwidth=1, padding=(12, 9))
        style.map("TButton", background=[("active", "#e3ebf7")], foreground=[("disabled", "#8796a8")])
        for name, color, hover in (("Primary", BLUE, "#174cb9"), ("Run", GREEN, "#06695d"),
                                   ("Stop", "#b73846", "#962b38")):
            style.configure(name+".TButton", background=color, foreground="white", bordercolor=color,
                            lightcolor=color, darkcolor=color, font=(self.font_family, 11, "bold"))
            style.map(name+".TButton", background=[("disabled", "#dce3ed"), ("active", hover)],
                      foreground=[("disabled", "#738399"), ("!disabled", "white")])
        style.configure("TEntry", fieldbackground=PAPER, bordercolor=LINE, padding=7)
        style.configure("TCombobox", fieldbackground=PAPER, background=PAPER, bordercolor=LINE, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", PAPER)], foreground=[("readonly", INK)])
        style.configure("TCheckbutton", background=PAPER)
        style.configure("TNotebook", background=PAPER, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(17, 10), font=(self.font_family, 11))
        style.map("TNotebook.Tab", background=[("selected", "#eaf0ff")], foreground=[("selected", BLUE)])
        style.configure("Sequence.Treeview", background=PAPER, fieldbackground=PAPER, foreground=INK,
                        rowheight=52, borderwidth=0, font=(self.font_family, 11))
        style.map("Sequence.Treeview", background=[("selected", "#e8efff")], foreground=[("selected", "#17499c")])
        style.configure("Picker.Treeview", rowheight=48, background=PAPER, fieldbackground=PAPER,
                        font=(self.font_family, 11), borderwidth=0)
        style.configure("TaskPicker.Treeview", rowheight=72, background=PAPER, fieldbackground=PAPER,
                        font=(self.font_family, 11), borderwidth=0)
        style.configure("Relations.Treeview", rowheight=60, background=PAPER, fieldbackground=PAPER,
                        font=(self.font_family, 10), borderwidth=0)
        style.configure("Treeview.Heading", background="#f3f6fb", font=(self.font_family, 10, "bold"))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self.build_header()
        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        body.columnconfigure(0, weight=0, minsize=285)
        body.columnconfigure(1, weight=1, minsize=410)
        body.columnconfigure(2, weight=0, minsize=375)
        body.rowconfigure(0, weight=1)
        sequence = ttk.Frame(body, style="Card.TFrame", padding=14)
        sequence.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        scene = ttk.Frame(body, style="Card.TFrame", padding=16)
        scene.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        # Learning and editing are both about one waypoint plan, so they share
        # the right column; the left column is the plan and what runs it.
        right = ttk.Frame(body)
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        learning = ttk.Frame(right, style="Card.TFrame", padding=16)
        learning.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        editor = ttk.Frame(right, style="Card.TFrame", padding=16)
        editor.grid(row=1, column=0, sticky="nsew")
        self.build_sequence(sequence)
        self.build_scene_card(scene)
        self.build_learning_card(learning)
        self.build_edit_card(editor)
        self.build_status()

    def ui_button(self, parent, text, command, style="TButton", **kwargs):
        return ttk.Button(parent, text=text, command=lambda: self.safe(command), style=style, **kwargs)

    def build_header(self):
        bar = ttk.Frame(self.host, style="Card.TFrame", padding=(22, 15))
        self.header_frame = bar
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="Waypoint Studio", style="Title.TLabel").grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 24))
        self.goal = tk.StringVar(value="Loading your task…")
        self.task_caption = tk.StringVar(value="CURRENT TASK")
        ttk.Label(bar, textvariable=self.task_caption, style="Muted.TLabel").grid(row=0, column=1, sticky="w")
        self.task_title_label = ttk.Label(bar, textvariable=self.goal, wraplength=570)
        self.task_title_label.grid(row=1, column=1, sticky="w", pady=(4, 0))
        tasks = ttk.Frame(bar, style="Card.TFrame")
        tasks.grid(row=0, column=2, rowspan=2, padx=(14, 8))
        self.previous_task_button = self.ui_button(tasks, "‹", lambda: self.change_task_by(-1), width=2)
        self.previous_task_button.grid(row=0, column=0)
        self.change_task_button = self.ui_button(tasks, "Change task ▾", self.task_picker, "Primary.TButton")
        self.change_task_button.grid(row=0, column=1, padx=4)
        self.next_task_button = self.ui_button(tasks, "›", lambda: self.change_task_by(1), width=2)
        self.next_task_button.grid(row=0, column=2)
        self.save_plan_button = self.ui_button(bar, "Save plan", self.save_plan)
        self.save_plan_button.grid(row=0, column=3, rowspan=2)

    def build_sequence(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        header = ttk.Frame(parent, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Waypoints", style="Heading.TLabel").pack(side="left")
        self.count_text = tk.StringVar(value="0 steps")
        ttk.Label(header, textvariable=self.count_text, style="Muted.TLabel").pack(side="right")
        self.plan_status = tk.StringVar(value="Learn from pi0.5 · up to 9 steps")
        ttk.Label(parent, textvariable=self.plan_status, style="Muted.TLabel", wraplength=270).grid(row=1, column=0, sticky="w", pady=(5, 12))
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.rows = ttk.Treeview(frame, show="tree", selectmode="browse", style="Sequence.Treeview", height=6)
        self.rows.column("#0", width=238, minwidth=200, stretch=True)
        self.rows.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.rows.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.rows.configure(yscrollcommand=scroll.set)
        self.rows.tag_configure("running", background="#d9f3e9", foreground="#096650")
        self.rows.tag_configure("failed", background="#ffe5e7", foreground="#9c2938")
        self.rows.bind("<<TreeviewSelect>>", lambda _: self.safe(self.select))
        self.ui_button(parent, "+ Add a waypoint", self.new).grid(row=3, column=0, sticky="ew", pady=(12, 7))
        edits = ttk.Frame(parent, style="Card.TFrame")
        edits.grid(row=4, column=0, sticky="ew")
        for col, (label, cmd) in enumerate((("↑", lambda: self.reorder(-1)), ("↓", lambda: self.reorder(1)),
                                           ("Copy", self.duplicate), ("Delete", self.delete))):
            edits.columnconfigure(col, weight=1)
            self.ui_button(edits, label, cmd, width=3 if col < 2 else 5).grid(row=0, column=col, sticky="ew", padx=(0, 3))
        ttk.Separator(parent).grid(row=5, column=0, sticky="ew", pady=17)
        self.ui_button(parent, "▶  Run all waypoints", self.run, "Run.TButton").grid(row=6, column=0, sticky="ew")
        ttk.Label(parent, text="Runs all saved steps, continuing after failures.", wraplength=270, style="Muted.TLabel").grid(row=7, column=0, sticky="w", pady=(6, 9))
        self.ui_button(parent, "Reset robot & scene", self.reset).grid(row=8, column=0, sticky="ew")
        self.inject_button = self.ui_button(parent, "Inject failures…", self.failure_picker)
        self.inject_button.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        self.failure_summary = tk.StringVar(value="No failures injected.")
        ttk.Label(parent, textvariable=self.failure_summary, style="Muted.TLabel",
                  wraplength=270).grid(row=10, column=0, sticky="w", pady=(6, 0))

    def build_learning_card(self, parent):
        """Recording and fitting a plan, beside the plan being edited."""
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="Learn from the VLA", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="VLA demos → object-relative plan" if self.gmm_mode else "Best run → one editable plan",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 13))
        settings = ttk.Frame(parent, style="Card.TFrame")
        settings.grid(row=2, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)
        self.learn_repeats = tk.StringVar(value="5" if self.gmm_mode else "2")
        self.learn_waypoints = tk.StringVar(value="Auto")
        ttk.Label(settings, text="Successful demos" if self.gmm_mode else "Learning runs").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=1, to=100, textvariable=self.learn_repeats,
                    width=6).grid(row=0, column=1, sticky="e", pady=3)
        ttk.Label(settings, text="Waypoints").grid(row=1, column=0, sticky="w")
        ttk.Combobox(settings, textvariable=self.learn_waypoints, state="readonly",
                     values=["Auto"] if self.gmm_mode else ["Auto"] + list(range(1, MAX_WAYPOINTS+1)),
                     width=6).grid(row=1, column=1, sticky="e", pady=3)
        self.learn_button = self.ui_button(parent, "Learn waypoints · VLA" if self.gmm_mode else "Learn waypoints · pi0.5", self.learn, "Primary.TButton")
        self.learn_button.grid(row=3, column=0, sticky="ew", pady=(13, 0))

    def build_scene_card(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        header = ttk.Frame(parent, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Robot & scene", style="Heading.TLabel").pack(side="left")
        self.robot_state = tk.StringVar(value="●  Ready")
        ttk.Label(header, textvariable=self.robot_state, foreground=GREEN).pack(side="right")
        ttk.Label(parent, text="Mouse wheel to zoom · drag to pan", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 14))
        tabs = self.view_tabs = ttk.Notebook(parent)
        tabs.grid(row=2, column=0, sticky="nsew")
        camera_tab = ttk.Frame(tabs, style="Card.TFrame")
        relations_tab = ttk.Frame(tabs, style="Card.TFrame", padding=(0, 10))
        tabs.add(camera_tab, text="Live camera")
        tabs.add(relations_tab, text="Waypoint relationships")
        camera_tab.columnconfigure(0, weight=1)
        camera_tab.rowconfigure(0, weight=1)
        self.camera = tk.Canvas(camera_tab, bg="#e9eef4", highlightthickness=0, width=370, height=450)
        self.camera.grid(row=0, column=0, sticky="nsew")
        self.camera.bind("<Configure>", lambda _: self.paint_camera())
        self.camera.bind("<MouseWheel>", self.camera_wheel)
        self.camera.bind("<Button-4>", self.camera_wheel)
        self.camera.bind("<Button-5>", self.camera_wheel)
        self.camera.bind("<Button-1>", self.camera_pan_start)
        self.camera.bind("<B1-Motion>", self.camera_pan_drag)
        controls = ttk.Frame(camera_tab, style="Card.TFrame")
        controls.grid(row=0, column=1, sticky="ne", padx=(10, 5), pady=6)
        controls.columnconfigure((0, 1, 2), weight=1)
        self.camera_zoom_text = tk.StringVar(value="100%")
        ttk.Label(controls, text="Zoom", style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(controls, textvariable=self.camera_zoom_text, width=5, anchor="e").grid(row=0, column=2, sticky="e")
        self.zoom_out_button = ttk.Button(controls, text="−", width=2, padding=(4, 3), command=lambda: self.zoom_camera(.8))
        self.zoom_out_button.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.zoom_in_button = ttk.Button(controls, text="+", width=2, padding=(4, 3), command=lambda: self.zoom_camera(1.25))
        self.zoom_in_button.grid(row=1, column=1, sticky="ew", padx=3, pady=(5, 0))
        self.zoom_fit_button = ttk.Button(controls, text="Fit", width=3, padding=(4, 3), command=self.reset_camera_zoom)
        self.zoom_fit_button.grid(row=1, column=2, sticky="ew", pady=(5, 0))
        ttk.Label(controls, text="Target speed").grid(row=2, column=0, columnspan=3, sticky="w", pady=(14, 5))
        self.playback_speed = tk.StringVar(value=f"{self.session.playback_rate:g}×")
        self.speed_selector = ttk.Combobox(controls, textvariable=self.playback_speed,
                                         values=("0.5×", "1×", "2×", "5×", "10×"),
                                         state="readonly", width=5)
        self.speed_selector.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.speed_selector.bind("<<ComboboxSelected>>", self.change_playback_speed)
        self.camera_controls = (self.zoom_out_button, self.zoom_in_button,
                                self.zoom_fit_button, self.speed_selector)
        self.actual_speed = tk.StringVar(value="Actual: —")
        ttk.Label(controls, textvariable=self.actual_speed, style="Muted.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        relations_tab.columnconfigure(0, weight=1)
        relations_tab.rowconfigure(1, weight=1)
        ttk.Label(relations_tab, text="Each target is defined relative to the reference below.",
                  style="Muted.TLabel", wraplength=380).grid(row=0, column=0, sticky="w", pady=(0, 10))
        relation_frame = ttk.Frame(relations_tab, style="Card.TFrame")
        relation_frame.grid(row=1, column=0, sticky="nsew")
        relation_frame.columnconfigure(0, weight=1)
        relation_frame.rowconfigure(0, weight=1)
        self.relationship_rows = ttk.Treeview(relation_frame, columns=("waypoint", "reference"),
            show="headings", selectmode="browse", style="Relations.Treeview", height=6)
        self.relationship_rows.heading("waypoint", text="Waypoint")
        self.relationship_rows.heading("reference", text="Reference · offset (cm)")
        self.relationship_rows.column("waypoint", width=100, minwidth=90, stretch=False)
        self.relationship_rows.column("reference", width=310, minwidth=250, stretch=True)
        self.relationship_rows.grid(row=0, column=0, sticky="nsew")
        relation_scroll = ttk.Scrollbar(relation_frame, command=self.relationship_rows.yview)
        relation_scroll.grid(row=0, column=1, sticky="ns")
        relation_xscroll = ttk.Scrollbar(relation_frame, orient="horizontal", command=self.relationship_rows.xview)
        relation_xscroll.grid(row=1, column=0, sticky="ew")
        self.relationship_rows.configure(yscrollcommand=relation_scroll.set, xscrollcommand=relation_xscroll.set)
        self.relationship_rows.tag_configure("edited", background="#fff3db", foreground="#945500")
        self.relationship_rows.tag_configure("running", background="#d9f3e9", foreground="#096650")
        self.relationship_rows.tag_configure("failed", background="#ffe5e7", foreground="#9c2938")
        self.relationship_rows.bind("<<TreeviewSelect>>", lambda _: self.safe(self.select_relationship))
        ttk.Label(relations_tab, text="Select a row to edit its target. Changes appear here as you type.",
                  style="Muted.TLabel", wraplength=380).grid(row=2, column=0, sticky="w", pady=10)
        self.pose_label = tk.StringVar(value="Gripper position appears after loading.")
        ttk.Label(parent, textvariable=self.pose_label, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=15)
        if self.gmm_mode:
            scene_actions = ttk.Frame(parent, style="Card.TFrame")
            scene_actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
            scene_actions.columnconfigure(0, weight=1)
            self.shuffle_button = self.ui_button(scene_actions, "Shuffle task", self.shuffle_scene)
            self.shuffle_button.grid(row=0, column=0, sticky="ew")

        robot_start = ttk.Frame(parent, style="Card.TFrame")
        robot_start.grid(row=5 if self.gmm_mode else 4, column=0, sticky="ew", pady=(8, 0))
        robot_start.columnconfigure(0, weight=1)
        self.random_robot_start = tk.BooleanVar(value=False)
        self.random_start_check = ttk.Checkbutton(robot_start, text="Random start on load/reset",
                                                  variable=self.random_robot_start)
        self.random_start_check.grid(row=0, column=0, sticky="w")
        self.random_start_button = self.ui_button(robot_start, "New robot start", self.new_robot_start)
        self.random_start_button.grid(row=0, column=1, padx=(6, 0))

        policy_actions = ttk.Frame(parent, style="Card.TFrame")
        policy_actions.grid(row=6 if self.gmm_mode else 5, column=0, sticky="ew", pady=(10, 0))
        policy_actions.columnconfigure(0, weight=1)
        choice = ttk.Frame(policy_actions, style="Card.TFrame")
        choice.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        choice.columnconfigure(1, weight=1)
        ttk.Label(choice, text="Policy").grid(row=0, column=0, padx=(0, 10))
        self.policy_choice = tk.StringVar(value=POLICY_LABELS[self.args.policy_backend])
        self.policy_selector = ttk.Combobox(choice, textvariable=self.policy_choice,
                                            values=list(POLICY_LABELS.values()), state="readonly")
        self.policy_selector.grid(row=0, column=1, sticky="ew")
        self.policy_selector.bind("<<ComboboxSelected>>", self.change_policy)
        self.run_policy_button = self.ui_button(policy_actions, "▶  Run policy", self.run_policy, "Run.TButton")
        self.run_policy_button.grid(row=1, column=0, sticky="ew")
        self.policy_hint = tk.StringVar()
        ttk.Label(policy_actions, textvariable=self.policy_hint,
                  style="Muted.TLabel", wraplength=380).grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.change_policy()
        from waypoint_coverage_ui import CoverageView
        self.policy_coverage = CoverageView(self, policy_actions)

    def build_edit_card(self, parent):
        outer = parent
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        scroller = tk.Canvas(outer, width=335, height=480, bg=PAPER, highlightthickness=0)
        scroller.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=scroller.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        scroller.configure(yscrollcommand=scrollbar.set)
        parent = ttk.Frame(scroller, style="Card.TFrame")
        content = scroller.create_window((0, 0), window=parent, anchor="nw")
        scroller.bind("<Configure>", lambda event: scroller.itemconfigure(content, width=event.width))
        parent.bind("<Configure>", lambda _: scroller.configure(scrollregion=scroller.bbox("all")))
        def wheel(event):
            x, y = event.x_root, event.y_root
            if (scroller.winfo_rootx() <= x < scroller.winfo_rootx()+scroller.winfo_width()
                    and scroller.winfo_rooty() <= y < scroller.winfo_rooty()+scroller.winfo_height()):
                direction = -1 if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0 else 1
                scroller.yview_scroll(direction*3, "units")
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind(sequence, wheel, add="+")
        parent.columnconfigure(0, weight=1)
        self.editor_title = tk.StringVar(value="New waypoint")
        ttk.Label(parent, textvariable=self.editor_title, style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="Adjust the target, then save or move.", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 15))
        self.name = tk.StringVar()
        ttk.Label(parent, text="Waypoint name").grid(row=2, column=0, sticky="w", pady=(0, 5))
        ttk.Entry(parent, textvariable=self.name).grid(row=3, column=0, sticky="ew")
        ttk.Label(parent, text="Position relative to").grid(row=4, column=0, sticky="w", pady=(14, 5))
        self.reference = tk.StringVar(value=WORLD)
        self.reference_label = tk.StringVar(value=friendly_name(WORLD))
        self.ui_button(parent, "", self.object_picker, textvariable=self.reference_label).grid(row=5, column=0, sticky="ew")
        self.reference.trace_add("write", lambda *_: self.reference_label.set(friendly_name(self.reference.get()) + "  ▾"))
        presets = ttk.Frame(parent, style="Card.TFrame")
        presets.grid(row=6, column=0, sticky="ew", pady=(7, 14))
        presets.columnconfigure(0, weight=1)
        presets.columnconfigure(1, weight=1)
        self.ui_button(presets, "Above object", self.above).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.ui_button(presets, "Use robot position", self.capture).grid(row=0, column=1, sticky="ew")
        self.ui_button(presets, "Update references from scene", self.refresh_references).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(parent, text="Target offset  ·  centimetres", style="Muted.TLabel").grid(row=7, column=0, sticky="w", pady=(0, 7))
        position = ttk.Frame(parent, style="Card.TFrame")
        position.grid(row=8, column=0, sticky="ew")
        position.columnconfigure(2, weight=1)
        self.xyz = [tk.StringVar(value="0.00") for _ in range(3)]
        for row, (axis, color) in enumerate((("X", "#b44152"), ("Y", "#09816b"), ("Z", BLUE))):
            ttk.Label(position, text=axis, foreground=color, font=(self.font_family, 12, "bold"), width=3).grid(row=row, column=0)
            self.ui_button(position, "−", lambda a=row: self.nudge(a, -.01), width=2).grid(row=row, column=1, padx=(0, 5), pady=3)
            ttk.Entry(position, textvariable=self.xyz[row], justify="center", width=9).grid(row=row, column=2, sticky="ew")
            self.ui_button(position, "+", lambda a=row: self.nudge(a, .01), width=2).grid(row=row, column=3, padx=(5, 0))
        self.hold_xy = tk.BooleanVar()
        ttk.Checkbutton(position, text="Vertical move · use previous X/Y", variable=self.hold_xy).grid(row=3, column=0, columnspan=4, sticky="w", pady=(5, 0))
        ttk.Label(parent, text="+ / − adjusts the target by 1 cm.\nFollow the matching arrow in the scene.\nNegative offsets go the opposite way.", style="Muted.TLabel").grid(row=9, column=0, sticky="w", pady=(5, 10))
        ttk.Label(parent, text="Target orientation  ·  world degrees", style="Muted.TLabel").grid(row=10, column=0, sticky="w", pady=(4, 7))
        rotation = ttk.Frame(parent, style="Card.TFrame")
        rotation.grid(row=11, column=0, sticky="ew")
        rotation.columnconfigure(2, weight=1)
        self.rpy = [tk.StringVar(value="0") for _ in range(3)]
        self.roll, self.pitch, self.yaw = self.rpy
        for row, (axis, color) in enumerate((("Roll / X", "#b44152"), ("Pitch / Y", "#09816b"), ("Yaw / Z", BLUE))):
            ttk.Label(rotation, text=axis, foreground=color, width=9).grid(row=row, column=0, sticky="w")
            self.ui_button(rotation, "−", lambda a=row: self.nudge_rotation(a, -1), width=2).grid(row=row, column=1, padx=(0, 5), pady=3)
            ttk.Entry(rotation, textvariable=self.rpy[row], justify="center", width=9).grid(row=row, column=2, sticky="ew")
            self.ui_button(rotation, "+", lambda a=row: self.nudge_rotation(a, 1), width=2).grid(row=row, column=3, padx=(5, 0))
        ttk.Label(parent, text="Roll, pitch and yaw of the target in the scene.\nChanges rotate the waypoint's colored axes.\n+ / − adjusts by 1°. Save to keep the angles.",
                  style="Muted.TLabel", wraplength=325).grid(row=12, column=0, sticky="w", pady=(5, 10))
        self.motion = tk.StringVar(value="Avoid obstacles")
        self.grip = tk.StringVar(value="keep")
        self.grip_display = tk.StringVar(value="Keep unchanged")
        grip_labels = {"keep": "Keep unchanged", "open": "Open", "close": "Close"}
        self.grip.trace_add("write", lambda *_: self.grip_display.set(grip_labels[self.grip.get()]))
        for row, title, var, values in ((13, "Movement", self.motion, list(MOTIONS)),
                                        (15, "Gripper before this move", self.grip_display, list(grip_labels.values()))):
            ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", pady=(4, 5))
            choice = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
            choice.grid(row=row+1, column=0, sticky="ew", pady=(0, 5))
            if var is self.grip_display:
                choice.bind("<<ComboboxSelected>>", lambda _: self.grip.set(next(k for k, v in grip_labels.items() if v == self.grip_display.get())))
        self.carry = tk.BooleanVar()
        self.grip_after = tk.StringVar(value="keep")
        self.grip_after_display = tk.StringVar(value="Keep unchanged")
        self.grip_after.trace_add("write", lambda *_: self.grip_after_display.set(grip_labels[self.grip_after.get()]))
        ttk.Label(parent, text="Gripper on arrival").grid(row=17, column=0, sticky="w", pady=(4, 5))
        after = ttk.Combobox(parent, textvariable=self.grip_after_display, values=list(grip_labels.values()), state="readonly")
        after.grid(row=18, column=0, sticky="ew", pady=(0, 5))
        after.bind("<<ComboboxSelected>>", lambda _: self.grip_after.set(next(k for k, v in grip_labels.items() if v == self.grip_after_display.get())))
        self.ui_button(parent, "Placement options…", self.advanced_options).grid(row=19, column=0, sticky="ew", pady=(7, 10))
        footer = ttk.Frame(outer, style="Card.TFrame")
        footer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        self.draft_hint = tk.StringVar(value="Editing changes the target only.")
        ttk.Label(footer, textvariable=self.draft_hint, style="Muted.TLabel", wraplength=335).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.save_waypoint_button = self.ui_button(footer, "Save new waypoint", self.save_waypoint, "Primary.TButton")
        self.save_waypoint_button.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.move_button = self.ui_button(footer, "▶  Move robot here", self.test, "Run.TButton")
        self.move_button.grid(row=2, column=0, sticky="ew")
        ttk.Label(footer, text="Same target on every click. Update references\nonly when you want to use new object poses.", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(7, 0))

    def build_status(self):
        bar = ttk.Frame(self.host, style="Card.TFrame", padding=(20, 12))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="Loading scene…")
        self.note = tk.StringVar(value="")
        self.save_state = tk.StringVar(value="No pending changes")
        ttk.Label(bar, textvariable=self.status, wraplength=790).grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.save_state, style="Muted.TLabel").grid(row=0, column=1, padx=20)
        self.stop_button = ttk.Button(bar, text="■  Stop robot  /  Esc", command=self.stop, style="Stop.TButton")
        self.stop_button.grid(row=0, column=2)

    def picker_window(self, title, width=740, height=620):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=PAPER)
        dialog.geometry(f"{width}x{height}+{self.root.winfo_rootx()+140}+{self.root.winfo_rooty()+90}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        return dialog

    def task_picker(self):
        from libero.libero import benchmark
        dialog = self.picker_window("Choose a task", width=1000, height=650)
        self.task_dialog = dialog
        dialog.rowconfigure(3, weight=1)
        ttk.Label(dialog, text="Choose a task", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=22, pady=20)
        filters = ttk.Frame(dialog, style="Card.TFrame")
        filters.grid(row=1, column=0, sticky="ew", padx=22)
        filters.columnconfigure(1, weight=1)
        suite_var = tk.StringVar(value=SUITE_LABELS[self.suite.get()])
        self.task_filter_suite = suite_var
        suite_box = ttk.Combobox(filters, textvariable=suite_var, values=list(SUITE_LABELS.values()), state="readonly", width=21)
        suite_box.grid(row=0, column=0, padx=(0, 12))
        search = tk.StringVar()
        self.task_filter_search = search
        entry = ttk.Entry(filters, textvariable=search)
        entry.grid(row=0, column=1, sticky="ew")
        options = ttk.Frame(dialog, style="Card.TFrame")
        options.grid(row=2, column=0, sticky="ew", padx=22, pady=10)
        ready_only = tk.BooleanVar(value=False)
        self.task_filter_ready = ready_only
        ttk.Checkbutton(options, text="Only show tasks I have saved", variable=ready_only).pack(side="left")
        summary = tk.StringVar()
        ttk.Label(options, textvariable=summary, style="Muted.TLabel").pack(side="right")
        rows = ttk.Treeview(dialog, show="tree headings", columns=("status", "steps", "saved"), style="TaskPicker.Treeview", selectmode="browse")
        self.task_picker_rows = rows
        rows.heading("#0", text="Task")
        rows.heading("status", text="Plan status")
        rows.heading("steps", text="Waypoints")
        rows.heading("saved", text="Saved")
        rows.column("#0", width=500, minwidth=360)
        rows.column("status", width=200, stretch=False)
        rows.column("steps", width=80, stretch=False, anchor="center")
        rows.column("saved", width=90, stretch=False, anchor="center")
        # Learning leaves a gray draft; green requires an explicit Save plan.
        rows.tag_configure("saved", foreground=GREEN)
        rows.tag_configure("review", foreground=MUTED)
        rows.tag_configure("empty", foreground=MUTED)
        rows.grid(row=3, column=0, sticky="nsew", padx=(22, 36))
        scroll = ttk.Scrollbar(dialog, command=rows.yview)
        scroll.grid(row=3, column=0, sticky="nse", padx=(0, 18))
        rows.configure(yscrollcommand=scroll.set)
        def fill(*_):
            selection = rows.selection()
            rows.delete(*rows.get_children())
            suite = next(k for k, v in SUITE_LABELS.items() if v == suite_var.get())
            benchmark_suite = benchmark.get_benchmark_dict()[suite]()
            saved_count = 0
            for i in range(benchmark_suite.n_tasks):
                language = benchmark_suite.get_task(i).language
                status, count, ready = self.task_plan_summary(suite, i)
                saved = self.task_saved(suite, i)
                saved_count += int(saved)
                if (search.get().lower() in f"{i+1:02d} {language}".lower()
                        and (not ready_only.get() or saved)):
                    words = textwrap.fill(language, width=55, subsequent_indent="      ")
                    tag = "saved" if saved else "review" if count else "empty"
                    rows.insert("", "end", iid=str(i), text=f"{i+1:02d}   {words}",
                                values=(status, str(count) if count else "—",
                                        "saved" if saved else "—"), tags=(tag,))
            summary.set(f"{saved_count} / {benchmark_suite.n_tasks} saved")
            current = selection[0] if selection else str(self.task_id) if suite == self.suite.get() else "0"
            if rows.exists(current):
                rows.selection_set(current)
                rows.see(current)
        def choose():
            selected = rows.selection()
            if not selected:
                return
            self.suite.set(next(k for k, v in SUITE_LABELS.items() if v == suite_var.get()))
            self.task_id = int(selected[0])
            dialog.destroy()
            self.safe(self.load_task)
        def learn_all():
            suite = next(k for k, v in SUITE_LABELS.items() if v == suite_var.get())
            dialog.destroy()
            self.safe(lambda: self.learn_all_tasks(suite))
        search.trace_add("write", fill)
        ready_only.trace_add("write", fill)
        suite_box.bind("<<ComboboxSelected>>", fill)
        rows.bind("<Double-Button-1>", lambda _: choose())
        dialog.bind("<Return>", lambda _: choose())
        footer = ttk.Frame(dialog, style="Card.TFrame")
        footer.grid(row=4, column=0, sticky="ew", padx=22, pady=20)
        ttk.Button(footer, text="Refresh build status", command=fill).pack(side="left")
        self.load_selected_task_button = ttk.Button(footer, text="Load selected task", command=choose, style="Primary.TButton")
        self.load_selected_task_button.pack(side="right")
        self.learn_all_tasks_button = ttk.Button(footer, text="Learn all tasks' waypoints", command=learn_all, style="Primary.TButton")
        self.learn_all_tasks_button.pack(side="right", padx=(0, 10))
        ttk.Label(dialog, text="Learn all keeps learned plans gray. Click Save plan yourself to mark a task green.",
                  style="Muted.TLabel", wraplength=950).grid(row=5, column=0, sticky="w", padx=22, pady=(0, 14))
        fill()
        entry.focus_set()

    def object_picker(self):
        self.require_scene()
        dialog = self.picker_window("Choose a target reference", 660, 640)
        dialog.rowconfigure(4, weight=1)
        ttk.Label(dialog, text="Relative to what?", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=22, pady=(20, 10))
        ttk.Label(dialog, text="Choose an object to make the waypoint follow its position.", style="Muted.TLabel").grid(row=1, column=0, sticky="w", padx=22)
        search = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=search)
        entry.grid(row=2, column=0, sticky="ew", padx=22, pady=15)
        advanced = tk.BooleanVar()
        ttk.Checkbutton(dialog, text="Include regions and joints", variable=advanced).grid(row=3, column=0, sticky="w", padx=22, pady=(0, 10))
        rows = ttk.Treeview(dialog, show="tree", style="Picker.Treeview", selectmode="browse")
        rows.grid(row=4, column=0, sticky="nsew", padx=(22, 36))
        scroll = ttk.Scrollbar(dialog, command=rows.yview)
        scroll.grid(row=4, column=0, sticky="nse", padx=(0, 18))
        rows.configure(yscrollcommand=scroll.set)
        relevant = {name for goal in self.session.world.domain.parsed_problem["goal_state"] for name in goal[1:]}
        domain = self.session.world.domain
        def fill(*_):
            rows.delete(*rows.get_children())
            names = list(domain.objects_dict) + list(domain.fixtures_dict)
            if advanced.get():
                names = list(self.session.poses())
            if self.reference.get() not in (WORLD, PREVIOUS) and self.reference.get() not in names:
                names.append(self.reference.get())
            names = [WORLD, PREVIOUS] + sorted(set(names), key=lambda n: (n not in relevant, friendly_name(n)))
            for name in names:
                if search.get().lower() in friendly_name(name).lower():
                    suffix = "  ·  task target" if name in relevant else ""
                    rows.insert("", "end", iid=name, text=friendly_name(name)+suffix)
        def choose():
            selected = rows.selection()
            if selected:
                self.reference.set(selected[0])
                dialog.destroy()
                self.safe(self.change_reference)
                self.mark_draft_changed()
        search.trace_add("write", fill)
        advanced.trace_add("write", fill)
        rows.bind("<Double-Button-1>", lambda _: choose())
        ttk.Button(dialog, text="Use selected reference", command=choose, style="Primary.TButton").grid(row=5, column=0, sticky="e", padx=22, pady=20)
        fill()
        entry.focus_set()

    def advanced_options(self):
        dialog = self.picker_window("Placement options", 510, 230)
        ttk.Label(dialog, text="Placement options", style="Heading.TLabel").grid(row=0, column=0, sticky="w", padx=22, pady=18)
        ttk.Checkbutton(dialog, text="Position the held object at this target", variable=self.carry).grid(row=1, column=0, sticky="w", padx=22, pady=10)
        ttk.Label(dialog, text="Use after a verified grasp. Adjust wrist angles in Target orientation in the editor.", style="Muted.TLabel", wraplength=455).grid(row=2, column=0, sticky="w", padx=22)
        ttk.Button(dialog, text="Done", command=dialog.destroy, style="Primary.TButton").grid(row=3, column=0, sticky="e", padx=22, pady=16)

    def failure_picker(self):
        """Arm one deliberate failure per waypoint, for Run all waypoints only."""
        self.require_scene()
        from waypoint_failures import (FAILURES, HINTS, INTENSITIES, allowed_intensities,
                                       failure_spec, waypoint_primitive)
        labels = dict([("", "No failure")] + list(FAILURES.items()))
        eligible = []
        for i, waypoint in enumerate(self.chain.waypoints):
            primitive = waypoint_primitive(waypoint)
            available = {key: name for key, name in labels.items()
                         if not key or allowed_intensities(key, primitive, waypoint.motion)}
            if len(available) > 1:
                eligible.append((i, waypoint, primitive, available))
        supported = {key for _, _, _, available in eligible for key in available if key}
        dialog = self.picker_window("Inject failures", 760, 680)
        dialog.rowconfigure(3, weight=1)
        ttk.Label(dialog, text="Inject failures", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=22, pady=(20, 8))
        ttk.Label(dialog, text="Choose a failure and its intensity, then click Apply failures and Run all waypoints. "
                               "Only waypoints with available failures are listed. "
                               "Transport only supports High Grasp loss: dropping the held object. "
                               "These choices apply to this session only.",
                  style="Muted.TLabel", wraplength=700).grid(row=1, column=0, sticky="w", padx=22)
        ttk.Label(dialog, text="\n".join(f"{name} \u2014 {HINTS[key]}." for key, name in FAILURES.items()
                                        if key in supported),
                  style="Muted.TLabel", wraplength=700).grid(row=2, column=0, sticky="w", padx=22, pady=(12, 4))
        listing = self.scrolled_list(dialog, row=3)
        listing.columnconfigure(1, weight=1)
        keys = {name: key for key, name in labels.items()}
        choices, intensities = {}, {}
        ttk.Label(listing, text="Waypoint", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(listing, text="Failure", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=12)
        ttk.Label(listing, text="Intensity", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        for display_row, (i, waypoint, primitive, available) in enumerate(eligible, 1):
            role = getattr(waypoint, "skill_role", "") or friendly_name(waypoint.name)
            ttk.Label(listing, text=f"{i+1:02d}   {role.title()}").grid(row=display_row, column=0, sticky="w", pady=5, padx=(2, 0))
            selection = self.injections.get(i)
            kind, intensity = failure_spec(selection) if selection else ("", "low")
            if kind not in available or intensity not in allowed_intensities(kind, primitive, waypoint.motion):
                kind, intensity = "", "low"
            choices[i] = tk.StringVar(value=labels[kind])
            intensities[i] = tk.StringVar(value=INTENSITIES[intensity])
            box = ttk.Combobox(listing, textvariable=choices[i], values=list(available.values()),
                               state="readonly", width=24)
            box.grid(row=display_row, column=1, sticky="e", padx=(12, 16), pady=5)
            strength = ttk.Combobox(listing, textvariable=intensities[i], width=8, state="disabled")
            strength.grid(row=display_row, column=2, sticky="e", pady=5)
            def update_intensity(*_, row=i, phase=primitive, motion=waypoint.motion, widget=strength):
                selected = keys[choices[row].get()]
                allowed = allowed_intensities(selected, phase, motion) if selected else ()
                widget.configure(values=[INTENSITIES[level] for level in allowed],
                                 state="readonly" if allowed else "disabled")
                if allowed and intensities[row].get() not in [INTENSITIES[level] for level in allowed]:
                    intensities[row].set(INTENSITIES[allowed[0]])
            choices[i].trace_add("write", update_intensity)
            update_intensity()
        if not choices:
            ttk.Label(listing, text="No failures are available for these waypoints.",
                      style="Muted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=18)
        def apply():
            levels = {name: key for key, name in INTENSITIES.items()}
            self.injections = {}
            for i, var in choices.items():
                kind, intensity = keys[var.get()], levels[intensities[i].get()]
                waypoint = self.chain.waypoints[i]
                if kind and intensity in allowed_intensities(kind, waypoint_primitive(waypoint), waypoint.motion):
                    self.injections[i] = dict(kind=kind, intensity=intensity)
            dialog.destroy()
            self.redraw_rows(self.editing_index)
            self.status.set(self.failure_summary.get())
        footer = ttk.Frame(dialog, style="Card.TFrame")
        footer.grid(row=4, column=0, sticky="ew", padx=22, pady=18)
        ttk.Button(footer, text="Clear all",
                   command=lambda: [var.set(labels[""]) for var in choices.values()]).pack(side="left")
        ttk.Button(footer, text="Apply failures", command=apply, style="Primary.TButton").pack(side="right")

    def scrolled_list(self, dialog, row):
        """A scrolling column inside a picker; returns the frame rows go in."""
        body = ttk.Frame(dialog, style="Card.TFrame")
        body.grid(row=row, column=0, sticky="nsew", padx=22)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        scroller = tk.Canvas(body, bg=PAPER, highlightthickness=0)
        scroller.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(body, orient="vertical", command=scroller.yview)
        bar.grid(row=0, column=1, sticky="ns")
        scroller.configure(yscrollcommand=bar.set)
        listing = ttk.Frame(scroller, style="Card.TFrame")
        item = scroller.create_window((0, 0), window=listing, anchor="nw")
        scroller.bind("<Configure>", lambda event: scroller.itemconfigure(item, width=event.width))
        listing.bind("<Configure>", lambda _: scroller.configure(scrollregion=scroller.bbox("all")))
        return listing

    def plan_tools(self):
        dialog = self.picker_window("Plan tools", 460, 400)
        ttk.Label(dialog, text="Plan tools", style="Heading.TLabel").grid(row=0, column=0, sticky="w", padx=22, pady=20)
        def action(fn):
            dialog.destroy()
            self.safe(fn)
        for row, (text, command) in enumerate((("Learn waypoints · GMM/GMR" if self.gmm_mode else "Learn waypoints · pi0.5", self.learn),
                                               ("Open a saved plan…", self.import_chain),
                                               ("Save a copy as…", self.export_chain)), 1):
            ttk.Button(dialog, text=text, command=lambda f=command: action(f)).grid(row=row, column=0, sticky="ew", padx=22, pady=7)
