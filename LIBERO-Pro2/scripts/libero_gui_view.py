"""Readable, neutral desktop layout for the multi-benchmark task explorer."""
import tkinter as tk
from tkinter import font as tkfont, ttk
from PIL import Image, ImageDraw, ImageTk

BG = "#f4f3ef"
PANEL = "#ffffff"
CARD = "#eeede8"
TEXT = "#242622"
MUTED = "#62665f"
ACCENT = "#276749"
RED = "#b43b3b"
EDGE = "#d4d6cd"


class ExplorerView:
    def style(self):
        self.root.tk.call("tk", "scaling", 96 / 72)
        self.fonts = {
            "body": tkfont.Font(family="DejaVu Sans", size=16),
            "small": tkfont.Font(family="DejaVu Sans", size=14),
            "section": tkfont.Font(family="DejaVu Sans", size=14, weight="bold"),
            "title": tkfont.Font(family="DejaVu Sans", size=24, weight="bold"),
            "instruction": tkfont.Font(family="DejaVu Sans", size=20, weight="bold"),
            "code": tkfont.Font(family="DejaVu Sans Mono", size=14),
        }
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family="DejaVu Sans", size=16)
        self.root.title("Robot Task Explorer · LIBERO-Pro2 & LIBERO-Max")
        width, height = min(1940, self.root.winfo_screenwidth()-60), min(1300, self.root.winfo_screenheight()-100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1320, 880)
        self.root.configure(bg=BG)
        for option, value in [("background", PANEL), ("foreground", TEXT), ("selectBackground", "#deeddf"),
                              ("selectForeground", TEXT), ("font", self.fonts["body"])]:
            self.root.option_add("*TCombobox*Listbox."+option, value)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, bordercolor=EDGE, font=self.fonts["body"])
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Panel.TLabel", background=PANEL)
        style.configure("Muted.TLabel", foreground=MUTED, font=self.fonts["small"])
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=self.fonts["small"])
        style.configure("Section.TLabel", background=PANEL, foreground=TEXT, font=self.fonts["section"])
        style.configure("Title.TLabel", font=self.fonts["title"])
        style.configure("Eyebrow.TLabel", foreground=ACCENT, font=self.fonts["section"])
        style.configure("TButton", background=PANEL, foreground=TEXT, padding=(16, 10), borderwidth=1)
        style.map("TButton", background=[("active", "#e3e9df"), ("disabled", CARD)], foreground=[("disabled", "#92968e")])
        style.configure("Run.TButton", background=ACCENT, foreground="white", font=self.fonts["section"], padding=(18, 14))
        style.map("Run.TButton", background=[("active", "#1f563b"), ("disabled", "#cbd4c9")], foreground=[("disabled", "#657461")])
        style.configure("Stop.TButton", background="#f8e9e7", foreground=RED)
        style.map("Stop.TButton", background=[("active", "#f0d2ce"), ("disabled", CARD)])
        style.configure("TCombobox", fieldbackground=PANEL, background=CARD, foreground=TEXT, arrowcolor=TEXT, padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL), ("disabled", CARD)], foreground=[("readonly", TEXT), ("disabled", MUTED)], selectbackground=[("readonly", PANEL)])
        style.configure("TSpinbox", fieldbackground=PANEL, background=CARD, foreground=TEXT, arrowcolor=TEXT, padding=8)
        style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, padding=8)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=MUTED, padding=(18, 12), font=self.fonts["small"])
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=EDGE, borderwidth=0)
        style.configure("Vertical.TScrollbar", background=EDGE, troughcolor=BG, arrowcolor=MUTED, borderwidth=0, arrowsize=18)
        style.configure("Conditions.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        font=self.fonts["small"], rowheight=34, bordercolor=EDGE)
        style.map("Conditions.Treeview", background=[("selected", "#deeddf")], foreground=[("selected", TEXT)])
        style.configure("Conditions.Treeview.Heading", background=CARD, foreground=TEXT, font=self.fonts["section"], padding=(4, 8))
        style.configure("Small.TButton", font=self.fonts["small"], padding=(8, 6))
        style.configure("Panel.TRadiobutton", background=PANEL, foreground=TEXT, font=self.fonts["small"])
        thumb = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(thumb)
        draw.rounded_rectangle((2, 3, 29, 28), radius=7, fill=ACCENT)
        for x in (12, 16, 20): draw.line((x, 10, x, 21), fill="white")
        self.slider_thumb = ImageTk.PhotoImage(thumb, master=self.root)
        style.element_create("TaskScale.slider", "image", self.slider_thumb)
        style.layout("Task.Horizontal.TScale", [("Horizontal.Scale.trough", {"sticky": "we", "children": [
            ("TaskScale.slider", {"side": "left", "sticky": ""})]})])
        style.configure("Task.Horizontal.TScale", background=PANEL, troughcolor=EDGE)

    def change_font_size(self, delta):
        target = min(22, max(14, self.fonts["body"].cget("size") + delta))
        self.fonts["body"].configure(size=target)
        for name in ("small", "section", "code"): self.fonts[name].configure(size=target-2)
        self.fonts["instruction"].configure(size=target+4)
        self.fonts["title"].configure(size=target+8)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(size=target)
        self.root.after_idle(self.draw_frames)

    def build(self):
        header = ttk.Frame(self.root, padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(header, text="Robot Task Explorer", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Connections", command=self.connections).pack(side="right")
        ttk.Button(header, text="A+", width=3, command=lambda: self.change_font_size(1)).pack(side="right", padx=(4, 12))
        ttk.Button(header, text="A−", width=3, command=lambda: self.change_font_size(-1)).pack(side="right")
        self.model_badge = ttk.Label(header, textvariable=self.model_health, style="Muted.TLabel")
        self.model_badge.pack(side="right", padx=18)
        self.sim_badge = ttk.Label(header, textvariable=self.sim_health, style="Muted.TLabel")
        self.sim_badge.pack(side="right", padx=8)
        body = ttk.Frame(self.root, padding=(20, 0, 20, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, style="Panel.TFrame", width=570)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 20))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)
        content = ttk.Frame(sidebar, style="Panel.TFrame", padding=18)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(5, weight=1)
        source = ttk.Frame(content, style="Panel.TFrame")
        source.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        source.columnconfigure(0, weight=3)
        source.columnconfigure(1, weight=2)
        ttk.Label(source, text="Benchmark", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(source, text="Suite", style="Section.TLabel").grid(row=0, column=1, sticky="w", pady=(0, 8), padx=(12, 0))
        benchmark = self.register(ttk.Combobox(source, textvariable=self.benchmark_var,
            values=["LIBERO-Pro2", "LIBERO-Max Lite", "LIBERO-Max"], state="readonly", width=21), "readonly")
        benchmark.grid(row=1, column=0, sticky="ew")
        benchmark.bind("<<ComboboxSelected>>", self.change_benchmark)
        suite = self.register(ttk.Combobox(source, textvariable=self.suite_var,
            values=["Spatial", "Object", "Goal", "Long horizon"], state="readonly", width=13), "readonly")
        suite.grid(row=1, column=1, sticky="ew", padx=(12, 0))
        suite.bind("<<ComboboxSelected>>", self.change_suite)
        heading = ttk.Frame(content, style="Panel.TFrame")
        heading.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(heading, text="Condition groups", style="Section.TLabel").pack(side="left")
        ttk.Button(heading, text="Collapse", style="Small.TButton", command=lambda: self.expand_condition_groups(False)).pack(side="right")
        ttk.Button(heading, text="Expand all", style="Small.TButton", command=lambda: self.expand_condition_groups(True)).pack(side="right", padx=6)
        find_condition = ttk.Frame(content, style="Panel.TFrame")
        find_condition.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(find_condition, text="Find condition", style="PanelMuted.TLabel").pack(side="left", padx=(0, 10))
        self.register(ttk.Entry(find_condition, textvariable=self.condition_query_var)).pack(side="left", fill="x", expand=True)
        ttk.Label(content, textvariable=self.condition_help_var, style="PanelMuted.TLabel", wraplength=520).grid(row=3, column=0, sticky="w", pady=(0, 10))
        self.timing_controls = ttk.Frame(content, style="Panel.TFrame")
        self.timing_controls.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        for name in ["All", "At start", "During task"]:
            self.register(ttk.Radiobutton(self.timing_controls, text=name, value=name,
                variable=self.condition_timing_var, style="Panel.TRadiobutton", command=self.refresh_condition_tree)).pack(side="left", padx=(0, 15))
        condition_box = ttk.Frame(content, style="Panel.TFrame")
        condition_box.grid(row=5, column=0, sticky="nsew")
        condition_box.columnconfigure(0, weight=1)
        condition_box.rowconfigure(0, weight=1)
        self.conditions = self.register(ttk.Treeview(condition_box, columns=("when", "tasks"),
            show="tree headings", selectmode="browse", height=15, style="Conditions.Treeview"))
        self.conditions.grid(row=0, column=0, sticky="nsew")
        self.conditions.heading("#0", text="Condition")
        self.conditions.heading("when", text="Timing")
        self.conditions.heading("tasks", text="Tasks")
        self.conditions.column("#0", width=345, minwidth=240, stretch=True)
        self.conditions.column("when", width=90, minwidth=90, stretch=False)
        self.conditions.column("tasks", width=76, minwidth=76, stretch=False, anchor="center")
        self.conditions.tag_configure("group", background=CARD, foreground=TEXT, font=self.fonts["section"])
        self.conditions.tag_configure("unavailable", foreground=MUTED)
        self.conditions.bind("<<TreeviewSelect>>", self.condition_tree_selected)
        cscroll = ttk.Scrollbar(condition_box, command=self.conditions.yview)
        cscroll.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(condition_box, orient="horizontal", command=self.conditions.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.conditions.configure(yscrollcommand=cscroll.set, xscrollcommand=horizontal.set)
        ttk.Label(content, textvariable=self.condition_count_var, style="PanelMuted.TLabel", wraplength=520).grid(row=6, column=0, sticky="w", pady=(12, 0))
        controls = ttk.Frame(sidebar, style="Panel.TFrame", padding=18)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.run_button = ttk.Button(controls, text="Run task", style="Run.TButton", command=lambda: self.start("run"))
        self.run_button.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(controls, style="Panel.TFrame")
        row.pack(fill="x")
        self.preview_button = ttk.Button(row, text="Preview scene", command=lambda: self.start("preview"))
        self.preview_button.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.stop_button = ttk.Button(row, text="Stop", style="Stop.TButton", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True)

        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)
        chooser = ttk.Frame(main)
        chooser.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        top = ttk.Frame(chooser)
        top.pack(fill="x")
        ttk.Label(top, text="Task", style="Eyebrow.TLabel").pack(side="left", padx=(0, 12))
        self.task_picker = self.register(ttk.Combobox(top, textvariable=self.task_var, state="readonly", width=60), "readonly")
        self.task_picker.pack(side="left", fill="x", expand=True)
        self.task_picker.bind("<<ComboboxSelected>>", self.list_changed)
        self.register(ttk.Button(top, text="‹", width=3, command=lambda: self.set_task(self.active_index-1))).pack(side="left", padx=(8, 4))
        self.register(ttk.Button(top, text="›", width=3, command=lambda: self.set_task(self.active_index+1))).pack(side="left")
        rolling = ttk.Frame(chooser)
        rolling.pack(fill="x", pady=(8, 0))
        ttk.Label(rolling, textvariable=self.task_count_var, style="Muted.TLabel").pack(side="left", padx=(0, 12))
        self.slider = self.register(ttk.Scale(rolling, from_=0, to=9, orient="horizontal", style="Task.Horizontal.TScale", command=self.slide_changed))
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 18))
        ttk.Label(rolling, text="Find task", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.register(ttk.Entry(rolling, textvariable=self.search_var, width=18)).pack(side="right")
        for event, delta in [("<Left>", -1), ("<Right>", 1), ("<Button-4>", -1), ("<Button-5>", 1)]:
            self.slider.bind(event, lambda e, d=delta: self.roll_task(d))
        self.slider.bind("<MouseWheel>", lambda e: self.roll_task(-1 if e.delta>0 else 1))
        self.slider.bind("<ButtonRelease-1>", lambda e: self.slider.set(max(0, self.active_index)))
        title = ttk.Label(main, textvariable=self.instruction_var, font=self.fonts["instruction"], wraplength=1100)
        title.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        info = ttk.Frame(main, style="Panel.TFrame", padding=(18, 14))
        info.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        self.condition_heading = ttk.Label(info, textvariable=self.change_title_var, style="Section.TLabel")
        self.condition_heading.pack(anchor="w")
        description = ttk.Label(info, textvariable=self.change_description_var, style="Panel.TLabel", wraplength=1080)
        description.pack(anchor="w", fill="x", pady=(6, 6))
        when = ttk.Label(info, textvariable=self.change_when_var, style="PanelMuted.TLabel", wraplength=1080)
        when.pack(anchor="w", fill="x")
        self.arm_row = ttk.Frame(info, style="Panel.TFrame")
        ttk.Label(self.arm_row, text="Max episode:", style="PanelMuted.TLabel").pack(side="left", padx=(0, 14))
        self.arm_picker = self.register(ttk.Combobox(self.arm_row, textvariable=self.arm_var,
            values=["With selected change", "Without mid-task change"], state="readonly", width=27), "readonly")
        self.arm_picker.pack(side="left")
        self.arm_picker.bind("<<ComboboxSelected>>", lambda e: self.selection_changed())
        ttk.Label(self.arm_row, text="Setup:", style="PanelMuted.TLabel").pack(side="left", padx=(18, 8))
        self.scenario_picker = self.register(ttk.Combobox(self.arm_row, textvariable=self.scenario_var,
            state="readonly", width=24), "readonly")
        self.scenario_picker.pack(side="left")
        self.scenario_picker.bind("<<ComboboxSelected>>", lambda e: self.selection_changed())
        ttk.Label(main, textvariable=self.condition_var, style="Muted.TLabel", wraplength=1080).grid(row=4, column=0, sticky="ew", pady=(0, 10))
        def resize(event):
            for widget in [title, description, when]: widget.configure(wraplength=max(360, event.width-40))
        main.bind("<Configure>", resize)
        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=5, column=0, sticky="nsew")
        live, settings, technical, commands, history = [ttk.Frame(self.tabs, padding=16) for _ in range(5)]
        for panel, text in [(live,"Live cameras"),(settings,"Settings"),(technical,"Condition details"),(commands,"Command & logs"),(history,"Results")]:
            self.tabs.add(panel, text=text)
        live.columnconfigure(0, weight=1);live.columnconfigure(1, weight=1);live.rowconfigure(1, weight=1)
        self.canvases = []
        for i, name in enumerate(["Main camera", "Wrist camera"]):
            ttk.Label(live, text=name, style="Muted.TLabel").grid(row=0, column=i, sticky="w", pady=(0,10))
            canvas = tk.Canvas(live, bg="#1b1c1a", highlightthickness=1, highlightbackground=EDGE, width=430, height=320)
            canvas.grid(row=1, column=i, sticky="nsew", padx=(0,8) if i==0 else (8,0))
            canvas.bind("<Configure>", lambda e: self.draw_frames())
            self.canvases.append(canvas)
        stats = ttk.Frame(live)
        stats.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14,8))
        for name, variable in [("Status",self.status_var),("Step",self.step_var),("Elapsed",self.elapsed_var)]:
            part=ttk.Frame(stats);part.pack(side="left",padx=(0,32))
            ttk.Label(part,text=name,style="Muted.TLabel").pack(anchor="w")
            ttk.Label(part,textvariable=variable,font=self.fonts["section"]).pack(anchor="w",pady=(3,0))
        self.progress=ttk.Progressbar(live,mode="determinate",maximum=220)
        self.progress.grid(row=3,column=0,columnspan=2,sticky="ew",pady=(0,12))
        ttk.Label(live,textvariable=self.result_var,style="Muted.TLabel",wraplength=1050).grid(row=4,column=0,columnspan=2,sticky="w",pady=(4,0))
        ttk.Label(settings,text="Episode settings",style="Eyebrow.TLabel").pack(anchor="w",pady=(0,18))
        fields=ttk.Frame(settings);fields.pack(fill="x")
        for i,(label,variable,end,helptext) in enumerate([
            ("Random seed",self.seed_var,4294967295,"Controls environment randomization."),
            ("Initial state",self.init_var,49,"Selects a saved starting robot and object state."),
            ("Maximum steps",self.limit_var,5000,"Ends the episode if the task is not finished."),
            ("Replan every",self.replan_var,50,"Number of actions to execute before requesting a new plan.")]):
            ttk.Label(fields,text=label).grid(row=i,column=0,sticky="w",pady=12,padx=(0,18))
            self.register(ttk.Spinbox(fields,textvariable=variable,from_=0 if i<2 else 1,to=end,width=10)).grid(row=i,column=1,pady=12)
            ttk.Label(fields,text=helptext,style="Muted.TLabel",wraplength=540).grid(row=i,column=2,sticky="w",padx=18)
        ttk.Label(settings,text="Current model actions",style="Eyebrow.TLabel").pack(anchor="w",pady=(24,12))
        actions=ttk.Frame(settings);actions.pack(fill="x")
        for i,name in enumerate(["Move X","Move Y","Move Z","Rotate X","Rotate Y","Rotate Z","Gripper"]):
            actions.columnconfigure(i,weight=1)
            ttk.Label(actions,text=name,style="Muted.TLabel").grid(row=0,column=i)
            ttk.Label(actions,textvariable=self.action_vars[i],font=self.fonts["code"]).grid(row=1,column=i,pady=10)
        self.detail_text=self.text_box(technical)
        top=ttk.Frame(commands);top.pack(fill="x",pady=(0,10))
        ttk.Label(top,text="Equivalent terminal command",style="Eyebrow.TLabel").pack(side="left")
        ttk.Button(top,text="Copy command",command=self.copy_command).pack(side="right")
        self.command_text=self.text_box(commands,height=7,expand=False)
        ttk.Label(commands,text="Session log",style="Muted.TLabel").pack(anchor="w",pady=(18,8))
        self.log_text=self.text_box(commands,height=8)
        ttk.Label(history,text="Recent episodes",style="Eyebrow.TLabel").pack(anchor="w",pady=(0,12))
        self.history_list=tk.Listbox(history,height=4,bg=PANEL,fg=TEXT,selectbackground="#deeddf",selectforeground=TEXT,
            relief="flat",highlightthickness=1,highlightbackground=EDGE,exportselection=False,font=self.fonts["small"])
        self.history_list.pack(fill="x",pady=(0,12))
        self.history_list.bind("<<ListboxSelect>>",self.show_history)
        self.history_text=self.text_box(history)
        ttk.Label(self.root,text="Expand a condition group → choose a change → choose a compatible task → preview or run.    A− / A+ changes text size.",style="Muted.TLabel",padding=(24,12)).pack(fill="x")
