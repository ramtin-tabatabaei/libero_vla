"""Desktop live failure/recovery viewer. Launch with bash run_recovery_gui.sh.

Select a saved task and condition, then watch fresh simulation frames. All
outcomes stay in memory; this viewer never records videos or writes results.
"""
import argparse
from pathlib import Path
import time

from studio_fonts import configure_tk_runtime

configure_tk_runtime()

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from waypoint_recovery_live import (BACKENDS, DEFAULT_EXPERIMENT, MODES, ROOT,
                                    LiveOptions, LiveRunner,
                                    discover_plans, eligible_rows)
from waypoint_failures import FAILURES, HINTS, INTENSITIES
from waypoint_gmm import GMMChain
import subgoal_planner as sp


SUITES = {'libero_goal': 'Goal', 'libero_object': 'Object', 'libero_spatial': 'Spatial',
          'libero_10': 'Long tasks', 'libero_90': 'LIBERO-90'}
CAMERAS = {'Scene': 'agentview', 'Wrist': 'robot0_eye_in_hand'}


class RecoveryViewer:
    def __init__(self, root, args):
        self.root = root
        self.plans = discover_plans(args.plan_root)
        if not self.plans:
            raise ValueError('No saved waypoint plans found in ' + str(args.plan_root))
        self.runner = LiveRunner(args.plan_root, args.experiment)
        self.runner.session.tick = self.pump
        self.busy = self.stopped = self.paused = self.closing = False
        self.waiting_for_vla = self.continue_requested = False
        self.photo = None
        self.last_frame = self.last_wall = 0.
        self.last_sim = None
        self.episode_start = None
        self.last_result = None
        self.controls = []
        self.suite_keys = sorted({p['suite'] for p in self.plans})
        suite = args.suite if args.suite in self.suite_keys else self.suite_keys[0]
        self.suite = tk.StringVar(value=SUITES.get(suite, suite))
        self.task = tk.StringVar()
        self.scene = tk.StringVar(value='Original')
        self.mode = tk.StringVar(value=MODES['recovery'])
        self.failure = tk.StringVar(value=FAILURES['collision'])
        self.intensity = tk.StringVar(value='Low')
        self.row = tk.StringVar(value='Automatic · Grasp')
        self.detection_latency = tk.BooleanVar(value=False)
        self.steps = tk.StringVar(value='400')
        self.limit = tk.StringVar(value='180')
        self.seed = tk.StringVar(value='1000')
        self.backend = tk.StringVar(value=BACKENDS[args.backend])
        self.port = tk.StringVar(value=str(args.port))
        self.saved_layout = tk.BooleanVar(value=True)
        self.speed = tk.StringVar(value='10×')
        self.camera = tk.StringVar(value='Scene')
        self.stage = tk.StringVar(value='Ready')
        self.detail = tk.StringVar(value='Choose a task and condition, then press Run.')
        self.metrics = tk.StringVar(value='Simulation time · —')
        self.direction_info = tk.StringVar(value='')
        self.controller_times = tk.StringVar(value='Waypoints · 0.00 s simulation | 0.00 s elapsed\n'
                                                 'VLA · 0.00 s simulation | 0.00 s elapsed')
        self.origin = tk.StringVar(value='')
        self.hint = tk.StringVar()
        self.build()
        self.update_tasks(args.task_id)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Escape>', lambda event: self.stop())
        self.root.bind('<space>', lambda event: self.toggle_pause()
                       if event.widget not in self.controls else None)

    @staticmethod
    def key(labels, value):
        return next(key for key, label in labels.items() if label == value)

    def build(self):
        self.root.title('LIBERO · Live failure recovery')
        self.root.geometry('1280x900')
        self.root.minsize(1050, 800)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('DejaVu Sans', 10))
        style.configure('Title.TLabel', font=('DejaVu Sans', 20, 'bold'))
        style.configure('Stage.TLabel', font=('DejaVu Sans', 15, 'bold'))
        style.configure('TButton', padding=(12, 8))
        style.configure('Run.TButton', background='#087e70', foreground='white')
        style.configure('Stop.TButton', background='#ad3445', foreground='white')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=(20, 16))
        header.grid(row=0, column=0, sticky='ew')
        ttk.Label(header, text='Live failure recovery', style='Title.TLabel').pack(side='left')
        ttk.Label(header, text='Live simulation · no recording').pack(side='right')
        body = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        body.grid(row=1, column=0, sticky='nsew')
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body)
        sidebar.grid(row=0, column=0, sticky='ns', padx=(0, 14))
        settings_canvas = tk.Canvas(sidebar, width=355, highlightthickness=0)
        settings_canvas.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=settings_canvas.yview)
        scrollbar.pack(side='right', fill='y')
        settings_canvas.configure(yscrollcommand=scrollbar.set)
        settings = ttk.Frame(settings_canvas, padding=(10, 2, 14, 10))
        settings_window = settings_canvas.create_window(0, 0, window=settings, anchor='nw')
        settings.bind('<Configure>', lambda event: settings_canvas.configure(
            scrollregion=settings_canvas.bbox('all')))
        settings_canvas.bind('<Configure>', lambda event: settings_canvas.itemconfigure(
            settings_window, width=event.width))
        settings.columnconfigure(0, weight=1)
        self.combo(settings, 'Task suite', self.suite,
                   [SUITES.get(s, s) for s in self.suite_keys], self.update_tasks)
        self.tasks_box = self.combo(settings, 'Task (IDs start at 0)', self.task, [], self.update_conditions)
        self.combo(settings, 'Scene layout', self.scene, ['Original', 'Shuffled'])
        check = ttk.Checkbutton(settings, text='Reuse saved paired layout when available', variable=self.saved_layout)
        check.pack(anchor='w', pady=(3, 8))
        self.controls.append(check)
        self.combo(settings, 'Run condition', self.mode, list(MODES.values()), self.update_conditions)
        self.failure_box = self.combo(settings, 'Failure', self.failure, list(FAILURES.values()), self.update_conditions)
        self.intensity_box = self.combo(settings, 'Intensity', self.intensity, list(INTENSITIES.values()), self.update_conditions)
        self.row_box = self.combo(settings, 'Failure waypoint', self.row, ['Automatic · Grasp'])
        ttk.Label(settings, text='Grasp faults · High grasp loss at the next Transport end',
                  wraplength=310, foreground='#52647a').pack(anchor='w', pady=(3, 2))
        ttk.Label(settings, textvariable=self.hint, wraplength=310, foreground='#52647a').pack(anchor='w', pady=(4, 8))
        self.latency_check = ttk.Checkbutton(settings, text='Delayed detection: finish waypoint plan',
                                             variable=self.detection_latency)
        self.latency_check.pack(anchor='w', pady=(0, 8))
        self.controls.append(self.latency_check)
        fields = ttk.Frame(settings)
        fields.pack(fill='x')
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=1)
        self.entry(fields, 'Waypoint limit (sim s)', self.limit, 0, 0)
        self.entry(fields, 'Policy step limit', self.steps, 0, 1)
        self.entry(fields, 'Layout seed', self.seed, 1, 0)
        self.backend_box = self.combo(settings, 'Recovery / normal policy', self.backend,
                                     list(BACKENDS.values()), self.update_conditions)
        self.port_entry = self.entry(settings, 'Pi 0.5 server port', self.port)
        self.policy_hint = ttk.Label(settings, wraplength=310, foreground='#52647a')
        self.policy_hint.pack(anchor='w', pady=(6, 8))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        viewbar = ttk.Frame(right)
        viewbar.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(viewbar, text='Live camera').pack(side='left')
        for label, variable, values in [('Speed', self.speed, ['0.5×', '1×', '2×', '5×', '10×']),
                                         ('Camera', self.camera, list(CAMERAS))]:
            selector = ttk.Combobox(viewbar, textvariable=variable, values=values,
                                   state='readonly', width=9)
            selector.pack(side='right', padx=(6, 12))
            if variable is self.camera:
                selector.bind('<<ComboboxSelected>>', lambda event: self.refresh())
            ttk.Label(viewbar, text=label).pack(side='right')
        self.canvas = tk.Canvas(right, background='#172b40', highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.canvas.create_text(300, 220, text='Load a scene or press Run', fill='#dce7f3',
                                font=('DejaVu Sans', 16))
        info = ttk.Frame(right, padding=(4, 12))
        info.grid(row=2, column=0, sticky='ew')
        ttk.Label(info, textvariable=self.stage, style='Stage.TLabel').pack(anchor='w')
        ttk.Label(info, textvariable=self.detail, wraplength=670).pack(anchor='w', pady=6)
        ttk.Label(info, textvariable=self.metrics).pack(anchor='w')
        ttk.Label(info, textvariable=self.direction_info).pack(anchor='w', pady=(4, 0))
        ttk.Label(info, textvariable=self.controller_times, font=('DejaVu Sans', 11, 'bold'),
                  justify='left').pack(anchor='w', pady=(8, 0))
        ttk.Label(info, textvariable=self.origin, wraplength=670, foreground='#52647a').pack(anchor='w', pady=(5, 0))
        actions = ttk.Frame(self.root, padding=(20, 8, 20, 16))
        actions.grid(row=2, column=0, sticky='ew')
        self.load_button = ttk.Button(actions, text='Load scene', command=lambda: self.operate(False))
        self.load_button.pack(side='left')
        self.run_button = ttk.Button(actions, text='Run selected condition', style='Run.TButton',
                                     command=lambda: self.operate(True))
        self.run_button.pack(side='left', padx=8)
        self.continue_button = ttk.Button(actions, text='Continue VLA', style='Run.TButton',
                                          command=self.continue_vla, state='disabled')
        self.continue_button.pack(side='left', padx=(0, 8))
        self.pause_button = ttk.Button(actions, text='Pause', command=self.toggle_pause, state='disabled')
        self.pause_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(actions, text='Stop · Esc', style='Stop.TButton',
                                      command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        ttk.Label(actions, text='Recovery waits for Continue VLA.').pack(side='right')
        self.controls += [self.load_button, self.run_button]

    def combo(self, parent, label, variable, values, callback=None):
        ttk.Label(parent, text=label).pack(anchor='w', pady=(5, 3))
        box = ttk.Combobox(parent, textvariable=variable, values=values, state='readonly', width=37)
        box.pack(fill='x', pady=(0, 4))
        self.controls.append(box)
        if callback:
            box.bind('<<ComboboxSelected>>', lambda event: callback())
        return box

    def entry(self, parent, label, variable, row=None, column=None):
        frame = ttk.Frame(parent)
        if row is None:
            frame.pack(fill='x')
        else:
            frame.grid(row=row, column=column, sticky='ew', padx=(0, 8), pady=3)
        ttk.Label(frame, text=label).pack(anchor='w')
        entry = ttk.Entry(frame, textvariable=variable, width=12)
        entry.pack(fill='x', pady=3)
        self.controls.append(entry)
        return entry

    def update_tasks(self, preferred=None):
        suite = next(s for s in self.suite_keys if SUITES.get(s, s) == self.suite.get())
        self.task_choices = {'%03d · %s' % (p['task_id'], p['language']): p
                             for p in self.plans if p['suite'] == suite}
        self.tasks_box.configure(values=list(self.task_choices))
        self.task.set(next((label for label, p in self.task_choices.items()
                            if p['task_id'] == preferred), next(iter(self.task_choices))))
        self.update_conditions()

    def update_conditions(self):
        plan = self.task_choices[self.task.get()]
        chain = GMMChain.load(plan['path'], plan['suite'], plan['task_id'])
        kind = self.key(FAILURES, self.failure.get())
        intensity = self.key(INTENSITIES, self.intensity.get())
        rows = eligible_rows(chain, kind, intensity)
        phase = 'Transport' if kind == 'grasp_loss' and intensity == 'high' else 'Grasp'
        automatic = 'Automatic · '+phase
        self.row_choices = {automatic: None}
        self.row_choices.update({'%d · %s' % (i+1, chain.waypoints[i].name): i for i in rows})
        self.row_box.configure(values=list(self.row_choices))
        if self.row.get() not in self.row_choices:
            self.row.set(automatic)
        hint = ('Low: prevent acquisition in Grasp; High: drop the held object at the end of the next Transport waypoint'
                if kind == 'grasp_loss' else
                'Before closing: random local XY shift and downward tool-Z press. Low: 4–6 cm XY + 6 cm down; High: 6–10 cm XY + 12 cm down. 3 simulation seconds.'
                if kind == 'collision' else HINTS[kind])
        if kind == 'orientation':
            hint = 'Strong random yaw with smaller roll/pitch. Low: 20–40° total + 1–2 cm XY; High: 40–80° total + 2–4 cm XY. Random direction, rotation biased toward the upper end.'
        self.hint.set(hint if rows else 'No applicable %s waypoint for this failure in this task.' % phase)
        failure_mode = self.key(MODES, self.mode.get()) in ('recovery', 'failure_only')
        for widget in (self.failure_box, self.intensity_box, self.row_box, self.latency_check):
            widget.state(['!disabled'] if failure_mode else ['disabled'])
        use_policy = self.key(MODES, self.mode.get()) in ('recovery', 'normal_policy')
        self.backend_box.state(['!disabled'] if use_policy else ['disabled'])
        pi05 = self.key(BACKENDS, self.backend.get()) == 'pi05'
        self.port_entry.state(['!disabled'] if use_policy and pi05 else ['disabled'])
        self.policy_hint.configure(text=('Pi 0.5 server: bash serve_pi05_libero.sh'
            if pi05 else 'Uses the installed local policy runtime.') if use_policy
            else 'This condition runs without a policy server.')
        self.run_button.state(['disabled'] if failure_mode and not rows else ['!disabled'])

    def options(self):
        p = self.task_choices[self.task.get()]
        try:
            mode = self.key(MODES, self.mode.get())
            options = LiveOptions(suite=p['suite'], task_id=p['task_id'], scene=self.scene.get().lower(),
                mode=mode, kind=self.key(FAILURES, self.failure.get()),
                intensity=self.key(INTENSITIES, self.intensity.get()),
                handoff_at_end=(self.detection_latency.get()
                                and mode in ('recovery', 'failure_only')),
                propagate_deviation=mode in ('recovery', 'failure_only'),
                max_waypoint_seconds=float(self.limit.get()), max_policy_steps=int(self.steps.get()),
                seed=int(self.seed.get()), port=int(self.port.get()),
                backend=self.key(BACKENDS, self.backend.get()),
                injection_row=self.row_choices.get(self.row.get()), use_saved_layout=self.saved_layout.get())
        except ValueError as exc:
            raise ValueError('Enter valid numbers for limits, seed and port.') from exc
        options.validate()
        return options

    def report(self, stage, message):
        if stage in ('Transition deviation', 'Collision direction', 'Orientation deviation'):
            self.direction_info.set(message)
            return
        self.stage.set(stage)
        self.detail.set(message)

    def operate(self, run):
        if self.busy:
            return
        try:
            options = self.options()
        except ValueError as exc:
            self.report('Check condition', str(exc))
            return
        self.busy, self.stopped, self.paused = True, False, False
        self.waiting_for_vla = self.continue_requested = False
        self.continue_button.state(['disabled'])
        self.last_result = None
        self.direction_info.set('')
        self.last_sim = self.episode_start = None
        self.last_frame = 0.
        for control in self.controls:
            control.state(['disabled'])
        for control in (self.pause_button, self.stop_button):
            control.state(['!disabled'])
        self.pause_button.configure(text='Pause')
        self.report('Loading scene', 'Preparing the selected task and layout…')
        self.root.update()
        try:
            self.runner.prepare(options)
            self.origin.set(self.runner.scene_origin)
            self.episode_start = float(self.runner.session.env.sim.data.time)
            self.last_sim, self.last_wall = self.episode_start, time.monotonic()
            self.refresh()
            self.pump()
            if run:
                self.last_result = self.runner.run(options, self.report, before_vla=self.wait_for_vla)
            else:
                self.report('Scene ready', self.runner.chain.language)
        except sp.Abort:
            self.report('Stopped', 'Robot stopped. Press Run to restart from the selected layout.')
        except Exception as exc:
            self.report('Could not complete run', '%s: %s' % (type(exc).__name__, exc))
        finally:
            self.busy = self.paused = False
            self.waiting_for_vla = self.continue_requested = False
            self.continue_button.state(['disabled'])
            for control in self.controls:
                control.state(['!disabled'])
            for control in (self.pause_button, self.stop_button):
                control.state(['disabled'])
            self.pause_button.configure(text='Pause')
            self.update_conditions()
            if self.closing:
                self.close()
            elif self.runner.session.env is not None:
                self.refresh()

    def pump(self):
        """Pump Tk on the simulation thread, including during policy inference."""
        self.root.update()
        while self.paused and not self.stopped:
            env = self.runner.session.env
            if env is not None:
                self.runner.timings.pause(env.sim.data.time)
                self.update_controller_times()
            self.root.update()
            time.sleep(.02)
            self.last_wall = time.monotonic()
        if self.runner.session.env is not None:
            self.runner.timings.resume(self.runner.session.env.sim.data.time)
        if self.stopped:
            raise sp.Abort()
        env = self.runner.session.env
        if env is None:
            return
        now = float(env.sim.data.time)
        # Session.motion_tick only paces waypoints. Pace here so OSC policy
        # actions use the same speed control, without counting inference twice.
        if self.last_sim is not None and now > self.last_sim:
            rate = float(self.speed.get().rstrip('×'))
            deadline = self.last_wall + min(now-self.last_sim, .1)/rate
            while time.monotonic() < deadline and not self.stopped and not self.paused:
                self.root.update()
                time.sleep(.005)
        if self.stopped:
            raise sp.Abort()
        if self.paused:
            return self.pump()
        if self.last_sim != now:
            self.last_sim, self.last_wall = now, time.monotonic()
        if time.monotonic()-self.last_frame >= 1/24:
            self.refresh()

    def refresh(self):
        env = self.runner.session.env
        if env is None:
            return
        pixels = env.sim.render(width=640, height=480, camera_name=CAMERAS[self.camera.get()])[::-1].copy()
        picture = Image.fromarray(pixels)
        width, height = max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())
        scale = min(width/picture.width, height/picture.height)
        picture = picture.resize((max(1, int(picture.width*scale)), max(1, int(picture.height*scale))),
                                 Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(picture, master=self.root)
        self.canvas.delete('all')
        self.canvas.create_image(width/2, height/2, image=self.photo)
        elapsed = 0. if self.episode_start is None else float(env.sim.data.time)-self.episode_start
        xyz = self.runner.session.world.eef()
        self.metrics.set('Simulation time · %.2f s     Gripper · X %.2f  Y %.2f  Z %.2f m'
                         % (elapsed, xyz[0], xyz[1], xyz[2]))
        self.update_controller_times()
        self.last_frame = time.monotonic()
        self.root.update_idletasks()

    def update_controller_times(self):
        env = self.runner.session.env
        now = 0. if env is None else float(env.sim.data.time)
        seconds = self.runner.timings.snapshot(now)
        self.controller_times.set('Waypoints · %.2f s simulation | %.2f s elapsed\n'
                                  'VLA · %.2f s simulation | %.2f s elapsed' % (
            seconds['waypoints_sim_seconds'], seconds['waypoints_wall_seconds'],
            seconds['vla_sim_seconds'], seconds['vla_wall_seconds']))

    def wait_for_vla(self, outcome):
        """Keep Tk responsive without advancing physics or starting inference."""
        if not outcome.get('handoff_ready') or not outcome.get('waypoint_execution_stopped'):
            raise ValueError('Waypoint execution must stop before Continue VLA can be enabled.')
        self.waiting_for_vla = True
        self.continue_requested = self.paused = False
        self.continue_button.state(['!disabled'])
        self.pause_button.state(['disabled'])
        self.pause_button.configure(text='Pause')
        self.refresh()
        try:
            while not self.continue_requested:
                self.pump()
                time.sleep(.01)
        finally:
            self.waiting_for_vla = False
            self.continue_button.state(['disabled'])
            if self.busy:
                self.pause_button.state(['!disabled'])

    def continue_vla(self):
        if self.busy and self.waiting_for_vla and not self.stopped:
            self.continue_requested = True

    def toggle_pause(self):
        if self.busy and not self.waiting_for_vla:
            self.paused = not self.paused
            self.pause_button.configure(text='Resume' if self.paused else 'Pause')

    def stop(self):
        if self.busy:
            self.stopped, self.paused = True, False

    def close(self):
        if self.busy:
            self.closing = True
            self.stop()
            return
        self.runner.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-root', type=Path, default=ROOT/'saved_plans')
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT,
                        help='Read existing paired layouts from this experiment; never write to it.')
    parser.add_argument('--suite', default='libero_goal')
    parser.add_argument('--task-id', type=int, default=1)
    parser.add_argument('--backend', choices=tuple(BACKENDS), default='pi05')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = tk.Tk()
    viewer = None
    try:
        viewer = RecoveryViewer(root, args)
        root.mainloop()
    finally:
        if viewer is not None:
            viewer.runner.close()
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == '__main__':
    main()
