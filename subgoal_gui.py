"""Task waypoint studio. Run with the libero Python environment.

MUJOCO_GL=glx python subgoal_gui.py

The data model lives in waypoint_chain.py; simulation lives in waypoint_session.py.
Edits never move the robot. Task chains are saved in task_waypoints/ automatically.
"""
import copy
from contextlib import nullcontext
import dataclasses
import hashlib
import shutil
from pathlib import Path
import time
import warnings
from studio_fonts import configure_tk_runtime

configure_tk_runtime()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageTk
import subgoal_planner as sp
from waypoint_chain import Chain, Waypoint, WORLD, PREVIOUS, MAX_WAYPOINTS, resolve, relative_waypoint
from waypoint_session import Session
from waypoint_ui import StudioLayout, MOTIONS, POLICY_LABELS, SUITE_LABELS, friendly_name

STORE = Path(__file__).resolve().parent / "task_waypoints"
SAVED = Path(__file__).resolve().parent / "saved_plans"
"""Plans the user chose to keep, written by Save plan."""


class Panel(StudioLayout):
    session_type = Session
    chain_type = Chain
    waypoint_limit = MAX_WAYPOINTS
    gmm_mode = False

    def __init__(self, root, args, host=None):
        self.root, self.args = root, args
        self.host = host if host is not None else root
        self.embedded = host is not None
        self.session = self.session_type(args)
        self.session.tick = self.pump
        self.session.realtime = True
        # Pace fast enough to watch, not so slow that a plan takes minutes.
        # Below the simulator's own step rate this throttles nothing.
        self.session.playback_rate = 10.0
        self.chain = None
        self.busy = self.stopped = self.closing = False
        self.draft = Waypoint()
        self.last_frame = 0.
        self.frame_cost = 0.
        self.playback_started = None
        self.photo = None
        self.camera_image = None
        self.camera_zoom = 1.
        self.camera_pan = np.zeros(2)
        self.form_dirty = False
        self.updating_form = False
        self.editing_index = None
        self.injections = {}
        """Deliberate failure per waypoint row; only Run all waypoints applies them."""
        self.suite = tk.StringVar(value=args.task_suite_name)
        self.task_id = args.task_id
        self.build_workspace()
        for var in [self.name, self.motion, self.grip, self.grip_after, self.carry, self.hold_xy]:
            var.trace_add("write", lambda *_: self.mark_draft_changed())
        for var in self.xyz:
            var.trace_add("write", lambda *_: self.position_changed())
        for var in self.rpy:
            var.trace_add("write", lambda *_: self.rotation_changed())
        for var in self.xyz[:2]:
            var.trace_add("write", lambda *_: self.unlock_xy())
        if not self.embedded:
            root.protocol("WM_DELETE_WINDOW", self.close)
            root.bind("<Escape>", lambda _: self.stop())
            root.bind("<Control-s>", lambda _: self.safe(self.save_plan))
            root.bind("<Control-t>", lambda _: self.safe(self.task_picker))
            root.bind("<Alt-Left>", lambda _: self.safe(lambda: self.change_task_by(-1)))
            root.bind("<Alt-Right>", lambda _: self.safe(lambda: self.change_task_by(1)))

    def safe(self, command):
        if self.busy:
            return
        try:
            command()
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror("Waypoint studio", str(exc), parent=self.root)

    def require_scene(self):
        if self.chain is None or self.session.env is None:
            raise ValueError("Load a task first.")

    def path(self):
        return STORE / self.chain.suite / f"task_{self.chain.task_id:03d}.json"

    def persist(self):
        if self.chain:
            self.chain.save(self.path())

    def mark_draft_changed(self):
        if self.updating_form or self.chain is None or self.busy:
            return
        self.form_dirty = True
        self.plan_status.set("Edited target · replay needed")
        self.save_state.set("●  Unsaved waypoint changes")
        self.draft_hint.set("Save this waypoint to keep your changes.")
        if getattr(self, "redraw_job", None):
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(180, self.redraw_preview)

    def redraw_preview(self):
        """Keep the scene axes and relationship list in sync with the editor."""
        self.redraw_job = None
        self.draw_relationships()
        self.paint_camera()

    def rotation_changed(self):
        self.mark_draft_changed()
        if self.updating_form or self.chain is None or self.busy:
            return
        if getattr(self, "redraw_job", None):
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after_idle(self.redraw_preview)

    def position_changed(self):
        self.mark_draft_changed()

    def orientation_basis(self, waypoint):
        if waypoint.reference == WORLD:
            return np.eye(3)
        if waypoint.reference == PREVIOUS:
            return self.previous()[1] if waypoint.orientation_relative_to_previous else np.eye(3)
        return self.editor_poses()[waypoint.reference][1]

    def unlock_xy(self):
        if not self.updating_form and not self.busy:
            # A manually entered X/Y must never be silently ignored by templates.
            self.hold_xy.set(False)

    def leave_draft(self):
        if not self.form_dirty:
            return True
        answer = messagebox.askyesnocancel("Save waypoint changes?", "Save your edited waypoint before continuing?", parent=self.root)
        if answer is None:
            return False
        if answer:
            self.save_waypoint()
        else:
            self.form_dirty = False
            if self.editing_index is not None and self.editing_index < len(self.chain.waypoints):
                self.show_form(self.chain.waypoints[self.editing_index])
            self.save_state.set("Changes discarded")
        return True

    def save_waypoint(self):
        self.require_scene()
        w = self.read_form()
        if self.editing_index is None:
            self.check_room()
            self.chain.waypoints.append(w)
            index = len(self.chain.waypoints)-1
        else:
            index = self.editing_index
            self.chain.waypoints[index] = w
        self.form_dirty = False
        self.editing_index = index
        self.commit(index)
        self.show_form(w)
        self.status.set(f"Saved waypoint {index+1}: {friendly_name(w.name)}")

    def saved_path(self):
        """Where Save plan writes, as opposed to where editing autosaves."""
        return self.saved_task_path(self.chain.suite, self.chain.task_id)

    def saved_task_path(self, suite, task_id):
        return SAVED / suite / f"task_{task_id:03d}.json"

    def save_plan(self):
        self.require_scene()
        if self.form_dirty:
            self.save_waypoint()
        else:
            self.persist()
        # Only an explicit Save plan action confirms this copy for the picker.
        target = self.saved_path()
        self.chain.save(target)
        target.with_suffix(".user_saved").write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "\n")
        self.save_state.set("✓  Saved to your plans")
        self.status.set(f"Saved {len(self.chain.waypoints)} waypoints to {target}")

    def task_saved(self, suite, task_id):
        target = self.saved_task_path(suite, task_id)
        try:
            return target.with_suffix(".user_saved").read_text().strip() == hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            return False

    def move_gripper(self, grip):
        self.require_scene()
        chain = dataclasses.replace(self.chain, waypoints=[Waypoint(
            f"{grip.title()} gripper", PREVIOUS, motion="grip", gripper=grip)])
        def move():
            result = self.session.run(chain, lambda *_: None)
            return f"Gripper {'opened' if grip == 'open' else 'closed'}." if result.startswith("Finished") else result
        self.operation(move, waypoint_playback=True)

    def load_task(self):
        if not self.leave_draft():
            if self.chain:
                self.suite.set(self.chain.suite)
                self.task_id = self.chain.task_id
            return
        self.rows.delete(*self.rows.get_children())
        self.editing_index = None
        self.injections = {}
        suite, task = self.suite.get(), self.task_id
        def load():
            self.chain = None
            self.chain = self.session.load(suite, task)
            if self.random_robot_start.get():
                self.apply_random_robot_start()
            if self.path().exists():
                try:
                    saved = self.chain_type.load(self.path(), suite, task)
                    self.session.preview(saved)
                    self.chain = saved
                except Exception:
                    # Do not let later autosaves destroy a file that could not load.
                    self.chain = None
                    raise
            else:
                return "Ready. Choose a policy and press Run to drive this task, or learn a waypoint plan."
            return "Ready. Select a waypoint to edit, or press Run all waypoints."
        self.operation(load)
        if self.closing or self.chain is None:
            return
        self.show_task()

    def show_task(self):
        """Display the current chain, including during batch learning."""
        self.goal.set(self.chain.language)
        self.task_caption.set(f"{SUITE_LABELS[self.chain.suite].upper()}  /  TASK {self.chain.task_id+1:02d}")
        self.note.set(f"{len(self.chain.waypoints)} saved waypoints • {self.chain.source}")
        self.redraw_rows()
        if self.chain.waypoints:
            self.rows.selection_set("0")
            self.editing_index = 0
            self.show_form(self.chain.waypoints[0])
        else:
            self.new()
        self.refresh()

    def change_task_by(self, delta):
        from libero.libero import benchmark
        if self.chain is None:
            return
        suite = self.chain.suite
        count = benchmark.get_benchmark_dict()[suite]().n_tasks
        self.suite.set(suite)
        self.task_id = (self.chain.task_id + delta) % count
        self.load_task()

    def task_plan_summary(self, suite, task_id):
        from waypoint_catalog import plan_summary
        return plan_summary(STORE, suite, task_id)

    def reset(self):
        self.require_scene()
        def reset():
            self.session.reset()
            if self.random_robot_start.get():
                self.apply_random_robot_start()
                return "Scene reset with a new random robot start; chain kept."
            return "Scene reset; chain kept."
        self.operation(reset)

    def apply_random_robot_start(self):
        from studio_robot_start import randomize_robot_start
        return randomize_robot_start(self.session)

    def new_robot_start(self):
        self.require_scene()
        def randomize():
            self.apply_random_robot_start()
            return "New random robot start. Object layout and waypoint plan kept."
        self.operation(randomize)

    def refresh_references(self):
        self.require_scene()
        self.session.refresh_references()
        self.refresh()
        self.status.set("References updated from the current scene. Offsets are unchanged.")

    def selected(self):
        selection = self.rows.selection()
        return int(selection[0]) if selection else None

    def editor_poses(self):
        """Reference frames the offset box is written and read against."""
        return self.session.reference_poses

    def previous(self):
        index = self.editing_index
        points = self.session.preview(self.chain)
        if index is None:
            return points[-1] if points else self.session.start_pose
        return points[index-1] if index else self.session.start_pose

    def read_form(self):
        w = copy.deepcopy(self.draft)
        w.name, w.reference = self.name.get().strip() or "Waypoint", self.reference.get()
        try:
            w.offset = [float(v.get()) / 100 for v in self.xyz]
        except ValueError:
            raise ValueError("Enter numbers for X, Y and Z in centimetres.")
        w.motion, w.gripper = MOTIONS[self.motion.get()], self.grip.get()
        w.gripper_after = self.grip_after.get()
        try:
            angles = [float(var.get()) for var in self.rpy]
        except ValueError:
            raise ValueError("Enter numbers for roll, pitch and yaw in degrees.")
        if not np.isfinite(angles).all():
            raise ValueError("Orientation angles in degrees must be finite.")
        # Opening or saving an unchanged form must preserve the exact stored
        # rotation, including learned poses rounded only for display.
        displayed = [float(value) for value in getattr(self, "shown_rpy", [])]
        if not np.array_equal(angles, displayed):
            world_rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            # The fields describe the frame drawn in the scene. Plans retain
            # their object/previous-relative rotations for connected replay.
            w.orientation = (np.asarray(self.orientation_basis(w)).T @ world_rotation).tolist()
        if w.carry != self.carry.get():
            w.seat = self.carry.get()
        w.carry = self.carry.get()
        w.hold_xy = self.hold_xy.get()
        if w.gripper == "close" and self.draft.motion == "grip" and w.reference in self.session.world.domain.objects_dict:
            w.grasp = w.reference
        elif w.gripper != "close":
            w.grasp = ""
        w.validate()
        if w.motion == "arc" and not w.joint:
            raise ValueError("Joint arcs come from generated templates; choose another motion for a new target.")
        return w

    def show_form(self, w):
        self.updating_form = True
        self.draft = copy.deepcopy(w)
        self.name.set(w.name)
        self.reference.set(w.reference)
        # Older plans remain readable. The editor offers pose movements;
        # opening and closing the fingers use the separate gripper controls.
        motion = "line" if w.motion == "grip" else w.motion
        self.motion.set(next(k for k, v in MOTIONS.items() if v == motion))
        self.grip.set(w.gripper)
        self.grip_after.set(w.gripper_after)
        self.carry.set(w.carry)
        self.hold_xy.set(w.hold_xy)
        self.shown_xyz = [f"{value*100:.3f}" for value in w.offset]
        for var, value in zip(self.xyz, self.shown_xyz):
            var.set(value)
        orientation = np.asarray(w.orientation)
        if self.chain is not None and self.session.env is not None:
            try:
                target = dataclasses.replace(w, motion=motion)
                orientation = self.session.editor_target(self.chain, self.editing_index, target)[1]
            except (ValueError, KeyError):
                pass
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)
            angles = Rotation.from_matrix(np.asarray(orientation)).as_euler("xyz", degrees=True)
        self.shown_rpy = [f"{0. if abs(value) < .0005 else value:.3f}" for value in angles]
        for var, value in zip(self.rpy, self.shown_rpy):
            var.set(value)
        self.updating_form = False
        self.form_dirty = False
        self.editor_title.set("New waypoint" if self.editing_index is None else f"Edit waypoint {self.editing_index+1:02d}")
        self.save_waypoint_button.configure(text="Save new waypoint" if self.editing_index is None else "Save waypoint changes")
        self.save_state.set("✓  All changes saved")
        self.draft_hint.set("Editing changes the target only.")
        self.draw_relationships()
        self.paint_camera()

    def new(self):
        self.require_scene()
        self.check_room()
        if not self.leave_draft():
            return
        self.editing_index = None
        self.rows.selection_remove(*self.rows.selection())
        pos, ori = self.session.gripper_pose()
        self.show_form(relative_waypoint(f"Waypoint {len(self.chain.waypoints)+1}", pos, ori, WORLD, self.session.poses(), self.previous()))

    def select(self):
        # Playback owns the scene target until the operation ends. Treeview
        # selection events can still arrive while its controls are disabled.
        if self.busy:
            if self.editing_index is not None and self.rows.exists(str(self.editing_index)):
                self.rows.selection_set(str(self.editing_index))
            return
        i = self.selected()
        if i == self.editing_index:
            return
        if i is not None and self.chain and i < len(self.chain.waypoints):
            if not self.leave_draft():
                if self.editing_index is not None:
                    self.rows.selection_set(str(self.editing_index))
                else:
                    self.rows.selection_remove(*self.rows.selection())
                return
            self.editing_index = i
            self.rows.selection_set(str(i))
            self.show_form(self.chain.waypoints[i])

    def change_reference(self):
        self.require_scene()
        # Keep the physical target fixed while expressing it in the new frame.
        ref = self.reference.get()
        self.reference.set(self.draft.reference)
        try:
            w = self.read_form()
            pos, ori = resolve(w, self.editor_poses(), self.previous())
        finally:
            self.reference.set(ref)
        updated = relative_waypoint(w.name, pos, ori, ref, self.editor_poses(), self.previous())
        w.reference, w.offset, w.orientation = ref, updated.offset, updated.orientation
        w.orientation_relative_to_previous = updated.orientation_relative_to_previous
        self.show_form(w)
        self.mark_draft_changed()

    def capture(self):
        self.require_scene()
        w = self.read_form()
        pos, ori = self.session.gripper_pose()
        target = relative_waypoint(w.name, pos, ori, w.reference, self.editor_poses(), self.previous())
        w.offset, w.orientation = target.offset, target.orientation
        w.orientation_relative_to_previous = target.orientation_relative_to_previous
        w.carry = w.seat = w.hold_xy = False
        self.show_form(w)
        self.mark_draft_changed()

    def above(self):
        self.require_scene()
        w = self.read_form()
        if w.reference in (WORLD, PREVIOUS):
            raise ValueError("Choose an object or region in Reference first.")
        pos = self.session.world.pos(w.reference)
        pos[2] = self.session.world.top_z(w.reference) + .15
        target = relative_waypoint(w.name, pos, sp.DOWN, w.reference, self.editor_poses(), self.previous())
        w.offset, w.orientation, w.motion = target.offset, target.orientation, "reach"
        w.orientation_relative_to_previous = target.orientation_relative_to_previous
        w.carry = w.seat = w.hold_xy = False
        self.show_form(w)
        self.mark_draft_changed()

    def lift(self):
        self.require_scene()
        self.show_form(Waypoint("Lift 15 cm", PREVIOUS, [0, 0, .15],
                                orientation=np.eye(3).tolist(),
                                orientation_relative_to_previous=True, motion="line"))

    def nudge(self, axis, delta):
        self.xyz[axis].set(f"{float(self.xyz[axis].get()) + delta*100:.3f}")
        self.draw_relationships()

    def nudge_rotation(self, axis, degrees):
        self.rpy[axis].set(f"{float(self.rpy[axis].get()) + degrees:.3f}")
        self.draw_relationships()

    def commit(self, index=None, edited=True):
        if edited and "pi0.5" in self.chain.source:
            self.chain.source = self.chain.source.split("; replay")[0].split("; edited")[0] + "; edited; replay unverified"
        self.persist()
        self.redraw_rows(index)
        self.draw_relationships()
        self.note.set(f"Saved {len(self.chain.waypoints)} waypoints • {self.chain.source}")
        self.save_state.set("✓  All changes saved")

    def add(self):
        self.require_scene()
        self.check_room()
        self.chain.waypoints.append(self.read_form())
        self.commit(len(self.chain.waypoints)-1)

    def apply(self):
        self.require_scene()
        i = self.selected()
        if i is None:
            raise ValueError("Select a waypoint to update, or use Add to chain.")
        self.chain.waypoints[i] = self.read_form()
        self.commit(i)

    def duplicate(self):
        if not self.leave_draft():
            return
        i = self.selected()
        if i is not None:
            self.check_room()
            count = len(self.chain.waypoints)
            self.chain.waypoints.insert(i+1, copy.deepcopy(self.chain.waypoints[i]))
            self.shift_injections({j: j if j <= i else j+1 for j in range(count)})
            self.commit(i+1)

    def delete(self):
        i = self.selected()
        if i is not None:
            count = len(self.chain.waypoints)
            self.chain.waypoints.pop(i)
            self.shift_injections({j: j-1 if j > i else j for j in range(count) if j != i})
            self.editing_index = None
            self.form_dirty = False
            index = min(i, len(self.chain.waypoints)-1) if self.chain.waypoints else None
            self.commit(index)
            if index is None:
                self.new()
            else:
                self.select()

    def reorder(self, delta):
        if not self.leave_draft():
            return
        i = self.selected()
        if i is not None and 0 <= i+delta < len(self.chain.waypoints):
            w = self.chain.waypoints
            w[i], w[i+delta] = w[i+delta], w[i]
            moved = {j: j for j in range(len(w))}
            moved[i], moved[i+delta] = i+delta, i
            self.shift_injections(moved)
            self.commit(i+delta)

    def row_detail(self, index, waypoint):
        """The list's second line: what a row is measured against.

        A row armed with a deliberate failure says that instead. It is the one
        thing about the row that watching the robot will not tell you until it
        has already happened.
        """
        from waypoint_failures import failure_label
        selection = self.injections.get(index)
        if selection:
            return "\u26a0 " + failure_label(selection)
        return friendly_name(waypoint.reference)

    def shift_injections(self, moved):
        """Keep armed failures with their waypoints after a list edit."""
        self.injections = {new: self.injections[old] for old, new in moved.items()
                           if old in self.injections}

    def redraw_rows(self, selected=None):
        self.rows.delete(*self.rows.get_children())
        armed = len(self.injections)
        self.failure_summary.set("No failures injected." if not armed else
                                 "\u26a0  %d waypoint%s set to fail on purpose."
                                 % (armed, "s" if armed > 1 else ""))
        self.count_text.set(f"{len(self.chain.waypoints)} adaptive steps" if self.gmm_mode else f"{len(self.chain.waypoints)} / 9 steps")
        source = self.chain.source
        if getattr(self.chain, 'execution_skill', {}):
            from waypoint_gmm_editor import skill_label
            self.plan_status.set(skill_label(self.chain.execution_skill))
        elif self.gmm_mode:
            self.plan_status.set("GMM/GMR · adaptive waypoints · replay needed" if "replay passed" not in source else "GMM/GMR · replay passed")
        elif "pi0.5" in source:
            state = "replay passed" if "replay passed" in source else "replay failed" if "replay FAILED" in source else "replay needed"
            self.plan_status.set(f"Learned from pi0.5 · {state}")
        else:
            self.plan_status.set("Manual plan" if source == "manual" else "Legacy plan · learn from pi0.5")
        for i, w in enumerate(self.chain.waypoints):
            name = friendly_name(w.name)
            detail = self.row_detail(i, w)
            if len(name) > 25:
                name = name[:24] + "…"
            if len(detail) > 29:
                detail = detail[:28] + "…"
            self.rows.insert("", "end", iid=str(i), text=f"{i+1:02d}   {name}\n       {detail}")
        if selected is not None:
            self.rows.selection_set(str(selected))
            self.rows.see(str(selected))

    def replace_chain(self, fn):
        self.require_scene()
        if not self.leave_draft():
            return
        # Automatic backup makes regeneration reversible.
        if self.path().exists():
            shutil.copy2(self.path(), self.path().with_suffix(".backup.json"))
        def replace():
            self.chain = fn(self.chain)
            return self.chain.source
        self.operation(replace)
        if not self.closing:
            self.editing_index = None
            self.injections = {}
            self.form_dirty = False
            self.commit(0 if self.chain.waypoints else None, edited=False)
            self.select() if self.chain.waypoints else self.new()

    def generate(self):
        self.learn()

    def selected_policy(self):
        return next(key for key, label in POLICY_LABELS.items() if label == self.policy_choice.get())

    def change_policy(self, event=None):
        name = POLICY_LABELS[self.selected_policy()]
        self.run_policy_button.configure(text=f"▶  Run {name}")
        self.policy_hint.set(f"Local Pi0.5 server · port {self.session.args.policy_port}.")

    def run_policy(self):
        self.require_scene()
        backend = self.selected_policy()
        name = POLICY_LABELS[backend]

        def execute():
            monitor = self.policy_coverage.prepare(self.session, self.chain, backend)
            with monitor if monitor is not None else nullcontext():
                result = self.session.run_policy(self.chain.language, report=self.status.set, backend=backend)
                if monitor is not None:
                    monitor.set_policy_result(result)
            outcome = "task succeeded" if result["success"] else "step limit reached"
            return f"{name}: {outcome} after {result['steps']} steps."

        self.operation(execute)

    def check_room(self):
        if self.chain and len(self.chain.waypoints) >= self.waypoint_limit:
            raise ValueError(f"This task already has {self.waypoint_limit} waypoints. Edit or remove an existing waypoint.")

    def learning_options(self):
        try:
            repeats = int(self.learn_repeats.get())
            count = None if self.learn_waypoints.get() == "Auto" else int(self.learn_waypoints.get())
        except ValueError:
            raise ValueError("Learning runs must be a whole number; waypoints must be Auto or 1–9.")
        if not 1 <= repeats <= 100:
            raise ValueError("Choose between 1 and 100 learning runs.")
        if count is not None and not 1 <= count <= MAX_WAYPOINTS:
            raise ValueError("Choose between 1 and 9 waypoints, or Auto.")
        return dict(repeats=repeats, waypoint_count=count)

    def learn_chain(self, chain, **options):
        return self.session.record_policy(chain, report=self.status.set,
                           archive_dir=STORE / "_demonstrations/gui", **options)

    def learn(self):
        self.require_scene()
        options = self.learning_options()
        self.replace_chain(lambda chain: self.learn_chain(chain, **options))

    def learn_all_tasks(self, suite):
        """Learn and autosave drafts; only Save plan marks a task kept."""
        from libero.libero import benchmark
        options = self.learning_options()
        if not self.leave_draft():
            return
        total = benchmark.get_benchmark_dict()[suite]().n_tasks
        completed, failures = [], []
        self.cancel_pending_redraw()

        def summary(state):
            text = f"{state}: learned {len(completed)}/{total} tasks in {SUITE_LABELS[suite]}. Drafts autosaved; click Save plan to keep a plan."
            if failures:
                text += " Failed tasks: " + "; ".join(f"{task+1:02d} ({error})" for task, error in failures) + "."
            return text

        def learn_all():
            for task_id in range(total):
                try:
                    self.pump()
                    self.suite.set(suite)
                    self.task_id = task_id
                    self.chain = None
                    self.editing_index = None
                    self.injections = {}
                    self.rows.delete(*self.rows.get_children())
                    if self.gmm_mode:
                        self.learned_model = self.model_file = None
                    self.robot_state.set(f"●  Learning task {task_id+1}/{total} · {len(completed)} learned")
                    self.status.set(f"Learning task {task_id+1}/{total} in {SUITE_LABELS[suite]}…")
                    self.chain = self.session.load(suite, task_id)
                    if self.path().exists():
                        saved = self.chain_type.load(self.path(), suite, task_id)
                        self.session.preview(saved)
                        self.chain = saved
                        shutil.copy2(self.path(), self.path().with_suffix(".backup.json"))
                    self.show_task()
                    self.pump()
                    learned = self.learn_chain(self.chain, **options)
                    self.pump()
                    self.chain = learned
                    self.show_task()
                    self.persist()
                    self.save_state.set("✓  Draft autosaved · click Save plan to keep it")
                    completed.append(task_id)
                except sp.Abort:
                    return summary("Stopped")
                except Exception as exc:
                    failures.append((task_id, str(exc)))
            return summary("Learning finished")

        self.operation(learn_all)

    def test(self):
        self.require_scene()
        w = self.read_form()
        index = self.editing_index
        def move():
            result = self.session.test_waypoint(self.chain, index, w, lambda i, result: self.report(
                index if index is not None else i, result))
            if result.startswith("Finished"):
                return "Robot reached this waypoint. Save it to keep your changes."
            detail = result.split(": ", 1)[1] if result.startswith("Stopped at ") else result
            label = f"Waypoint {index+1}" if index is not None else "New waypoint"
            return f"{label}: {detail}"
        self.operation(move, waypoint_playback=True)

    def failure_injector(self):
        """The deliberate failures armed for this run, ready to patch the executor."""
        from waypoint_failures import Injector
        return Injector(self.session, self.injections, waypoints=self.chain.waypoints)

    def run(self):
        self.require_scene()
        if not self.leave_draft():
            return
        if not self.chain.waypoints:
            raise ValueError("Add or generate waypoints first.")
        # Validate the armed failures before moving the robot.
        injector = self.failure_injector()
        def follow(index, result):
            # Every executor reports a row as Running before it starts moving,
            # which is the last moment a fault can still be armed for that row.
            if result == "Running":
                injector.start_row(index)
            self.report(index, result, follow=True)
        def run():
            self.session.refresh_references()
            previous_mode = self.session.continue_on_failure
            self.session.continue_on_failure = True
            try:
                with injector:
                    result = self.session.run(self.chain, follow)
            finally:
                self.session.continue_on_failure = previous_mode
            injected = ("  ·  injected: " + "; ".join(injector.fired)) if injector.fired else ""
            if injector.missed:
                injected += "  ·  not injected: " + "; ".join(injector.missed)
            if result.startswith("Finished"):
                failures = self.session.last_run_failures
                if failures:
                    goal = "reached" if self.session.env.check_success() else "not reached"
                    return (f"Sequence finished with {len(failures)} failed waypoint(s). "
                            f"Task goal {goal}." + injected)
                return ("Sequence complete — task goal reached." if self.session.env.check_success()
                        else "Sequence finished. Task goal not reached — review the waypoints.") + injected
            return result + injected
        self.operation(run, waypoint_playback=True)

    def report(self, index, result, follow=False):
        # Full playback follows the active row; a single-waypoint test keeps
        # its unsaved draft in the editor. show_form redraws both scene axes
        # and the map before the executor starts moving to this waypoint.
        if follow and 0 <= index < len(self.chain.waypoints):
            # Only an executor start event changes the active waypoint. A late
            # completion/status message must not move the scene highlight.
            if result != "Running" and self.editing_index != index:
                return
            if result == "Running":
                self.editing_index = index
                self.rows.selection_set(str(index))
                self.show_form(self.chain.waypoints[index])
        self.status.set(f"Waypoint {index+1}: {result}")
        for item in self.rows.get_children():
            self.rows.item(item, tags=())
        if self.rows.exists(str(index)):
            self.rows.item(str(index), tags=("running" if result == "Running" else "failed" if not result.startswith("ok") else "",))
            self.rows.see(str(index))

    def import_chain(self):
        self.require_scene()
        if not self.leave_draft():
            return
        path = filedialog.askopenfilename(filetypes=[("Task chains", "*.json")])
        if path:
            chain = self.chain_type.load(path, self.chain.suite, self.chain.task_id)
            if len(chain.waypoints) > self.waypoint_limit:
                raise ValueError(f"This file exceeds the {self.waypoint_limit}-waypoint limit.")
            self.session.preview(chain)  # validate references before replacing
            if self.path().exists():
                shutil.copy2(self.path(), self.path().with_suffix(".backup.json"))
            self.chain = chain
            self.editing_index = None
            self.injections = {}
            self.form_dirty = False
            self.commit(0 if chain.waypoints else None, edited=False)
            self.select() if chain.waypoints else self.new()

    def export_chain(self):
        self.require_scene()
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=self.path().name)
        if path:
            self.chain.save(path)
            self.status.set(f"Exported {path}")

    def lock(self, enabled):
        if enabled:
            self.states = []
            def walk(parent):
                for child in parent.winfo_children():
                    if (isinstance(child, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Treeview))
                            and child is not self.stop_button and child not in self.camera_controls):
                        self.states.append((child, child.state()))
                        child.state(["disabled"])
                    walk(child)
            walk(self.host)
            if getattr(self, "app", None):
                self.app.set_busy(self, True)
        else:
            for widget, state in self.states:
                widget.state(["!disabled"])
                widget.state(state)
            if getattr(self, "app", None):
                self.app.set_busy(self, False)

    def operation(self, fn, waypoint_playback=False):
        self.busy, self.stopped = True, False
        self.start_playback_timing()
        self.lock(True)
        self.robot_state.set("●  Working…")
        self.status.set("Working… Press Stop robot or Esc to interrupt.")
        self.root.update_idletasks()
        try:
            with self.session.waypoint_view() if waypoint_playback else nullcontext():
                self.status.set(str(fn()))
        except sp.Abort:
            self.status.set("Stopped. The chain is kept.")
        except Exception as exc:
            self.status.set(f"Failed: {exc}")
            messagebox.showerror("Operation failed", str(exc), parent=self.root)
        finally:
            self.busy = False
            # Idle again: the axes go back to following the editor's own target.
            self.session.running_target = None
            self.robot_state.set("●  Ready")
            self.lock(False)
            if self.closing:
                self.app.close() if getattr(self, "app", None) else self.close()
            elif self.session.env is not None:
                self.refresh()

    def pump(self):
        # The clock limits rendering only; report(..., "Running") owns progress.
        # Leave at least 80% of the time between frames for simulation. A slow
        # renderer must not cause another frame immediately on the next step.
        if time.monotonic() - self.last_frame > max(1/30, 4*self.frame_cost):
            self.refresh()
        self.root.update()
        if self.stopped:
            raise sp.Abort()

    def refresh(self):
        started = time.monotonic()
        self.camera_image = Image.fromarray(self.session.image())
        self.paint_camera()
        position = self.session.world.eef()*100
        self.pose_label.set(f"Gripper  ·  X {position[0]:.1f}   Y {position[1]:.1f}   Z {position[2]:.1f} cm")
        self.draw_relationships()
        self.last_frame = time.monotonic()
        self.frame_cost = self.last_frame - started
        if self.playback_started is not None:
            wall_start, sim_start = self.playback_started
            elapsed = self.last_frame - wall_start
            simulated = float(self.session.env.sim.data.time) - sim_start
            if elapsed > 0 and simulated > 0:
                self.actual_speed.set(f"Actual: {simulated/elapsed:.2f}×")

    def start_playback_timing(self):
        self.playback_started = None
        self.actual_speed.set("Actual: —")
        if self.session.env is not None:
            self.playback_started = (time.monotonic(), float(self.session.env.sim.data.time))

    def paint_camera(self):
        if self.camera_image is None:
            return
        width, height = max(self.camera.winfo_width(), 2), max(self.camera.winfo_height(), 2)
        self.clamp_camera_pan()
        img = self.camera_image.resize(self.camera_image_size(),
                         Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.camera.delete("all")
        center = np.array([width/2, height/2]) + self.camera_pan
        self.camera.create_image(*center, image=self.photo, tags="camera_image")
        self.paint_plan(img.width, img.height)
        self.paint_offset_axes(img.width, img.height)
        # A reference can share the waypoint's origin. Keep the editable wrist
        # axes visible above it when their arrows overlap.
        self.camera.tag_raise("target_axes")

    def camera_image_size(self):
        width, height = max(self.camera.winfo_width(), 2), max(self.camera.winfo_height(), 2)
        source = self.camera_image
        scale = min(width/source.width, height/source.height) * self.camera_zoom
        return max(1, int(source.width*scale)), max(1, int(source.height*scale))

    def clamp_camera_pan(self):
        if self.camera_image is None:
            return
        image_size = np.array(self.camera_image_size())
        viewport = np.array([self.camera.winfo_width(), self.camera.winfo_height()])
        limits = np.maximum(0., (image_size-viewport)/2)
        self.camera_pan = np.clip(self.camera_pan, -limits, limits)

    def zoom_camera(self, factor, anchor=None):
        old_zoom = self.camera_zoom
        self.camera_zoom = float(np.clip(old_zoom*factor, 1., 4.))
        middle = np.array([self.camera.winfo_width()/2, self.camera.winfo_height()/2])
        anchor = middle if anchor is None else np.asarray(anchor, float)
        center = middle + self.camera_pan
        self.camera_pan = anchor + (center-anchor)*(self.camera_zoom/old_zoom) - middle
        self.camera_zoom_text.set(f"{self.camera_zoom*100:.0f}%")
        self.paint_camera()

    def change_playback_speed(self, event=None):
        self.session.playback_rate = float(self.playback_speed.get().rstrip("×"))
        if self.busy:
            self.start_playback_timing()

    def reset_camera_zoom(self):
        self.camera_zoom = 1.
        self.camera_pan = np.zeros(2)
        self.camera_zoom_text.set("100%")
        self.paint_camera()

    def camera_wheel(self, event):
        number, delta = getattr(event, "num", None), getattr(event, "delta", 0)
        if number in (4, 5) or delta:
            up = number == 4 or delta > 0
            self.zoom_camera(1.2 if up else 1/1.2, (event.x, event.y))
        return "break"

    def camera_pan_start(self, event):
        self.camera_drag_origin = np.array([event.x, event.y])
        self.camera_drag_pan = self.camera_pan.copy()

    def camera_pan_drag(self, event):
        if not hasattr(self, "camera_drag_origin"):
            return
        delta = np.array([event.x, event.y])-self.camera_drag_origin
        self.camera_pan = self.camera_drag_pan + delta
        self.paint_camera()

    def camera_view(self, image_width, image_height):
        """Project world points into the rendered camera image.

        Returns a function giving pixel coordinates, or None for a point behind
        the lens or outside the picture, so callers can simply skip it.
        """
        sim = self.session.env.sim
        cid = sim.model.camera_name2id("agentview")
        rotation = sim.data.cam_xmat[cid].reshape(3, 3).T
        position = sim.data.cam_xpos[cid]
        width, height = self.camera.winfo_width(), self.camera.winfo_height()
        center = np.array([width/2, height/2]) + getattr(self, "camera_pan", np.zeros(2))
        focal = .5 * image_height / np.tan(np.radians(sim.model.cam_fovy[cid]) * .5)

        # Integer resizing can slightly change the image's original aspect ratio.
        source = self.camera_image
        aspect = source.width / source.height if source is not None else image_width / image_height
        focal_x = focal * image_width / (image_height * aspect)

        def project(point, clip=True):
            local = rotation @ (np.asarray(point, float) - position)
            if local[2] >= -1e-6:
                return None
            pixel = center + [focal_x*local[0]/-local[2], -focal*local[1]/-local[2]]
            if clip and not (max(0, center[0]-image_width/2) < pixel[0] < min(width, center[0]+image_width/2) and
                             max(0, center[1]-image_height/2) < pixel[1] < min(height, center[1]+image_height/2)):
                return None
            return pixel
        project.rotation = rotation
        return project

    def paint_plan(self, image_width, image_height):
        """The frame the robot is moving to: where the wrist goes, and how.

        Editing, this is the target of the selected row, following the draft as
        it is typed, so the axes move with the numbers in the offset box.

        Running, it is the pose the executor was actually given, which is not
        always the stored one: a grasp adjusted for clearance, or a contact
        refreshed from an object that has moved since the run began, is a
        different pose by centimetres. The axes belong on the waypoint the
        robot is doing, not on the one the file remembers.
        """
        if self.chain is None or self.session.env is None:
            return
        live = self.session.running_target if self.busy else None
        if live is not None:
            position, orientation = live
        else:
            try:
                position, orientation = self.session.editor_target(
                    self.chain, self.editing_index, self.read_form())
            except (ValueError, KeyError, tk.TclError):
                return
        label = "New waypoint"
        if self.editing_index is not None and self.editing_index < len(self.chain.waypoints):
            label = self.plan_labels()[self.editing_index]
        self.draw_camera_frame(image_width, image_height, position,
                               np.asarray(orientation), label)

    def paint_offset_axes(self, image_width, image_height):
        """Draw the actual reference used to resolve the editor's offsets."""
        reference = self.reference.get()
        if self.chain is None or self.session.env is None or reference == WORLD:
            return
        try:
            if reference == PREVIOUS:
                origin, _ = self.previous()
                # PREVIOUS translations use world axes, even when the wrist
                # orientation is relative to the preceding wrist rotation.
                basis = np.eye(3)
                index = self.editing_index
                count = len(self.chain.waypoints) if index is None else index
                label = f"Previous waypoint {count}" if count else "Start pose"
                target, _ = self.session.editor_target(self.chain, index, self.read_form())
                project = self.camera_view(image_width, image_height)
                a, b = project(origin), project(target)
                if a is not None and b is not None:
                    self.camera.create_line(*a, *b, fill="#ec8c24", width=2,
                                            dash=(5, 4), tags="plan_overlay")
            else:
                pose = self.editor_poses().get(reference)
                if pose is None:
                    return
                origin, basis = pose
                label = friendly_name(reference)
        except (ValueError, KeyError, tk.TclError):
            return
        # Snapshot / predicted frames must not be mixed with live positions:
        # moving an object does not move the saved editor target.
        self.draw_camera_frame(image_width, image_height,
                               origin, basis, label, caption_offset=24, frame_tag="reference_axes")

    def draw_camera_frame(self, image_width, image_height, origin, basis, label,
                          length=.08, caption_offset=-18, frame_tag="target_axes"):
        """Draw one XYZ triad over the camera image, labelled.

        An axis pointing nearly at or away from the lens has no length on
        screen, so it is drawn as a ring and says which way it goes instead of
        becoming an invisible stub.
        """
        project = self.camera_view(image_width, image_height)
        start = project(origin)
        if start is None:
            return
        x, y = start
        c = self.camera
        reference = frame_tag == "reference_axes"
        colors = ("#f0a2ac", "#94d7be", "#b3c7f1") if reference else ("#e84a5f", "#20c997", "#448aff")
        tags = ("offset_axes", frame_tag)
        for i, (axis, color) in enumerate(zip(("X", "Y", "Z"), colors)):
            end = project(origin + length*basis[:, i], clip=False)
            if end is None:
                continue
            delta = end - start
            span = np.linalg.norm(delta)
            name = "+" + axis
            if span < 8:
                toward = (project.rotation @ basis[:, i])[2] > 0
                name += " toward you" if toward else " away"
                c.create_oval(x-4, y-4, x+4, y+4, outline=color, width=2,
                              tags=tags)
                where = start + [14, 18]
            else:
                c.create_line(x, y, *end, fill=color, width=2 if reference else 3,
                              dash=(4, 3) if reference else (),
                              arrow="last", arrowshape=(9, 11, 4), tags=tags)
                where = end + 12*delta/span
            c.create_text(*where, text=name, fill=color,
                          font=(self.font_family, 11, "bold"), tags=tags)
        c.create_oval(x-3, y-3, x+3, y+3, fill="white", outline="#10233c",
                      tags=tags)
        self.camera_caption(x, y+caption_offset, label, frame_tag)

    def camera_caption(self, x, y, text, frame_tag="target_axes"):
        """White text over a camera image needs an outline to stay readable."""
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.camera.create_text(x+dx, y+dy, text=text, fill="#10233c",
                                    tags=("plan_overlay", frame_tag),
                                    font=(self.font_family, 10, "bold"))
        self.camera.create_text(x, y, text=text, fill="white", tags=("plan_overlay", frame_tag),
                                font=(self.font_family, 10, "bold"))

    def stop(self):
        self.stopped = True

    def plan_labels(self):
        """Short name per row: the primitive where there is one, else the name."""
        labels = []
        for i, w in enumerate(self.chain.waypoints, 1):
            role = getattr(w, "skill_role", "") or friendly_name(w.name)
            labels.append(f"{i}  {role.title()}")
        return labels

    def draw_relationships(self):
        """Show every waypoint's defining frame and local offset, including edits."""
        table = self.relationship_rows
        records = []
        if self.chain is not None:
            for index, stored in enumerate(self.chain.waypoints):
                waypoint = stored
                if index == self.editing_index and not self.updating_form:
                    try:
                        waypoint = self.read_form()
                    except (ValueError, KeyError, tk.TclError):
                        pass
                role = getattr(waypoint, "skill_role", "") or friendly_name(waypoint.name)
                if waypoint.reference == PREVIOUS:
                    previous = self.chain.waypoints[index-1] if index else None
                    label = (f"Previous waypoint {index:02d} · {friendly_name(previous.name)}"
                             if previous is not None else "Start pose")
                elif waypoint.reference == WORLD:
                    label = "World · fixed scene coordinates"
                else:
                    label = friendly_name(waypoint.reference)
                offset = "   ".join(f"{axis} {value*100:+.2f}" for axis, value in zip("XYZ", waypoint.offset))
                tags = tuple(self.rows.item(str(index), "tags")) if self.rows.exists(str(index)) else ()
                if index == self.editing_index and self.form_dirty:
                    tags = ("edited",)
                records.append((str(index), (f"{index+1:02d} {role.title()}", f"{label}\n{offset}"), tags))
        signature = (tuple(records), self.editing_index)
        if signature == getattr(self, "relationship_signature", None):
            return
        self.relationship_signature = signature
        identifiers = {record[0] for record in records}
        for identifier in table.get_children():
            if identifier not in identifiers:
                table.delete(identifier)
        for position, (identifier, values, tags) in enumerate(records):
            if table.exists(identifier):
                table.item(identifier, values=values, tags=tags)
                table.move(identifier, "", position)
            else:
                table.insert("", "end", iid=identifier, values=values, tags=tags)
        selected = str(self.editing_index) if self.editing_index is not None else None
        if selected in identifiers:
            if table.selection() != (selected,):
                table.selection_set(selected)
                table.see(selected)
        elif table.selection():
            table.selection_remove(*table.selection())

    def select_relationship(self):
        selection = self.relationship_rows.selection()
        if not selection or self.busy or int(selection[0]) == self.editing_index:
            return
        self.rows.selection_set(selection[0])
        self.rows.see(selection[0])
        self.select()
        # A cancelled draft change restores the relationship selection too.
        self.relationship_signature = None
        self.draw_relationships()

    def cancel_pending_redraw(self):
        job = getattr(self, "redraw_job", None)
        if job:
            self.root.after_cancel(job)
            self.redraw_job = None

    def close(self):
        if self.busy:
            self.closing = True
            self.stop()
            return
        if not self.leave_draft():
            self.closing = False
            return
        self.cancel_pending_redraw()
        if self.session.env is not None:
            self.session.env.close()
        self.root.destroy()


class StudioApplication:
    """The VLA waypoint editor: one task, one draft, one simulator."""
    def __init__(self, root, args):
        from waypoint_gmm_editor import GMMPanel
        self.root = root
        root.title("LIBERO | Waypoint Studio")
        root.geometry("1500x970")
        root.minsize(1240, 840)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame = ttk.Frame(root)
        frame.grid(row=0, column=0, sticky="nsew")
        self.panels = [GMMPanel(root, args, frame)]
        for panel in self.panels:
            panel.app = self
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.active.stop())
        root.bind("<Control-s>", lambda _: self.active.safe(self.active.save_plan))
        root.bind("<Control-t>", lambda _: self.active.safe(self.active.task_picker))
        root.bind("<Alt-Left>", lambda _: self.active.safe(lambda: self.active.change_task_by(-1)))
        root.bind("<Alt-Right>", lambda _: self.active.safe(lambda: self.active.change_task_by(1)))
        root.after(100, self.changed)

    @property
    def active(self):
        return self.panels[0]

    def changed(self, event=None):
        panel = self.active
        if panel.session.env is None and not panel.busy:
            panel.safe(panel.load_task)

    def set_busy(self, active, busy):
        """No sibling editor to lock out; the panel owns its own controls."""

    def close(self):
        running = next((p for p in self.panels if p.busy), None)
        if running:
            running.closing = True
            running.stop()
            return
        for panel in self.panels:
            if not panel.leave_draft():
                for p in self.panels:
                    p.closing = False
                return
        for panel in self.panels:
            panel.cancel_pending_redraw()
            if panel.session.env is not None:
                panel.session.env.close()
        self.root.destroy()


def main(args):
    root = tk.Tk()
    StudioApplication(root, args)
    root.mainloop()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(sp.Args))
