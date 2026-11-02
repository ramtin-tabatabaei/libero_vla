"""GMM/GMR learning backend for the same full waypoint editor as option 1."""
import copy
import dataclasses
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
from tkinter import filedialog, ttk

import numpy as np

from subgoal_gui import Panel
import subgoal_planner as sp
from waypoint_chain import WORLD, PREVIOUS
from waypoint_session import Session
from waypoint_gmm import GMMChain, GMMWaypoint, chain_from_model, check_learned_motion
from waypoint_learning import Config, learn, load_demos, load_model, save_model, write_json
from waypoint_replay import initial_state, reconstruct_fused

GMM_STORE = Path(__file__).parent/'task_waypoints_gmm'
MODEL_STORE = Path(__file__).parent/'waypoint_models'
SAVED_STORE = Path(__file__).parent/'saved_plans'
"""Plans the user chose to keep, written by Save plan."""


def skill_label(skill):
    """What kind of minimal VLA plan this is, for the list and the hint."""
    if skill.get('object_push') and not skill.get('candidates'):
        return 'VLA keyframes · measured object push'
    if not skill.get('candidates') and not skill.get('stages'):
        return 'VLA keyframes · measured drawer push'
    pushes = 'push + ' if skill.get('pregrasp_push') else ''
    if skill.get('stages'):
        return f'VLA keyframes · {len(skill["stages"])} objects · grasp feedback'
    return f'VLA keyframes · {pushes}grasp and placement feedback'


class GMMSession(Session):
    def load(self, suite, task_id, cameras=True):
        chain = super().load(suite, task_id, cameras)
        self.training_state = self.env.sim.get_state().flatten().copy()
        self.training_gripper = self.executor.gripper
        self.last_shuffle_mapping = None
        return GMMChain(chain.suite, chain.task_id, chain.language, source='GMM/GMR; no demonstrations learned yet')

    def restore_training_scene(self):
        self.env.set_init_state(self.training_state)
        self.attach(settle=False)
        self.executor.gripper = self.training_gripper

    def shuffle_scene(self, seed=None):
        """Start from the original scene and exclude the last displayed shuffle."""
        from waypoint_replay import shuffle_objects, shuffle_positions
        before = self.env.sim.get_state().flatten().copy()
        gripper = self.executor.gripper
        try:
            self.restore_training_scene()
            try:
                mapping = shuffle_objects(self, seed=seed, exclude_mapping=self.last_shuffle_mapping)
            except ValueError as exc:
                if 'No different supported, non-overlapping shuffle' not in str(exc) and 'Shuffle needs at least two' not in str(exc):
                    raise
                mapping = shuffle_positions(self, seed=seed, exclude_mapping=self.last_shuffle_mapping)
        except BaseException:
            # A failed or cancelled shuffle must leave the displayed scene intact.
            self.env.set_init_state(before)
            self.attach(settle=False)
            self.executor.gripper = gripper
            raise
        self.last_shuffle_mapping = mapping
        return mapping

    def obstacles(self, start, end, ignore=()):
        from waypoint_skill import path_obstacle_top
        return path_obstacle_top(self.world, start, end, ignore)

    def underside(self, name):
        """Height of an object's lowest point in the snapshot being edited."""
        return self.reference_poses[name][0][2]-self.reference_bottom[name]

    def reachable(self, position, orientation, ignore=()):
        from waypoint_skill import reachable
        return reachable(self, position, orientation, ignore)

    def support_of(self, name):
        from waypoint_skill import supporting_bodies
        return supporting_bodies(self.world, name)

    def refresh_references(self):
        super().refresh_references()
        self.previewed = None

    def preview(self, chain):
        if not getattr(chain, 'execution_skill', {}):
            return super().preview(chain)
        # Clearance now costs reachability tests, and the editor redraws its
        # map on every keystroke. The answer only changes when the scene
        # snapshot does, which is exactly when references are refreshed.
        key = (id(chain), id(chain.execution_skill), len(chain.waypoints))
        cached = getattr(self, 'previewed', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        from waypoint_skill_editor import nominal_targets
        targets = nominal_targets(chain.execution_skill, self.reference_poses,
                                  self.start_pose, self.obstacles, self.underside,
                                  self.reachable, self.support_of)
        self.previewed = (key, targets)
        return targets

    def skill_poses(self, chain):
        """Frames a skill plan's offsets are measured against.

        A drawer the plan opens is somewhere else by the time the bowl goes in,
        so transport and place read the frame where the push leaves it. Editing
        one of those rows must resolve against that same frame, or a one
        centimetre nudge would appear to move the target by the drawer's travel.
        """
        if not getattr(chain, 'execution_skill', {}):
            return self.reference_poses
        from waypoint_skill_editor import effective_poses
        return effective_poses(chain.execution_skill, self.reference_poses)

    def editor_target(self, chain, index, waypoint):
        if chain.execution_skill and index is not None:
            targets = self.preview(chain)
            if waypoint == chain.waypoints[index]:
                return targets[index]
            previous = targets[index-1] if index else self.start_pose
            return super().resolve_target(waypoint, self.skill_poses(chain), previous, {})
        return super().editor_target(chain, index, waypoint)

    def test_waypoint(self, chain, index, waypoint, report):
        if chain.execution_skill:
            position, orientation = self.editor_target(chain, index, waypoint)
            waypoint = dataclasses.replace(waypoint, reference=WORLD,
                offset=np.asarray(position).tolist(), orientation=np.asarray(orientation).tolist())
            chain = dataclasses.replace(chain, execution_skill={}, waypoints=[waypoint])
            return super().test_waypoint(chain, 0, waypoint, report)
        return super().test_waypoint(chain, index, waypoint, report)

    def run(self, chain, report):
        with self.playback_policy():
            return self._run_skill_chain(chain, report)

    def _run_skill_chain(self, chain, report):
        if not getattr(chain, 'execution_skill', {}):
            return super().run(chain, report)
        chain.validate()
        self.last_run_failures = []
        # Keep the displayed targets from before the failure. Recomputing them
        # after a drop would make the remaining plan follow the fallen object.
        targets = self.preview(chain) if self.continue_on_failure else None
        active_row = -1
        def progress(index, message):
            nonlocal active_row
            index = min(index, len(chain.waypoints)-1)
            if message == 'Running':
                active_row = index
            report(index, message)
        from waypoint_skill import run_skill
        folder = None
        if getattr(self, 'save_replay_archive', True):
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            folder = MODEL_STORE/'_gui_replays'/stamp
            write_json(folder/'initial_state.json', dict(initial_state=self.env.sim.get_state().flatten().tolist()))
            chain.save(folder/'plan.json')
        # Internal clearance moves share their semantic row; release is the
        # final visible row. Replay uses measured poses, not policy images. The editor's camera
        # renders independently; suspend the otherwise unused per-step sensors.
        domain = self.env.env
        images = {name: (obs.is_enabled(), obs.is_active()) for name,obs in domain._observables.items()
                  if name.endswith('_image')}
        result = None
        try:
            for name in images:
                domain.modify_observable(name, 'enabled', False)
                domain.modify_observable(name, 'active', False)
            # run_skill reports one index per listed waypoint.
            completion = getattr(self, 'on_waypoint_finished', None)
            callbacks = {} if completion is None else {'waypoint_finished': completion}
            result = run_skill(self, chain.execution_skill, progress, mode='constrained', **callbacks)
            if not result['completed'] and self.continue_on_failure:
                failed_row = max(active_row, 0)
                failure = dict(row=failed_row, result=result['result'])
                report(failed_row, result['result'])
                stop_row = getattr(self, 'stop_after_waypoint', None)
                if stop_row is not None and failed_row >= stop_row:
                    # A stopped skill may have aborted before Grasp finished.
                    # Keep it here so a live handoff can use its stopped state,
                    # without replaying the tail or claiming Grasp completed.
                    self.last_run_failures = [failure]
                    return 'Stopped: '+result['result']
                # The skill cannot carry on with missing prerequisites such
                # as a lost grasp. Replay the remaining displayed targets once,
                # recording failed rows without retrying the failed skill.
                start = active_row + 1
                remaining = [dataclasses.replace(w, reference=WORLD,
                    offset=np.asarray(targets[i][0]).tolist(),
                    orientation=np.asarray(targets[i][1]).tolist(),
                    orientation_relative_to_previous=False, learned={},
                    carry=False, seat=False, hold_xy=False)
                    for i, w in enumerate(chain.waypoints) if i >= start]
                tail = dataclasses.replace(chain, execution_skill={}, waypoints=remaining)
                trace = []
                def continued(index, message):
                    row = start + index
                    if message != 'Running':
                        trace.append(dict(row=row, result=message,
                            position=remaining[index].offset,
                            orientation=remaining[index].orientation))
                    report(row, message)
                if completion is not None:
                    self.on_waypoint_finished = lambda index, message: completion(start+index, message)
                try:
                    Session.run(self, tail, continued)
                finally:
                    if completion is not None:
                        self.on_waypoint_finished = completion
                self.last_run_failures = [failure] + [dict(f, row=f['row']+start)
                                                      for f in self.last_run_failures]
                result.update(completed=True, continued_after_failure=True,
                              failures=self.last_run_failures, continuation=trace,
                              result='Finished remaining waypoints after failure',
                              task_success=bool(self.env.check_success()))
        except BaseException as exc:
            result = dict(completed=False, result=type(exc).__name__+': '+str(exc),
                          task_success=bool(self.env.check_success()))
            raise
        finally:
            if result is not None and folder is not None:
                result['shuffle_mapping'] = self.last_shuffle_mapping
                write_json(folder/'result.json', result)
                self.last_replay_report = folder/'result.json'
            for name,(enabled,active) in images.items():
                domain.modify_observable(name, 'enabled', enabled)
                domain.modify_observable(name, 'active', active)
            if images:
                domain._get_observations(force_update=True)
        if result['completed']:
            done = dict(release='released', push='pushed', place='placed',
                        move='reached', grasp='grasped', transport='carried')
            last = chain.waypoints[-1].skill_role or 'release'
            if not result.get('continued_after_failure'):
                report(len(chain.waypoints)-1, 'ok (%s; task success: %s)'
                       % (done.get(last, 'finished'), result['task_success']))
            return 'Finished object-relative waypoints; task success: '+str(result['task_success'])
        return 'Stopped: '+result['result']

    def resolve_target(self, waypoint, poses, previous, context):
        metadata = getattr(waypoint, 'learned', {})
        node = metadata.get('node', {})
        if (not node.get('frames') or metadata.get('edited') or
                waypoint.reference != metadata.get('original_reference') or
                not np.allclose(waypoint.offset, node['relative_position_mean']) or
                not np.allclose(waypoint.orientation, node['relative_orientation_mean'])):
            return super().resolve_target(waypoint, poses, previous, context)
        if 'episode_frames' not in context:
            context['episode_frames'] = copy.deepcopy(poses)
        key = ('gmr', metadata['segment'])
        if context.get('gmr_interval') != key:
            anchor = previous if context.get('preview') else self.gripper_pose()
            live = poses if context.get('preview') else self.poses()
            context['gmr_anchor'] = anchor
            context['gmr_frames'] = [anchor if b['role'] == 'anchor' else
                (live if b['policy'] == 'interval_start' else context['episode_frames'])[b['name']]
                for b in node['frames']]
            context['gmr_interval'] = key
        return reconstruct_fused(node, context['gmr_frames'], context['gmr_anchor'])

    def update_run_frames(self, poses, waypoint, context):
        if 'episode_frames' not in context:
            context['episode_frames'] = copy.deepcopy(poses)
        metadata = getattr(waypoint, 'learned', {})
        if not metadata or waypoint.reference in (WORLD, PREVIOUS) or waypoint.reference != metadata.get('original_reference'):
            super().update_run_frames(poses, waypoint, context)
            return
        if metadata['reference_policy'] == 'interval_start':
            key = (metadata['segment'], waypoint.reference)
            # Start a fresh anchor for each contiguous interval in an edited plan.
            if context.get('active_interval') != key:
                context[key] = self.poses()[waypoint.reference]
                context['active_interval'] = key
            poses[waypoint.reference] = context[key]
        else:
            key = ('episode', waypoint.reference)
            if key not in context:
                context[key] = context['episode_frames'][waypoint.reference]
            poses[waypoint.reference] = context[key]

    def check_motion_result(self, waypoint, result, position, orientation):
        if getattr(waypoint, 'learned', {}):
            return check_learned_motion(self, waypoint, result, position, orientation)
        return super().check_motion_result(waypoint, result, position, orientation)


class GMMPanel(Panel):
    session_type = GMMSession
    chain_type = GMMChain
    waypoint_limit = 100
    gmm_mode = True

    def __init__(self, root, args, host=None):
        self.learned_model = None
        self.model_file = None
        self.model_index = {}
        super().__init__(root, args, host)
        self.refresh_model_index()

    def path(self):
        return GMM_STORE/self.chain.suite/f'task_{self.chain.task_id:03d}.json'

    def saved_task_path(self, suite, task_id):
        return SAVED_STORE/suite/f'task_{task_id:03d}.json'

    def refresh_model_index(self):
        self.model_index = {}
        ranks = {}
        if not MODEL_STORE.exists():
            return
        for path in sorted(MODEL_STORE.rglob('model.json')):
            try:
                data = json.loads(path.read_text())
                if data.get('model_type') in ('event_frame_residual_gmm_gmr', 'task_parameterized_gmm_gmr', 'object_relative_vla_waypoints'):
                    # Directory names are not model versions. Prefer connected
                    # primitive models, then the most recently saved candidate.
                    from waypoint_primitives import PRIMITIVES
                    nodes = data.get('waypoints', [])
                    connected = bool(nodes) and all(n.get('primitive') in PRIMITIVES and
                        n.get('retargeting') == 'task_endpoint' for n in nodes)
                    key = (data['suite'], data['task_id'])
                    rank = (bool(data.get('object_relative_skill')), connected, path.stat().st_mtime_ns, str(path))
                    if key not in ranks or rank > ranks[key]:
                        self.model_index[key], ranks[key] = path, rank
            except (OSError, ValueError, KeyError):
                continue

    def task_plan_summary(self, suite, task_id):
        path = GMM_STORE/suite/f'task_{task_id:03d}.json'
        if path.exists():
            try:
                chain = GMMChain.load(path, suite, task_id)
                if chain.execution_skill:
                    return skill_label(chain.execution_skill), len(chain.waypoints), False
                ready = 'replay passed' in chain.source and 'edited' not in chain.source
                return ('GMM/GMR · replay passed' if ready else 'GMM/GMR · needs replay', len(chain.waypoints), ready)
            except (ValueError, KeyError):
                return 'Cannot read GMM/GMR plan', 0, False
        model = self.model_index.get((suite, task_id))
        if model:
            return 'GMM/GMR model available', len(load_model(model)['waypoints']), False
        return 'Not learned yet', 0, False

    def load_task(self):
        self.refresh_model_index()
        previous_chain = self.chain
        previous_model, previous_file = self.learned_model, self.model_file
        super().load_task()
        if self.chain is previous_chain:
            self.learned_model, self.model_file = previous_model, previous_file
            return
        self.learned_model = self.model_file = None
        if self.closing or self.chain is None:
            return
        if self.chain.model_path and Path(self.chain.model_path).exists():
            self.model_file = Path(self.chain.model_path)
            self.learned_model = load_model(self.model_file)
        elif not self.chain.waypoints:
            path = self.model_index.get((self.chain.suite, self.chain.task_id))
            if path:
                self.adopt_model(load_model(path), path)
        if (self.learned_model and self.chain.execution_skill and
                (self.chain.execution_skill.get('representation') != 'single_reference_minimal' or
                 self.chain.execution_skill.get('orientation_scheme') != 'precontact_object_and_previous') and
                'edited' not in self.chain.source):
            from waypoint_skill_editor import upgrade_model
            upgraded = upgrade_model(self.learned_model)
            if upgraded is not self.learned_model:
                folder = self.model_folder()
                save_model(upgraded, folder)
                self.adopt_model(upgraded, folder/'model.json')
        if (self.learned_model and not self.chain.execution_skill and self.chain.waypoints
                and 'edited' not in self.chain.source and all(w.learned and not w.learned.get('edited') for w in self.chain.waypoints)):
            from waypoint_skill_editor import upgrade_model
            upgraded = upgrade_model(self.learned_model)
            if upgraded is not self.learned_model:
                folder = self.model_folder()
                save_model(upgraded, folder)
                self.adopt_model(upgraded, folder/'model.json')
        self.waypoint_limit = self.chain.max_waypoints
        if not self.chain.waypoints:
            self.status.set('Choose successful demo count, then Learn waypoints · GMM/GMR. Plan tools can fit saved demonstrations.')
        elif not self.chain.execution_skill and any(not w.learned.get('primitive') for w in self.chain.waypoints):
            self.status.set('Legacy plan loaded. Plan tools → Rebuild primitives from this model’s VLA recordings to enable connected primitive replay.')
        if self.refresh_skill_template():
            self.commit(self.editing_index, edited=False)

    def refresh_skill_template(self):
        """Re-express a VLA skill plan against the layout now on screen.

        Offsets are measured in the recorded scene, and clearance also depends
        on the tallest object actually present. Recomputing them here keeps the
        centimetres in the offset box equal to the point Move robot here and
        Run all waypoints actually aim at.
        """
        chain = self.chain
        if chain is None or not chain.execution_skill or self.session.env is None:
            return False
        from waypoint_skill_editor import skill_waypoints
        waypoints = skill_waypoints(chain.execution_skill, self.session.preview(chain),
                                    self.session.reference_poses, self.session.start_pose)
        before = [dataclasses.asdict(w) for w in chain.waypoints]
        after = [dataclasses.asdict(w) for w in waypoints]
        if before == after:
            return False
        chain.waypoints = waypoints
        chain.execution_skill['waypoint_template'] = after
        chain.validate()
        return True

    def resync_plan(self):
        """Refresh displayed offsets after anything that moves the scene."""
        if self.busy or self.form_dirty or not self.refresh_skill_template():
            return
        index = self.editing_index
        self.commit(index, edited=False)
        if index is not None and index < len(self.chain.waypoints):
            self.show_form(self.chain.waypoints[index])
            self.rows.selection_set(str(index))

    def adopt_model(self, model, path):
        if (model['suite'], model['task_id']) != (self.chain.suite, self.chain.task_id):
            raise ValueError('Select this model’s task before opening it.')
        from waypoint_skill_editor import upgrade_model
        upgraded = upgrade_model(model)
        if upgraded is not model:
            model, folder = upgraded, self.model_folder()
            save_model(model, folder)
            path = folder/'model.json'
        chain = chain_from_model(model, path)
        self.session.preview(chain)
        if self.path().exists():
            shutil.copy2(self.path(), self.path().with_suffix('.backup.json'))
        self.chain, self.learned_model, self.model_file = chain, model, Path(path)
        self.waypoint_limit = chain.max_waypoints
        self.editing_index = None
        self.injections = {}
        self.form_dirty = False
        self.refresh_skill_template()
        self.commit(0, edited=False)
        self.select()
        self.status.set('GMM/GMR waypoints loaded. Edit, save, move to a waypoint, or run the whole plan.')

    def commit(self, index=None, edited=True):
        if edited and self.chain and self.chain.execution_skill:
            # Manual targets must never be silently ignored by a skill executor.
            self.chain.execution_skill = {}
            self.chain.source = 'pi0.5 keyframes; edited manual replay (grasp feedback disabled)'
        super().commit(index, edited)

    def show_form(self, waypoint):
        super().show_form(waypoint)
        metadata = getattr(waypoint, 'learned', {})
        if self.chain and self.chain.execution_skill:
            self.draft_hint.set(skill_label(self.chain.execution_skill)
                                + '. Offsets follow the scene on screen; editing switches to manual replay.')
        if metadata:
            suffix = ' · edited mean' if metadata.get('edited') else ''
            self.draft_hint.set(f"{metadata.get('primitive') or 'Legacy phase'} · GMR tolerance: {metadata['position_tolerance']*1000:.1f} mm / "
                                f"{np.degrees(metadata['orientation_tolerance']):.1f}°{suffix}")

    def mark_draft_changed(self):
        if not self.updating_form and self.chain is not None and not self.busy:
            if getattr(self.draft, 'learned', {}):
                self.draft.learned['edited'] = True
        super().mark_draft_changed()

    def editor_poses(self):
        if self.chain is None:
            return self.session.reference_poses
        return self.session.skill_poses(self.chain)

    def read_form(self):
        waypoint = super().read_form()
        if not isinstance(waypoint, GMMWaypoint):
            waypoint = GMMWaypoint(**dataclasses.asdict(waypoint))
        return waypoint

    def change_reference(self):
        old_reference, new_reference = self.draft.reference, self.reference.get()
        super().change_reference()
        if getattr(self.draft, 'learned', {}) and old_reference != new_reference:
            poses = self.session.reference_poses
            old_basis = np.eye(3) if old_reference in (WORLD, PREVIOUS) else poses[old_reference][1]
            new_basis = np.eye(3) if new_reference in (WORLD, PREVIOUS) else poses[new_reference][1]
            transform = new_basis.T @ old_basis
            covariance = np.asarray(self.draft.learned['position_covariance'])
            self.draft.learned['position_covariance'] = (transform @ covariance @ transform.T).tolist()
            # Tangent rotation covariance uses a right residual; changing the
            # external reference left-multiplies both rotations and leaves it intact.
            self.draft.learned['original_reference'] = new_reference
            self.draft.learned['reference_policy'] = 'episode_start'

    def fit(self, demos, folder, reuse_task_recordings=False):
        self.status.set('Learning object-relative VLA keyframes and contact constraints…')
        config = Config(max_waypoints=self.waypoint_limit)
        # Run pure numerical fitting off the Tk thread; only the main thread pumps Tk.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            from waypoint_skill_editor import fit_editor_model
            archive_root = folder.parent if reuse_task_recordings else None
            future = pool.submit(fit_editor_model, demos, config, None, archive_root)
            while not future.done():
                self.pump()
                time.sleep(.01)
            model = future.result()
        finally:
            # Pure fitting can finish in the background after Stop; no result is
            # adopted or written once the UI operation has been interrupted.
            pool.shutdown(wait=False)
        self.pump()
        save_model(model, folder)
        chain = chain_from_model(model, folder/'model.json')
        if model.get('supplemented_demonstrations'):
            chain.source += f"; reused {len(model['supplemented_demonstrations'])} compatible saved recording(s)"
        self.session.preview(chain)
        self.learned_model, self.model_file = model, folder/'model.json'
        return chain

    def model_folder(self):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        return MODEL_STORE/'gui'/self.chain.suite/f'task_{self.chain.task_id:03d}'/stamp

    def learning_options(self):
        try:
            successes_needed = int(self.learn_repeats.get())
        except ValueError:
            raise ValueError('Successful demonstrations must be a whole number.')
        if not 2 <= successes_needed <= 100:
            raise ValueError('GMM/GMR requires between 2 and 100 successful demonstrations.')
        return dict(successes_needed=successes_needed)

    def learn_chain(self, chain, successes_needed):
        from waypoint_demo import record
        folder = self.model_folder()
        archive = folder/'demonstrations'
        archive.mkdir(parents=True)
        original_state = self.session.env.sim.get_state().flatten().copy()
        original_gripper = self.session.executor.gripper
        demos, attempts = [], []
        ready = False
        try:
            for attempt in range(successes_needed*6):
                self.pump()
                # Learn every attempt from the task's original, unshuffled scene.
                self.session.restore_training_scene()
                self.status.set(f'GMM/GMR · {len(demos)}/{successes_needed} successes · attempt {attempt+1}')
                record(self.session, chain.language, max_steps=800 if chain.suite=='libero_10' else 500,
                       progress=lambda step,total: self.status.set(
                           f'GMM/GMR · {len(demos)}/{successes_needed} successes · attempt {attempt+1} · step {step}/{total}'))
                demo = copy.deepcopy(self.session.last_demonstration)
                demo['placement_mode'] = 'fixed_unshuffled'
                usable = demo['success'] and len(demo['samples'])>=3 and not demo['samples'][0].get('task_success', False)
                dest = archive if usable else archive/'failed'
                dest.mkdir(exist_ok=True)
                path = dest/f'demo_{attempt:03d}.json'
                write_json(dest/f'demo_{attempt:03d}.state.json', dict(initial_state=demo.pop('initial_state')))
                demo['replay_state_file'] = f'demo_{attempt:03d}.state.json'
                write_json(path, demo)
                demo['_path'] = str(path.resolve())
                attempts.append(dict(file=str(path), success=bool(demo['success']), usable=bool(usable)))
                if usable:
                    demos.append(demo)
                if len(demos) >= successes_needed:
                    from waypoint_skill_editor import can_fit_editor_model
                    ready = can_fit_editor_model(demos, folder.parent)
                    if ready:
                        break
                    self.status.set(
                        f'{len(demos)} successful runs, but no compatible event pair yet; recording another run…')
            if not ready:
                raise ValueError(
                    f'Attempt limit reached: {len(demos)} successful demonstrations but no compatible event pair. '
                    f'Saved recordings: {archive}')
        finally:
            write_json(folder/'recording_report.json', dict(successes=len(demos), attempts=attempts))
            self.session.env.set_init_state(original_state)
            self.session.attach(settle=False)
            self.session.executor.gripper = original_gripper
        chain = self.fit(demos, folder, reuse_task_recordings=True)
        return chain

    def learn_saved(self):
        self.require_scene()
        folder = filedialog.askdirectory(parent=self.root, title='Choose successful demonstrations for this task')
        if folder:
            demos = load_demos(folder)
            if any((d['suite'], d['task_id']) != (self.chain.suite, self.chain.task_id) for d in demos):
                raise ValueError('Demonstrations must match the selected task.')
            task_root = (MODEL_STORE/'gui'/self.chain.suite/
                         f'task_{self.chain.task_id:03d}').resolve()
            chosen = Path(folder).resolve()
            reuse = chosen == task_root or task_root in chosen.parents
            self.replace_chain(lambda chain: self.fit(
                demos, self.model_folder(), reuse_task_recordings=reuse))

    def open_model(self):
        self.require_scene()
        if not self.leave_draft():
            return
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(MODEL_STORE),
                                         title='Open learned GMM/GMR model for this task', filetypes=[('Learned model', '*.json')])
        if path:
            self.adopt_model(load_model(path), path)

    def import_chain(self):
        previous = self.chain
        super().import_chain()
        if self.chain is not previous and self.chain is not None:
            self.waypoint_limit = self.chain.max_waypoints
            self.model_file = Path(self.chain.model_path) if self.chain.model_path else None
            self.learned_model = load_model(self.model_file) if self.model_file and self.model_file.exists() else None

    def refresh_references(self):
        super().refresh_references()
        if self.chain is not None and self.chain.execution_skill:
            self.resync_plan()
            self.status.set('References updated from the current scene. '
                            'VLA keyframe offsets were recomputed for this layout.')

    def restore_recorded_scene(self):
        self.require_scene()
        if self.learned_model is None:
            raise ValueError('This plan has no available learned model recording.')
        paths = [Path(p) for p in self.learned_model['demonstration_files'] if p and Path(p).exists()]
        if not paths:
            raise ValueError('Original demonstration not found on disk.')
        demo = json.loads(paths[0].read_text())
        state = initial_state(demo, paths[0])
        def restore():
            self.session.env.set_init_state(state)
            self.session.attach(settle=False)
            return 'Recorded scene restored. Edited GMM/GMR waypoints kept.'
        self.operation(restore)
        self.resync_plan()

    def shuffle_scene(self):
        self.require_scene()
        def shuffle():
            mapping = self.session.shuffle_scene()
            return 'New shuffled layout: '+', '.join(mapping)+'. Run all waypoints to test.'
        self.operation(shuffle)
        self.resync_plan()

    def restore_first_setup(self):
        self.require_scene()
        def restore():
            self.session.restore_training_scene()
            return 'First setup restored: robot and objects reset. Waypoints kept.'
        self.operation(restore)
        self.resync_plan()

    def relearn_recordings(self):
        self.require_scene()
        if self.learned_model is None:
            raise ValueError('Open a learned model first.')
        paths = self.learned_model.get('demonstration_files', [])
        if len(paths) < 2 or any(not p or not Path(p).exists() for p in paths):
            raise ValueError('At least two original VLA recordings must be available.')
        demos = [dict(json.loads(Path(p).read_text()), _path=str(Path(p).resolve())) for p in paths]
        self.replace_chain(lambda chain: self.fit(demos, self.model_folder()))

    def plan_tools(self):
        dialog = self.picker_window('Plan tools · VLA waypoints', 560, 650)
        ttk.Label(dialog, text='Plan tools · VLA waypoints', style='Heading.TLabel').grid(row=0, column=0, sticky='w', padx=22, pady=20)
        def action(command):
            dialog.destroy()
            self.safe(command)
        choices = [('Record & learn GMM/GMR waypoints', self.learn), ('Learn from saved demonstrations…', self.learn_saved),
                   ('Build robust waypoints from this model’s VLA recordings', self.relearn_recordings),
                   ('Shuffle objects for replay', self.shuffle_scene),
                   ('Open learned GMM/GMR model…', self.open_model), ('Restore model’s recorded scene', self.restore_recorded_scene),
                   ('Open an edited GMM/GMR plan…', self.import_chain), ('Save a copy as…', self.export_chain)]
        for i,(label,command) in enumerate(choices,1):
            ttk.Button(dialog, text=label, command=lambda c=command: action(c)).grid(row=i,column=0,sticky='ew',padx=22,pady=7)
