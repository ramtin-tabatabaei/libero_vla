"""Geometry and evidence regressions for object-relative VLA skills."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from test_waypoint_learning import placed
from waypoint_skill import (learn_skill, learn_pregrasp_push, relative, compose, inverse)


def two_stage_placed(bend=0.):
    first, second = placed(bend=bend), placed(bend=bend)
    first_samples = first['samples'] + [first['terminal_sample']]
    second_samples = second['samples'] + [second['terminal_sample']]
    for sample in first_samples:
        sample['object_poses']['blocker'] = sample['object_poses'].pop('source')
        sample['grasped'] = ['blocker' if name == 'source' else name
                             for name in sample['grasped']]
        if 'source' in sample['contact_state']:
            sample['contact_state']['blocker'] = sample['contact_state'].pop('source')
        sample['object_poses']['source'] = copy.deepcopy(
            second['initial_object_poses']['source'])
    blocker_at_release = copy.deepcopy(first_samples[-1]['object_poses']['blocker'])
    for sample in second_samples:
        sample['object_poses']['blocker'] = copy.deepcopy(blocker_at_release)
    result = copy.deepcopy(second)
    result['samples'] = first_samples + second['samples']
    result['terminal_sample'] = second['terminal_sample']
    result['initial_object_poses']['blocker'] = copy.deepcopy(
        first['initial_object_poses']['source'])
    result['poses']['blocker'] = copy.deepcopy(first['initial_object_poses']['source'])
    result['objects'] = ['blocker', 'source']
    result['goals'] = [['on', 'source', 'destination'], ['turnon', 'stove']]
    for sample in result['samples'] + [result['terminal_sample']]:
        sample['goal_predicates'] = [False, True]
    return result


class ObjectRelativeSkillTest(unittest.TestCase):
    def test_pose_roundtrip_and_carried_object_placement(self):
        eef = (np.array([.2, -.1, .3]), Rotation.from_rotvec([.1, .2, .7]).as_matrix())
        obj = (np.array([.21, -.09, .25]), Rotation.from_rotvec([.2, -.1, .3]).as_matrix())
        held = relative(obj, eef)
        target = (np.array([-.2, .3, .15]), Rotation.from_rotvec([0, 0, -.5]).as_matrix())
        command = compose(target, inverse(held))
        predicted = compose(command, held)
        np.testing.assert_allclose(predicted[0], target[0], atol=1e-12)
        np.testing.assert_allclose(predicted[1], target[1], atol=1e-12)

    def test_grasps_are_measured_poses_not_averaged(self):
        demos = [placed(bend=-.01), placed(), placed(bend=.01)]
        for index, demo in enumerate(demos):
            demo['_path'] = str(index)
        skill = learn_skill(demos)
        self.assertEqual(skill['source'], 'source')
        self.assertEqual(skill['destination'], 'destination')
        for candidate in skill['candidates']:
            demo = next(d for d in demos if d.get('_path') == candidate.get('demo_file'))
            stable = demo['samples'][candidate['grasp_sample']]
            expected_position = relative([stable['position'], stable['orientation']],
                                         stable['object_poses']['source'])[0]
            np.testing.assert_allclose(candidate['grasp'][0], expected_position)
            entry = demo['samples'][candidate['grasp_entry_sample']]
            _, grasp_orientation = compose(entry['object_poses']['source'],
                                           candidate['grasp'])
            np.testing.assert_allclose(grasp_orientation, stable['orientation'])

    def test_orientation_uses_object_pose_before_grasp_rotation(self):
        demo = placed()
        rotation = Rotation.from_euler('z', .65).as_matrix()
        from waypoint_learning import samples
        ss = samples(demo)
        grasp = next(i for i in range(2, len(ss))
                     if all('source' in sample.get('grasped', [])
                            for sample in ss[i-2:i+1]))
        for sample in ss[grasp:]:
            object_pose = sample['object_poses']['source']
            object_pose[1] = (rotation @ np.asarray(object_pose[1])).tolist()
        candidate = learn_skill([demo])['candidates'][0]
        entry = ss[candidate['grasp_entry_sample']]
        stable = ss[candidate['grasp_sample']]
        _, replayed_orientation = compose(entry['object_poses']['source'],
                                          candidate['grasp'])
        np.testing.assert_allclose(replayed_orientation, stable['orientation'])
        _, old_orientation = compose(entry['object_poses']['source'],
                                     candidate['stable_grasp'])
        self.assertGreater(Rotation.from_matrix(
            old_orientation.T @ np.asarray(stable['orientation'])).magnitude(), .6)
        peak = candidate['transport_sample']
        _, transport_orientation = compose(ss[peak]['object_poses']['destination'],
                                           ([0, 0, 0], candidate['transport_orientation']))
        np.testing.assert_allclose(transport_orientation, ss[peak]['orientation'])

    def test_failed_and_unmeasured_grasps_cannot_train_skill(self):
        demo = placed()
        for sample in demo['samples']+[demo['terminal_sample']]:
            sample['grasped'] = []
        with self.assertRaisesRegex(ValueError, 'measured grasp'):
            learn_skill([demo])
        demo = placed();demo['success'] = False
        with self.assertRaises(ValueError):learn_skill([demo])
        demo = placed();demo['placement_mode'] = 'shuffled'
        with self.assertRaises(ValueError):learn_skill([demo])

    def test_minimal_plan_uses_one_reference_per_waypoint(self):
        from waypoint_skill_editor import fit_editor_model
        from waypoint_gmm import chain_from_model
        model = fit_editor_model([placed(bend=-.01), placed(), placed(bend=.01)])
        skill = model['object_relative_skill']
        self.assertEqual(skill['representation'], 'single_reference_minimal')
        self.assertFalse(any('transport' in c for c in skill['candidates']))
        chain = chain_from_model(model)
        # One row per motion the executor drives: lift clear, cross to the
        # object, descend, lift it, cross to the goal, lower it, let go.
        self.assertEqual([w.skill_role for w in chain.waypoints],
                         ['move','move','grasp','transport','transport','place','release'])
        self.assertEqual([w.reference for w in chain.waypoints],
                         ['Previous waypoint','source','source','source',
                          'destination','destination','Previous waypoint'])
        for waypoint in chain.waypoints:
            self.assertEqual(np.asarray(waypoint.offset).shape, (3,))
        self.assertTrue(chain.waypoints[-1].orientation_relative_to_previous)
        np.testing.assert_allclose(chain.waypoints[-1].orientation, np.eye(3), atol=1e-12)
        self.assertEqual(skill['orientation_scheme'], 'precontact_object_and_previous')
        from waypoint_skill_editor import nominal_targets
        candidate = skill['candidates'][0]
        poses = {'source': tuple(np.asarray(v) for v in candidate['initial_source']),
                 'destination': tuple(np.asarray(v) for v in candidate['initial_destination'])}
        targets = nominal_targets(skill, poses, (np.zeros(3), np.eye(3)))
        # Row 2 is the approach onto the object, row 5 the loaded traverse.
        np.testing.assert_allclose(targets[1][1],
            poses['source'][1] @ np.asarray(candidate['approach_orientation']))
        np.testing.assert_allclose(targets[4][1],
            poses['destination'][1] @ np.asarray(candidate['transport_orientation']))

    def test_drawer_push_comes_from_joint_motion_before_grasp(self):
        joint = 'wooden_cabinet_1_top_level'
        frame = [np.array([0., -.23, .9]).tolist(), np.eye(3).tolist()]
        trajectory = []
        for i in range(32):
            q = -min(max(i-8, 0)*.012, .16)
            trajectory.append(dict(position=[.04, -.09-q, 1.1],
                orientation=np.eye(3).tolist(), gripper='open',
                grasped=['bowl'] if i >= 27 else [],
                joint_positions={joint:q}, object_poses={joint:frame}))
        demo = dict(success=True, placement_mode='fixed_unshuffled',
                    joints={joint:{}}, samples=trajectory,
                    terminal_sample=trajectory[-1])
        push = learn_pregrasp_push([demo], 'wooden_cabinet_1_top_region')
        self.assertEqual(push['joint'], joint)
        self.assertAlmostEqual(push['target_joint'], -.16)
        self.assertEqual(len(push['candidates']), 1)
        for key in ('approach', 'contact', 'finish'):
            self.assertEqual(len(push['candidates'][0][key]), 2)
        # Motion occurring only after the bowl is grasped is not an opening
        # prerequisite and must not be replayed before the grasp.
        for i, sample in enumerate(trajectory):
            sample['grasped'] = ['bowl'] if i >= 10 else []
        self.assertIsNone(learn_pregrasp_push([demo], 'wooden_cabinet_1_top_region'))

    def test_two_object_sequence_repeats_every_single_reference_row(self):
        from waypoint_skill import learn_sequential_skill
        from waypoint_skill_editor import fit_editor_model, nominal_targets
        from waypoint_gmm import chain_from_model
        demos = [two_stage_placed(bend=-.01), two_stage_placed(bend=.01)]
        skill = learn_sequential_skill(demos)
        self.assertEqual([stage['source'] for stage in skill['stages']],
                         ['blocker', 'source'])
        self.assertEqual(len(skill['primitives']), 10)
        model = fit_editor_model(demos)
        chain = chain_from_model(model)
        self.assertEqual(len(chain.waypoints), 14)
        self.assertEqual([w.skill_role for w in chain.waypoints],
                         ['move','move','grasp','transport','transport','place','release']*2)
        self.assertTrue(chain.waypoints[6].orientation_relative_to_previous)
        self.assertTrue(chain.waypoints[13].orientation_relative_to_previous)
        poses = {}
        for stage in model['object_relative_skill']['stages']:
            candidate = stage['candidates'][0]
            poses[stage['source']] = tuple(np.asarray(v) for v in candidate['initial_source'])
            poses[stage['destination']] = tuple(np.asarray(v) for v in candidate['initial_destination'])
        self.assertEqual(len(nominal_targets(model['object_relative_skill'],
                                              poses, (np.zeros(3), np.eye(3)))), 14)

    def test_world_transform_does_not_change_contact_relationships(self):
        demo = placed();other = copy.deepcopy(demo)
        frame = (np.array([.3, -.4, .1]), Rotation.from_rotvec([0, 0, .8]).as_matrix())
        for sample in other['samples']+[other['terminal_sample']]:
            p, r = compose(frame, (sample['position'], sample['orientation']))
            sample['position'], sample['orientation'] = p.tolist(), r.tolist()
            for name, pose in sample['object_poses'].items():
                p, r = compose(frame, pose)
                sample['object_poses'][name] = [p.tolist(), r.tolist()]
        a = learn_skill([demo])['candidates'][0]
        b = learn_skill([other])['candidates'][0]
        for key in ['grasp','release_object']:
            np.testing.assert_allclose(a[key][0], b[key][0], atol=1e-12)
            np.testing.assert_allclose(a[key][1], b[key][1], atol=1e-12)


class SkillEditorTest(unittest.TestCase):
    def test_model_chain_roundtrip_and_manual_edit_guard(self):
        import tempfile
        from pathlib import Path
        from waypoint_skill_editor import fit_editor_model
        from waypoint_learning import save_model, load_model
        from waypoint_gmm import chain_from_model, GMMChain
        model = fit_editor_model([placed(), placed()])
        self.assertEqual(model['model_type'], 'object_relative_vla_waypoints')
        with tempfile.TemporaryDirectory() as folder:
            save_model(model, folder)
            loaded = load_model(folder)
            chain = chain_from_model(loaded, str(Path(folder)/'model.json'))
            chain.save(Path(folder)/'plan.json')
            chain = GMMChain.load(Path(folder)/'plan.json', 'synthetic', 0)
            self.assertEqual(len(chain.waypoints), 7)
            self.assertTrue(chain.execution_skill)
            chain.waypoints[2].offset[0] += .01
            with self.assertRaisesRegex(ValueError, 'keyframes were changed'):
                chain.validate()
            chain.execution_skill = {}
            chain.validate()

    def test_legacy_adaptive_plan_without_orientation_flag_still_loads(self):
        import json
        import tempfile
        from pathlib import Path
        from waypoint_skill_editor import fit_editor_model
        from waypoint_gmm import chain_from_model, GMMChain
        chain = chain_from_model(fit_editor_model([placed(), placed()]))
        data = {'version': 2, 'kind': 'gmm_editor_chain', **__import__('dataclasses').asdict(chain)}
        for waypoint in data['waypoints']:
            waypoint.pop('orientation_relative_to_previous')
        for waypoint in data['execution_skill']['waypoint_template']:
            waypoint.pop('orientation_relative_to_previous')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'plan.json'
            path.write_text(json.dumps(data))
            loaded = GMMChain.load(path, 'synthetic', 0)
        self.assertFalse(loaded.waypoints[-1].orientation_relative_to_previous)

    def test_archived_recordings_require_the_same_task_and_original_setup(self):
        import json
        import tempfile
        from pathlib import Path
        from waypoint_skill_editor import archived_matching_demos, can_fit_editor_model
        current, incompatible = placed(), placed()
        # A second transfer nothing demonstrates leaves no semantic skill to
        # fit, so only shared event signatures can supply a plan here.
        current['goals'].append(['on', 'blocker', 'destination'])
        incompatible['goals'].append(['on', 'blocker', 'destination'])
        for i in range(40, 44):
            incompatible['samples'][i]['contact_state'] = {}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root/'older'/'demonstrations'
            archive.mkdir(parents=True)
            valid = copy.deepcopy(current)
            wrong_setup = copy.deepcopy(current)
            wrong_setup['initial_object_poses']['source'][0][0] += .1
            wrong_task = copy.deepcopy(current)
            wrong_task['task_id'] = 1
            for i, demo in enumerate((valid, wrong_setup, wrong_task)):
                (archive/f'demo_{i:03d}.json').write_text(
                    json.dumps(demo, default=lambda value: value.item()
                               if isinstance(value, np.generic) else value.tolist()))
            matching = archived_matching_demos([current, incompatible], root)
            self.assertEqual(len(matching), 1)
            self.assertTrue(matching[0]['_path'].endswith('demo_000.json'))
            self.assertFalse(can_fit_editor_model([current, incompatible]))
            self.assertTrue(can_fit_editor_model([current, incompatible], root))

    def test_manual_commit_explicitly_disables_adaptive_execution(self):
        from unittest.mock import patch
        from waypoint_skill_editor import fit_editor_model
        from waypoint_gmm import chain_from_model
        from waypoint_gmm_editor import GMMPanel
        from subgoal_gui import Panel
        panel = object.__new__(GMMPanel)
        panel.chain = chain_from_model(fit_editor_model([placed(), placed()]))
        with patch.object(Panel, 'commit'):
            panel.commit(2)
        self.assertFalse(panel.chain.execution_skill)
        self.assertIn('manual replay', panel.chain.source)

    def test_single_waypoint_move_never_runs_complete_skill(self):
        from unittest.mock import patch
        from waypoint_skill_editor import fit_editor_model
        from waypoint_gmm import chain_from_model
        from waypoint_gmm_editor import GMMSession
        from waypoint_session import Session
        chain = chain_from_model(fit_editor_model([placed(), placed()]))
        session = object.__new__(GMMSession)
        with patch.object(session, 'editor_target', return_value=(np.zeros(3), np.eye(3))), \
             patch.object(Session, 'test_waypoint', return_value='Finished') as move:
            session.test_waypoint(chain, 2, chain.waypoints[2], lambda *_: None)
        submitted = move.call_args[0][0]
        self.assertFalse(submitted.execution_skill)
        self.assertEqual(len(submitted.waypoints), 1)


def drawer_demo(joint='wooden_cabinet_1_middle_level', travel=.16, grasp_from=None,
                descent=.0008, goals=None):
    """An open-hand pull that moves a measured slide joint."""
    frame = [np.array([0., -.23, .9]).tolist(), np.eye(3).tolist()]
    trajectory = []
    for i in range(32):
        q = -min(max(i-8, 0)*travel/12, travel)
        trajectory.append(dict(position=[.04, -.09+q, 1.1+q/travel*descent],
            orientation=np.eye(3).tolist(), gripper='open', timestamp=i*.05,
            grasped=['bowl'] if grasp_from is not None and i >= grasp_from else [],
            joint_positions={joint: q}, object_poses={joint: frame}, contact_state={}))
    return dict(version=2, suite='synthetic', task_id=7, success=True,
                placement_mode='fixed_unshuffled', joints={joint: {}},
                language='open the middle drawer', instruction='open the middle drawer',
                objects=[], fixtures=['wooden_cabinet_1'], poses={joint: frame},
                initial_object_poses={joint: frame},
                goals=goals or [['open', 'wooden_cabinet_1_middle_region']],
                samples=trajectory, terminal_sample=trajectory[-1],
                end_position=trajectory[-1]['position'],
                end_orientation=trajectory[-1]['orientation'])


class MinimalPlanShapeTest(unittest.TestCase):
    """The plan the editor lists is the plan the executor aims at."""

    def test_articulation_only_task_learns_a_standoff_contact_and_push(self):
        from waypoint_skill import learn_push_skill
        from waypoint_skill_editor import fit_editor_model
        from waypoint_gmm import chain_from_model
        demos = [drawer_demo(), drawer_demo(travel=.17)]
        skill = learn_push_skill(demos)
        self.assertEqual(skill['primitives'], ['move', 'push'])
        self.assertEqual(skill['candidates'], [])
        chain = chain_from_model(fit_editor_model(demos))
        self.assertEqual([w.skill_role for w in chain.waypoints], ['move', 'move', 'push'])
        self.assertEqual({w.reference for w in chain.waypoints},
                         {'wooden_cabinet_1_middle_level'})
        self.assertEqual(chain.waypoints[0].gripper, 'open')

    def test_articulation_learner_rejects_a_pick_and_place_goal(self):
        from waypoint_skill import learn_push_skill
        with self.assertRaisesRegex(ValueError, 'single articulated goal'):
            learn_push_skill([placed(), placed()])

    def test_push_keeps_the_measured_finish_pose(self):
        """A squared-off pull loses the handle; the descent is real evidence."""
        from waypoint_skill import learn_pregrasp_push
        demo = drawer_demo(grasp_from=27, descent=-.0008)
        push = learn_pregrasp_push([demo], 'wooden_cabinet_1_middle_region')
        contact, finish = push['candidates'][0]['contact'], push['candidates'][0]['finish']
        self.assertNotAlmostEqual(finish[0][2], contact[0][2], places=5)

    def test_connectors_move_along_one_reference_axis(self):
        from waypoint_skill_editor import fit_editor_model, nominal_targets, skill_waypoints
        model = fit_editor_model([placed(bend=-.01), placed(), placed(bend=.01)])
        skill = model['object_relative_skill']
        candidate = skill['candidates'][0]
        yaw = Rotation.from_rotvec([0, 0, .4]).as_matrix()
        poses = {'source': (np.asarray(candidate['initial_source'][0]), yaw),
                 'destination': (np.asarray(candidate['initial_destination'][0]), yaw)}
        start = (np.zeros(3), np.eye(3))
        rows = skill_waypoints(skill, nominal_targets(
            skill, poses, start, obstacles=lambda a, b, ignore: .3), poses, start)
        # Approach onto the grasp, grasp up into the lift, traverse down to the
        # placement: each pair shares a reference and must move on one axis.
        for first, second in ((1, 2), (2, 3), (4, 5)):
            a, b = rows[first], rows[second]
            self.assertEqual(a.reference, b.reference)
            moved = np.abs(np.asarray(b.offset)-a.offset) > 1e-9
            self.assertEqual(int(moved.sum()), 1,
                             (first, second, a.name, b.name, a.offset, b.offset))

    def test_stored_offsets_resolve_to_the_previewed_targets(self):
        from waypoint_chain import resolve
        from waypoint_skill_editor import (fit_editor_model, nominal_targets,
                                           skill_waypoints, effective_poses)
        model = fit_editor_model([placed(bend=-.01), placed(), placed(bend=.01)])
        skill = model['object_relative_skill']
        candidate = skill['candidates'][0]
        poses = {'source': tuple(np.asarray(v) for v in candidate['initial_source']),
                 'destination': tuple(np.asarray(v) for v in candidate['initial_destination'])}
        start = (np.zeros(3), np.eye(3))
        targets = nominal_targets(skill, poses, start,
                                  obstacles=lambda a, b, ignore: .4,
                                  bottom=lambda name: 0.)
        rows = skill_waypoints(skill, targets, poses, start)
        stored, previous = effective_poses(skill, poses), start
        for waypoint, (position, orientation) in zip(rows, targets):
            resolved = resolve(waypoint, stored, previous)
            np.testing.assert_allclose(resolved[0], position, atol=1e-12)
            np.testing.assert_allclose(resolved[1], orientation, atol=1e-12)
            previous = position, orientation

    def test_carried_traverse_clears_a_tall_obstacle(self):
        """Only what the move passes over has to be cleared."""
        from waypoint_skill import carry_height, clearance_height
        approach = clearance_height(.10, 0., .18, obstacle_top=.30)
        self.assertAlmostEqual(approach, .38)
        # A can hanging 12 cm below the wrist needs the wrist that much higher.
        self.assertAlmostEqual(carry_height(approach, .40, .30, .28), .44)
        self.assertEqual(carry_height(approach, .40, None, None), approach)

    def test_fallback_gmm_plan_is_reduced_to_primitive_boundaries(self):
        from waypoint_skill_editor import minimal_gmm_waypoints
        nodes = ([dict(primitive='move', gripper_command='keep')]*4
                 + [dict(primitive='grasp', gripper_command='keep')]*3
                 + [dict(primitive='grasp', gripper_command='close')]
                 + [dict(primitive='transport', gripper_command='keep')]*5
                 + [dict(primitive='place', gripper_command='open')]
                 + [dict(primitive='release', gripper_command='keep')]*2)
        reduced = minimal_gmm_waypoints(dict(waypoints=nodes))
        self.assertEqual([n['primitive'] for n in reduced['waypoints']],
                         ['move', 'grasp', 'transport', 'place', 'release'])
        self.assertEqual([n['gripper_command'] for n in reduced['waypoints']],
                         ['keep', 'close', 'keep', 'open', 'keep'])
        self.assertEqual(reduced['waypoint_reduction'], dict(
            policy='primitive_boundaries', before=16, after=5))
        # A skill plan is already minimal and is never rewritten.
        skill = dict(object_relative_skill=dict(kind='x'), waypoints=nodes)
        self.assertIs(minimal_gmm_waypoints(skill), skill)


class MinimalPlanCoverageTest(unittest.TestCase):
    """Task shapes option 2 has to reduce to the six primitives."""

    def demos_with(self, goals, **kwargs):
        demos = []
        for bend in (-.01, .01):
            demo = placed(bend=bend, **kwargs)
            demo['goals'] = [list(goal) for goal in goals]
            for sample in demo['samples']+[demo['terminal_sample']]:
                sample['goal_predicates'] = [True]*len(goals)
            demos.append(demo)
        return demos

    def test_articulation_goal_beside_a_transfer_adds_one_push(self):
        """"Turn on the stove and put the pot on it" is one transfer."""
        from waypoint_skill import learn_skill
        demos = self.demos_with([['on', 'source', 'destination'], ['turnon', 'stove']])
        skill = learn_skill(demos)
        self.assertEqual(skill['source'], 'source')
        self.assertEqual(skill['primitives'],
                         ['move', 'grasp', 'transport', 'place', 'release'])

    def test_two_spatial_goals_are_never_one_transfer(self):
        from waypoint_skill import learn_skill
        demos = self.demos_with([['on', 'source', 'destination'],
                                 ['on', 'other', 'destination']])
        with self.assertRaises(ValueError):
            learn_skill(demos)

    def test_placement_without_a_recorded_release_is_accepted_when_settled(self):
        """LIBERO stops the episode before the fingers ever open."""
        from waypoint_skill import learn_skill
        demos = []
        for bend in (-.01, .01):
            demo = placed(bend=bend)
            samples = demo['samples']+[demo['terminal_sample']]
            opened = next(i for i, s in enumerate(samples)
                          if s['gripper'] == 'open' and i > 30)
            demo['samples'] = samples[:opened]
            demo['terminal_sample'] = samples[opened-1]
            for sample in demo['samples']+[demo['terminal_sample']]:
                sample['goal_predicates'] = [True]
            demos.append(demo)
        self.assertTrue(all(s['gripper'] == 'close'
                            for s in demos[0]['samples'][-3:]))
        skill = learn_skill(demos)
        self.assertEqual(len(skill['candidates']), 2)
        # The same demonstrations with the goal unmet are not a placement.
        for demo in demos:
            for sample in demo['samples']+[demo['terminal_sample']]:
                sample['goal_predicates'] = [False]
        with self.assertRaises(ValueError):
            learn_skill(demos)

    def test_last_completed_hold_is_the_one_learned(self):
        """A fumble and re-grasp must not define the placement."""
        from waypoint_skill import learn_skill
        clean = learn_skill([placed(), placed(bend=.01)])['candidates'][0]
        demos = []
        for bend in (0., .01):
            demo = placed(bend=bend)
            samples = demo['samples']
            # Drop the object early, then pick it up again and finish.
            for sample in samples[20:24]:
                sample['gripper'], sample['grasped'] = 'open', []
            demos.append(demo)
        fumbled = learn_skill(demos)['candidates'][0]
        self.assertGreater(fumbled['grasp_sample'], 24)
        np.testing.assert_allclose(fumbled['release_object'][0],
                                   clean['release_object'][0], atol=1e-9)

    def test_seated_placement_is_judged_by_the_object_not_the_wrist(self):
        """A wrist that stops high has still placed the object it was holding."""
        from waypoint_skill import seated

        class Session:
            def __init__(self, positions):
                self.positions = list(positions)
                self.world = type('W', (), {'pos': lambda _, name: self.positions.pop(0)})()
                self.executor = type('E', (), {'hold': lambda *_: None})()
        goal = np.array([.1, .2, .3])
        rested = Session([goal+[0, 0, .004], goal+[0, 0, .004]])
        self.assertIn('seated', seated(rested, 'bowl', goal, 'failed: short'))
        # Still moving: the object may be sliding out, not resting.
        moving = Session([goal, goal+[0, .01, 0]])
        self.assertEqual(seated(moving, 'bowl', goal, 'failed: short'), 'failed: short')
        # At rest, but nowhere near where the demonstration left it.
        astray = Session([goal+[.09, 0, 0], goal+[.09, 0, 0]])
        self.assertEqual(seated(astray, 'bowl', goal, 'failed: short'), 'failed: short')


class PickPlaceProgressTest(unittest.TestCase):
    def replay(self, retry=False, stages=1, retreat=False, keep_going=False,
               completions=None, abort_motion=False, failed_close=False):
        from waypoint_skill import run_skill
        wrist = [np.array([0., -.2, .9]), np.eye(3)]
        source = np.array([0., 0., .5])
        destination = np.array([.2, .1, .5])
        state = dict(held=False, closes=0, checks=0, row=None, status=None)
        events, actions = [], []

        def report(row, status):
            events.append((row, status))
            state.update(row=row, status=status)

        def line(position, orientation):
            actions.append(('move', state['row'], state['status']))
            if abort_motion and state['row'] == 2:
                return 'no collision-free route to the target'
            wrist[:] = [np.asarray(position).copy(), np.asarray(orientation).copy()]
            if state['held']:
                source[:] = wrist[0]
            return 'ok'

        def grip(sg):
            actions.append(('close' if sg.gripper > 0 else 'open', state['row'], state['status']))
            if sg.gripper > 0:
                state['closes'] += 1
                state['held'] = not failed_close and (not retry or state['closes'] > 1)
                if failed_close:
                    return 'failed: injected grasp loss; fingers stayed open'
            else:
                state['held'] = False
            return 'ok'

        def checked_grasp(*_):
            state['checks'] += 1
            return state['held']

        def finished(row, status):
            if completions is not None:
                completions.append((row, status, state.copy(), actions.copy()))

        domain = SimpleNamespace(
            robots=[SimpleNamespace(gripper='fingers')],
            objects_dict={'source': SimpleNamespace(contact_geoms=['can'])},
            _check_grasp=checked_grasp)
        session = SimpleNamespace(
            continue_on_failure=keep_going,
            world=SimpleNamespace(domain=domain, pos=lambda _: source.copy(),
                aabb=lambda _: (source-[.02, .02, .02], source+[.02, .02, .02])),
            executor=SimpleNamespace(line=line, run_subgoal=grip, hold=lambda _: None),
            env=SimpleNamespace(check_success=lambda: not retreat),
            poses=lambda: {'source': (source.copy(), np.eye(3)),
                           'destination': (destination.copy(), np.eye(3))},
            gripper_pose=lambda: tuple(v.copy() for v in wrist))
        candidate = dict(grasp=([0., 0., 0.], np.eye(3).tolist()),
                         transport_height=.3,
                         release_object=([0., 0., 0.], np.eye(3).tolist()))
        skill = dict(source='source', destination='destination',
                     candidates=[candidate, candidate],
                     primitives=['move', 'move', 'grasp', 'transport', 'transport', 'place', 'release'])
        if stages > 1:
            skill = dict(stages=[copy.deepcopy(skill) for _ in range(stages)])
        with patch('waypoint_skill.path_obstacle_top', return_value=.5), \
                patch('waypoint_skill.supporting_bodies', return_value=()), \
                patch('waypoint_gmm.check_learned_pose', side_effect=lambda s, w, r, p, o: r):
            result = run_skill(session, skill, report, waypoint_finished=finished)
        if abort_motion or failed_close:
            self.assertFalse(result['completed'], result)
        elif keep_going and retry:
            self.assertFalse(result['completed'], result)
            self.assertEqual(result['result'], 'failed: nothing between the fingers')
        else:
            self.assertTrue(result['completed'], result)
        return events, actions

    def test_completion_waits_for_close_and_physical_grasp_check(self):
        for missed in (False, True):
            with self.subTest(missed=missed):
                completions = []
                self.replay(retry=missed, keep_going=True, completions=completions)
                completions = [item for item in completions if item[0] == 2]
                self.assertEqual(len(completions), 1)
                row, status, state, actions = completions[0]
                self.assertEqual(row, 2)
                self.assertEqual(state['closes'], 1)
                self.assertEqual(state['checks'], 1)
                self.assertEqual(state['held'], not missed)
                self.assertEqual(actions[-1][0], 'close')
                self.assertTrue(status.startswith('failed' if missed else 'ok'))

    def test_path_failure_before_close_has_no_grasp_completion(self):
        completions = []
        _, actions = self.replay(abort_motion=True, keep_going=True, completions=completions)
        self.assertEqual(completions, [])
        self.assertFalse(any(command == 'close' for command, _, _ in actions))

    def test_injected_close_failure_finishes_only_after_physical_check(self):
        completions = []
        self.replay(failed_close=True, keep_going=True, completions=completions)
        self.assertEqual(len(completions), 1)
        _, status, state, _ = completions[0]
        self.assertEqual(state['closes'], 1)
        self.assertEqual(state['checks'], 1)
        self.assertFalse(state['held'])
        self.assertIn('injected grasp loss', status)

    def test_completion_rows_follow_sequential_object_offsets(self):
        completions = []
        self.replay(stages=2, completions=completions)
        self.assertEqual([row for row, *_ in completions], [2, 3, 4, 9, 10, 11])

    def test_transport_completion_follows_full_movement_and_load_check(self):
        completions = []
        self.replay(completions=completions)
        transport = [item for item in completions if item[0] in (3, 4)]
        self.assertEqual([row for row, *_ in transport], [3, 4])
        for row, status, state, actions in transport:
            self.assertTrue(status.startswith('ok'))
            self.assertTrue(state['held'])
            self.assertGreaterEqual(state['checks'], 2)
            self.assertEqual(actions[-1], ('move', row, 'Running'))

    def test_continuing_playback_exits_failed_grasp_without_replaying_move_rows(self):
        events, actions = self.replay(retry=True, keep_going=True)
        starts = [row for row, status in events if status == 'Running']
        self.assertEqual(starts, [0, 0, 1, 2])
        self.assertEqual([row for command, row, _ in actions if command == 'close'], [2])

    def test_grasp_retry_revisits_approach_rows_without_shifting_transport(self):
        events, actions = self.replay(retry=True)
        starts = [row for row, status in events if status == 'Running']
        self.assertEqual(starts, [0, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual([row for command, row, _ in actions if command == 'close'], [2, 2])
        self.assertIn((2, 'failed: nothing between the fingers'), events)
        self.assertTrue(all(status == 'Running' for command, _, status in actions if command == 'close'))

    def test_release_is_active_before_fingers_open(self):
        for retreat in (False, True):
            with self.subTest(retreat=retreat):
                events, actions = self.replay(retreat=retreat)
                releases = [(row, status) for command, row, status in actions if command == 'open']
                self.assertEqual(releases[-1], (6, 'Running'))
                self.assertEqual(events[-1], (6, 'ok (released)'))
                if retreat:
                    self.assertEqual(actions[-1], ('move', 6, 'Running'))

    def test_retry_in_first_object_does_not_shift_second_object_rows(self):
        events, actions = self.replay(retry=True, stages=2)
        starts = [row for row, status in events if status == 'Running']
        self.assertEqual(starts[-8:], [7, 7, 8, 9, 10, 11, 12, 13])
        self.assertEqual([row for command, row, _ in actions if command == 'close'], [2, 2, 9])
        self.assertEqual(events[-1], (13, 'ok (released)'))


class ObjectPushExecutionTest(unittest.TestCase):
    def test_push_tracks_planar_goal_without_following_tip_or_repeating_targets(self):
        from waypoint_skill import run_object_push
        for behavior in ('advancing', 'stalled', 'sideways'):
            with self.subTest(behavior=behavior):
                source = np.array([0., 0., .9])
                basis = Rotation.from_euler('z', .4).as_matrix()
                wrist = [np.array([0., 0., 1.1]), np.eye(3)]
                row = [0]
                pushes = []

                def line(position, orientation):
                    wrist[:] = [np.array(position).copy(), np.array(orientation).copy()]
                    if row[0] == 2 and position[2] < .97:
                        pushes.append(wrist[0].copy())
                        if behavior == 'advancing':
                            source[0] += .02
                        elif behavior == 'sideways':
                            source[1] += .015
                        # Simulate rocking and lifting under contact load.
                        source[2] = .94
                        basis[:] = Rotation.from_euler('xy', [.5, -.3]).as_matrix()
                    return 'ok'

                session = SimpleNamespace(
                    executor=SimpleNamespace(line=line, run_subgoal=lambda _: 'ok'),
                    world=SimpleNamespace(pos=lambda _: source.copy()),
                    env=SimpleNamespace(check_success=lambda: source[0] >= .095),
                    poses=lambda: {'plate': (source.copy(), basis.copy()),
                                   'goal': (np.zeros(3), np.eye(3))},
                    gripper_pose=lambda: tuple(v.copy() for v in wrist))
                stroke = dict(contact=([-.02, 0., .02], np.eye(3).tolist()),
                              target_object=([.1, 0., .9], np.eye(3).tolist()), travel=.1)
                shove = dict(source='plate', destination='goal', travel=.1,
                             candidates=[dict(strokes=[stroke])])
                with patch('waypoint_skill.reachable', return_value=True), \
                        patch('waypoint_skill.guarded_move', return_value=(None, 'unchecked')), \
                        patch('waypoint_gmm.check_learned_pose', side_effect=lambda s, w, r, p, o: r):
                    result = run_object_push(session, shove, lambda i, _: row.__setitem__(0, i))
                trace = [t for t in result['waypoints'] if t['primitive'] == 'push']
                self.assertEqual(result['completed'], behavior == 'advancing')
                self.assertGreaterEqual(len(trace), 4)
                for step in trace:
                    self.assertAlmostEqual(step['position'][2], .92)
                    np.testing.assert_allclose(step['orientation'],
                                               Rotation.from_euler('z', .4).as_matrix())
                self.assertTrue(np.all(np.diff(np.array(pushes)[:, 0]) > 0))
                if behavior != 'advancing':
                    self.assertEqual(len(trace), 4)
                    self.assertIn('stopped moving', result['result'])


if __name__ == '__main__':unittest.main()
