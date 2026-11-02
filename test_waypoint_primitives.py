"""Primitive evidence and fixed-scene learning under unseen object placements."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from test_waypoint_learning import placed, rebuild
from waypoint_learning import Config, detect_events, learn
from waypoint_primitives import PRIMITIVES
from waypoint_replay import dense_nodes, reconstruct_fused, reconstruct_target
from waypoint_gmm import chain_from_model


class PrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = placed()
        cls.model = learn([copy.deepcopy(cls.demo), copy.deepcopy(cls.demo)],
                          Config(samples_per_interval=8, components=2, max_waypoints=200))

    def test_standard_sequence_and_preserved_gripper_commands(self):
        phases = [s['primitive'] for s in self.model['segments']]
        unique = [p for i, p in enumerate(phases) if i == 0 or p != phases[i-1]]
        self.assertEqual(unique, ['move', 'grasp', 'transport', 'place', 'release', 'move'])
        self.assertTrue(set(phases) <= set(PRIMITIVES))
        commands = {w['semantic_event']['key']: w['gripper_command'] for w in self.model['waypoints']
                    if w['gripper_command'] != 'keep'}
        self.assertEqual(list(commands.values()), ['close', 'open'])
        self.assertTrue(all(w.get('primitive') for w in dense_nodes(self.model)))

    def test_push_requires_ungrasped_contact_and_motion(self):
        d = copy.deepcopy(self.demo)
        for i, s in enumerate(d['samples']+[d['terminal_sample']]):
            s['gripper'] = 'open'
            s['grasped'] = []
            s['contact_state'] = {'source': {'kind': 'object', 'both': False}} if 10 <= i < 40 else {}
            s['object_poses']['source'][0] = [max(0, min(i-10, 30))*.003, 0., 0.]
        self.assertIn('push', [e['primitive'] for e in detect_events(d)])
        for s in d['samples']+[d['terminal_sample']]:
            s['object_poses']['source'][0] = [0., 0., 0.]
        self.assertNotIn('push', [e['primitive'] for e in detect_events(d)])

    def test_rotational_push_requires_contact_and_no_grasp(self):
        from waypoint_primitives import label_samples
        samples = [dict(position=[0, 0, 0], grasped=[],
                        contact_state={'source': {'both': False}},
                        object_poses={'source': ([0, 0, 0],
                            Rotation.from_rotvec([0, 0, i*.02]).as_matrix())})
                   for i in range(5)]
        self.assertEqual(label_samples(samples, []), ['push']*5)
        for sample in samples:
            sample['contact_state'] = {}
        self.assertEqual(label_samples(samples, []), ['move']*5)
        for sample in samples:
            sample['contact_state'] = {'source': {'both': True}}
            sample['grasped'] = ['source']
        self.assertNotIn('push', label_samples(samples, []))

    def test_drawer_joint_travel_is_push_without_reported_contact(self):
        from waypoint_primitives import label_samples
        samples = [dict(position=[0., i*.01, 1.], grasped=[],
                        contact_state={}, joint_positions={'drawer_joint': -i*.01})
                   for i in range(5)]
        self.assertEqual(label_samples(samples, []), ['push']*5)
        for sample in samples:
            sample['grasped'] = ['bowl']
        self.assertNotIn('push', label_samples(samples, []))

    def test_model_discovery_prefers_connected_primitives_then_recency(self):
        from waypoint_gmm_editor import GMMPanel
        panel = object.__new__(GMMPanel)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, connected, timestamp in (('a_current', True, 30),
                                                ('b_older', True, 20),
                                                ('z_legacy', False, 40)):
                path = root/name/'model.json'
                path.parent.mkdir()
                node = dict(primitive='move', retargeting='task_endpoint') if connected else {}
                path.write_text(json.dumps(dict(model_type='task_parameterized_gmm_gmr',
                                                suite='synthetic', task_id=0, waypoints=[node])))
                os.utime(path, (timestamp, timestamp))
            with patch('waypoint_gmm_editor.MODEL_STORE', root):
                panel.refresh_model_index()
            self.assertEqual(panel.model_index[('synthetic', 0)], root/'a_current/model.json')

    def test_fixed_training_scene_tracks_independently_shuffled_objects(self):
        home = np.array([.07, -.18, .24])
        for layout in ({'source': [.30, .22, 0], 'destination': [-.18, -.25, 0]},
                       {'source': [-.22, .18, 0], 'destination': [.28, -.10, 0]}):
            built = rebuild(self.model, layout, home)
            np.testing.assert_allclose(built[0][1], home, atol=1e-12)
            for node, p, r in built:
                if node['gripper_command'] != 'keep':
                    name = node['reference']['name']
                    target = reconstruct_target(node, (np.array(layout[name]), np.eye(3)))
                    np.testing.assert_allclose(p, target[0], atol=1e-12)
                    np.testing.assert_allclose(r, target[1], atol=1e-12)

    def test_move_transport_first_target_and_endpoint(self):
        anchor = (np.array([.3, -.2, .4]), Rotation.from_rotvec([.1, .2, -.3]).as_matrix())
        for si, segment in enumerate(self.model['segments']):
            if segment['primitive'] not in ('move', 'transport'):
                continue
            nodes = [n for n in self.model['waypoints'] if n['segment'] == si]
            self.assertEqual(len(nodes), 2)
            self.assertTrue(all(len(n['frames']) == 1 for n in nodes))
            frames = [(np.array([.6, .1, .2]), np.eye(3)) for _ in nodes[0]['frames']]
            p, r = reconstruct_fused(nodes[0], frames, anchor)
            np.testing.assert_allclose(p, anchor[0], atol=1e-12)
            np.testing.assert_allclose(r, anchor[1], atol=1e-12)
            p, r = reconstruct_fused(nodes[-1], frames, anchor)
            expected = reconstruct_target(nodes[-1], frames[0])
            np.testing.assert_allclose(p, expected[0], atol=1e-12)
            np.testing.assert_allclose(r, expected[1], atol=1e-12)

    def test_editor_preserves_frames_and_uses_same_targets(self):
        from waypoint_gmm_editor import GMMSession
        chain = chain_from_model(self.model)
        session = object.__new__(GMMSession)
        session.reference_poses = {n: (np.array(p), np.array(r)) for n, (p, r) in self.demo['poses'].items()}
        session.reference_poses['source'][0][:2] = [.3, .2]
        session.reference_poses['destination'][0][:2] = [-.2, -.1]
        session.start_pose = (np.array([0, -.12, .2]), np.eye(3))
        session.reference_carry = (np.zeros(3), 0.)
        session.reference_bottom = {}
        result = session.preview(chain)
        context, previous = {}, session.start_pose
        with patch.object(session, 'poses', return_value=session.reference_poses):
            for w, expected in zip(chain.waypoints, result):
                self.assertIn(w.name.lower(), PRIMITIVES)
                self.assertEqual(w.learned['node']['frames'], self.model['waypoints'][chain.waypoints.index(w)]['frames'])
                with patch.object(session, 'gripper_pose', return_value=previous):
                    actual = session.resolve_target(w, session.reference_poses, previous, context)
                np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
                previous = actual


if __name__ == '__main__':
    unittest.main()
