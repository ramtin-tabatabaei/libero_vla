"""Headless LIBERO adapter tests; no VLA server or trained policy required."""
import dataclasses
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import subgoal_planner as sp
from waypoint_gmm_editor import GMMSession
from waypoint_chain import relative_waypoint
from waypoint_observation import episode_metadata,measured_sample
from waypoint_replay import replay,reconstruct_target,shift_object,shuffle_objects
from waypoint_learning import Config


class LearningSimulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session=GMMSession(sp.Args());cls.session.load('libero_object',0,cameras=False)
        cls.initial=cls.session.env.sim.get_state().flatten().copy()

    @classmethod
    def tearDownClass(cls):cls.session.env.close()

    def setUp(self):
        self.s=self.session;self.s.env.set_init_state(self.initial);self.s.attach(settle=False)
        self.s.last_shuffle_mapping = None

    def test_measurement_schema_and_predicates(self):
        metadata=episode_metadata(self.s);sample=measured_sample(self.s,self.s.env.env._get_observations(),[0]*6+[-1],0)
        self.assertEqual(metadata['version'],2);self.assertTrue(metadata['goals'])
        for key in ('timestamp','eef_linear_velocity','eef_angular_velocity','gripper_qvel','contact_state','object_poses','goal_predicates','joint_positions'):
            self.assertIn(key,sample)
        np.testing.assert_allclose(sample['position'],self.s.gripper_pose()[0]);np.testing.assert_allclose(sample['orientation'],self.s.gripper_pose()[1])

    def test_record_measured_terminal_and_restore(self):
        from waypoint_demo import record
        class Policy:
            def __init__(self,port):pass
            def connect(self):return {}
            def infer(self,obs):return {'actions':np.tile([0]*6+[-1],(5,1))}
            def close(self):pass
        before=self.s.env.sim.get_state().flatten().copy()
        with patch('waypoint_demo.PolicyConnection',side_effect=Policy) as connection,patch('waypoint_demo.observation',lambda *_:{}):
            record(self.s,self.s.task.language,max_steps=3)
        connection.assert_called_once_with(self.s.args.policy_port)
        d=self.s.last_demonstration
        self.assertEqual(d['episode_length'],3);self.assertEqual(d['terminal_sample']['step'],3)
        self.assertGreater(d['terminal_sample']['timestamp'],d['samples'][-1]['timestamp'])
        np.testing.assert_allclose(self.s.env.sim.get_state().flatten(),before,atol=1e-12)

    def test_record_respects_custom_configured_and_explicit_ports(self):
        from waypoint_demo import record
        original_port = self.s.args.policy_port
        self.s.args.policy_port = 8123
        try:
            class Policy:
                def __init__(self, port): pass
                def connect(self): return {}
                def infer(self, obs): return {'actions':np.tile([0]*6+[-1], (5,1))}
                def close(self): pass
            for kwargs, expected in (({},8123), ({'port':8000},8000)):
                with self.subTest(port=expected):
                    with patch('waypoint_demo.PolicyConnection',side_effect=Policy) as connection, \
                         patch('waypoint_demo.observation',lambda *_:{}):
                        record(self.s,self.s.task.language,max_steps=3,**kwargs)
                    connection.assert_called_once_with(expected)
        finally:
            self.s.args.policy_port = original_port

    def test_actual_shift_and_target_reconstruction(self):
        name='alphabet_soup_1';old=self.s.poses()[name];p,r=self.s.gripper_pose()
        w=relative_waypoint('test',p,r,name,self.s.poses(),(p,r))
        node=dict(relative_position_mean=w.offset,relative_orientation_mean=w.orientation)
        info=shift_object(self.s,name,[.01,0,0]);new=self.s.poses()[name]
        self.assertAlmostEqual(info['position'][0]-old[0][0],.01,delta=.002)
        target,orientation=reconstruct_target(node,new)
        np.testing.assert_allclose(target,new[0]+new[1]@np.asarray(w.offset),atol=1e-10)
        self.assertGreater(np.linalg.norm(target-p),.005)
        self.assertLess(Rotation.from_matrix(orientation.T@r).magnitude(),.02)
        with self.assertRaisesRegex(ValueError,'horizontal'):shift_object(self.s,name,[0,0,.1])

    def test_shuffle_keeps_goal_site_parent_and_is_repeatable(self):
        goals = self.s.world.domain.parsed_problem['goal_state']
        destinations = {self.s.world.parent_body(g[2]) for g in goals
                        if len(g) >= 3 and g[0].lower() in ('in', 'on')}
        self.assertIn('basket_1', destinations)
        before = self.s.poses()
        mapping = shuffle_objects(self.s, seed=7)
        self.assertTrue(mapping)
        self.assertFalse(destinations.intersection(mapping))
        for name, donor in mapping.items():
            self.assertNotEqual(name, donor)
            np.testing.assert_allclose(self.s.poses()[name][0][:2], before[donor][0][:2], atol=.003)
        for name in destinations:
            np.testing.assert_allclose(self.s.poses()[name][0], before[name][0], atol=.003)
        self.s.env.set_init_state(self.initial)
        self.s.attach(settle=False)
        self.assertEqual(shuffle_objects(self.s, seed=7), mapping)

    def test_repeated_shuffle_and_exact_first_setup_restoration(self):
        original = self.s.training_state.copy()
        previous = None
        for _ in range(4):
            mapping = self.s.shuffle_scene(seed=7)
            self.assertNotEqual(mapping, previous)
            self.assertFalse(np.allclose(self.s.env.sim.get_state().flatten(), original))
            previous = mapping
        self.s.restore_training_scene()
        np.testing.assert_allclose(self.s.env.sim.get_state().flatten(), original, atol=1e-12)
        self.assertEqual(self.s.executor.gripper, self.s.training_gripper)
        self.assertNotEqual(self.s.shuffle_scene(seed=7), previous)
        np.testing.assert_array_equal(self.s.training_state, original)

    def test_failed_shuffle_keeps_current_scene_and_previous_mapping(self):
        previous = self.s.shuffle_scene(seed=7)
        before = self.s.env.sim.get_state().flatten().copy()
        with patch('waypoint_replay.shuffle_objects', side_effect=ValueError('No different layout')):
            with self.assertRaisesRegex(ValueError, 'No different'):
                self.s.shuffle_scene()
        np.testing.assert_allclose(self.s.env.sim.get_state().flatten(), before, atol=1e-12)
        self.assertEqual(self.s.last_shuffle_mapping, previous)

    def test_replay_original_and_shifted_targets(self):
        name='alphabet_soup_1';p,r=self.s.gripper_pose();p=p+[0,0,.008]
        w=relative_waypoint('test',p,r,name,self.s.poses(),self.s.gripper_pose())
        node=dict(segment=0,index=0,phase=0.,reference=dict(name=name,role='target',policy='episode_start'),
                  relative_position_mean=w.offset,relative_orientation_mean=w.orientation,
                  position_tolerance=.004,orientation_tolerance=.03,semantic_event=None,gripper_command='keep')
        c=Config();cov=np.eye(10)*1e-5
        model=dict(config=dataclasses.asdict(c),waypoints=[node],segments=[dict(reference=node['reference'],position=[w.offset],orientation=[w.orientation],
                   covariance=[cov.tolist()],phase=[0.],gripper_probability=[0.],start_event=dict(key='start',kind='start'),end_event=dict(key='end',kind='end'))])
        for mode in ('waypoints','trajectory'):
            for moved in (False,True):
                self.s.env.set_init_state(self.initial);self.s.attach(settle=False)
                if moved:shift_object(self.s,name,[.01,0,0])
                result=replay(self.s,model,mode)
                self.assertTrue(result['completed'],result)
                self.assertFalse(result['task_success'])  # Target tracking is not whole-task success.


    def test_carried_object_travels_with_the_wrist_when_checking_collisions(self):
        """A free-joint object does not follow arm qpos on its own."""
        from waypoint_rrt import Scene
        name = 'alphabet_soup_1'
        scene = Scene(self.s.env, carried_bodies=(name,))
        self.assertTrue(scene.carried, 'expected a free joint for '+name)
        def offset():
            position, rotation = self.s.gripper_pose()
            return rotation.T@(self.s.world.pos(name)-position)
        before, start = offset(), scene.arm_qpos()
        elsewhere = start+np.array([.2, -.15, .1, .1, -.1, .1, .0])
        scene._set_arm(elsewhere)
        np.testing.assert_allclose(offset(), before, atol=1e-9)
        scene._set_arm(start)
        np.testing.assert_allclose(offset(), before, atol=1e-9)
        # A plain scene leaves the object behind, so its own sweep is unchecked.
        plain = Scene(self.s.env, ignore_bodies=(name,))
        self.assertEqual(plain.carried, [])
        held = self.s.world.pos(name).copy()
        plain._set_arm(elsewhere)
        np.testing.assert_allclose(self.s.world.pos(name), held, atol=1e-9)
        plain._set_arm(start)

    def test_collision_check_restores_the_scene_it_sampled(self):
        from waypoint_rrt import Scene
        name = 'alphabet_soup_1'
        scene = Scene(self.s.env, carried_bodies=(name,))
        before = self.s.env.sim.get_state().flatten().copy()
        scene.in_collision(scene.arm_qpos()+np.array([.3, .2, -.2, .1, .1, .1, .1]))
        np.testing.assert_allclose(self.s.env.sim.get_state().flatten(), before, atol=1e-9)

    def test_straight_carry_is_checked_before_it_is_driven(self):
        """A clear line, a blocked one and an unfollowable one are distinct."""
        from waypoint_skill import probe_line, planning_scene
        position, rotation = self.s.gripper_pose()
        scene = planning_scene(self.s)
        status, q = probe_line(self.s, scene, position+[0, 0, .05], rotation)
        self.assertEqual(status, 'clear')
        self.assertTrue(scene.valid(q))
        # Straight down through the table is not a free-space move; whether the
        # arm bottoms out or simply cannot follow, it is never reported clear.
        status, _ = probe_line(self.s, scene, position-[0, 0, .45], rotation)
        self.assertIn(status, ('blocked', 'unreachable'))


    def test_only_obstacles_under_the_path_have_to_be_cleared(self):
        """Climbing over a cabinet the arm never crosses wastes its reach."""
        from waypoint_skill import path_obstacle_top
        world = self.s.world
        names = list(world.domain.objects_dict)
        tallest = max(names, key=lambda n: world.aabb(n)[1][2])
        here = world.pos(tallest)
        over = path_obstacle_top(world, here+[0, 0, .3], here+[0, 0, .1])
        self.assertAlmostEqual(over, world.aabb(tallest)[1][2], places=6)
        # A path that never passes over anything clears nothing.
        away = np.array([here[0]+2., here[1]+2., here[2]])
        self.assertIsNone(path_obstacle_top(world, away, away+[0, .1, 0]))
        # The object being picked up is not its own obstacle.
        self.assertNotEqual(path_obstacle_top(world, here+[0, 0, .3], here,
                                              ignore=(tallest,)), over)


    def test_planner_identifies_its_robot_from_the_model(self):
        """No joint names, joint count or name prefixes are assumed."""
        from waypoint_rrt import Scene
        scene = Scene(self.s.env, args=self.s.executor.args)
        robot = self.s.env.env.robots[0]
        self.assertEqual(scene.joints, list(robot.robot_joints))
        self.assertEqual(len(scene.qadr), len(scene.joints))
        self.assertEqual(scene.limits.shape, (len(scene.joints), 2))
        own = set(robot.robot_model.bodies) | set(robot.gripper.bodies)
        model = self.s.env.sim.model
        for geom in scene.robot_geoms:
            self.assertIn(model.body_id2name(model.geom_bodyid[geom]), own)
        self.assertTrue(scene.robot_geoms)

    def test_edge_checks_bound_end_effector_travel_not_just_joint_travel(self):
        """A joint-space spacing alone lets the wrist step over a thin object."""
        import dataclasses
        from waypoint_rrt import Scene, RRTConnect
        scene = Scene(self.s.env, args=self.s.executor.args)
        start = scene.arm_qpos()
        # Several sweeps, so the check does not hinge on one lucky direction.
        rng = np.random.default_rng(1)
        goals = [np.clip(start+rng.normal(scale=.35, size=len(start)),
                         scene.limits[:, 0], scene.limits[:, 1]) for _ in range(12)]

        def worst_gap(cartesian):
            args = dataclasses.replace(self.s.executor.args,
                                       collision_check_cartesian=cartesian)
            planner = RRTConnect(scene, args, np.random.default_rng(0))
            gaps, samples = [], 0
            for goal in goals:
                seen, real = [], scene.valid
                scene.valid = lambda q: (seen.append(np.array(q)), True)[1]
                try:
                    planner._edge_clear(start, goal)
                finally:
                    scene.valid = real
                samples += len(seen)
                saved, points = scene.arm_qpos(), []
                for q in [start]+seen:
                    scene._set_arm(q)
                    points.append(scene.eef_pos().copy())
                scene._set_arm(saved)
                gaps.append(float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1))))
            return max(gaps), samples

        loose, few = worst_gap(1e9)      # joint spacing only
        tight, many = worst_gap(.0075)   # with the Cartesian bound
        self.assertGreater(loose, .018)  # steps over an 18 mm object
        self.assertLess(tight, .018)     # cannot
        self.assertGreater(many, few)    # and pays for it in samples


if __name__=='__main__':unittest.main()
