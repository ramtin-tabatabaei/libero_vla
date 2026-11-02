"""Physical return and measured fault history on the existing LIBERO adapter."""
from dataclasses import asdict

import numpy as np


def test_physical_return_reaches_an_actual_visit_without_rewinding_time():
    import subgoal_planner as sp
    from waypoint_gmm_editor import GMMSession
    from recovery.scene import enable_robot_collisions
    from recovery.return_points import ReturnPoint
    from recovery.return_sim import PhysicalReturnController, snapshot

    session = GMMSession(sp.Args())
    try:
        session.load('libero_goal', 1, cameras=False)
        enable_robot_collisions(session.world)
        historical = snapshot(session)
        session.executor.hold(3)
        before = float(session.env.sim.data.time)
        candidate = ReturnPoint('past', 0, 'approach', .1, 'completed_duration', historical)
        outcome = PhysicalReturnController(session).run(candidate)
        assert outcome.safe_reached, outcome.detail
        assert session.env.sim.data.time > before > historical.time
        assert np.linalg.norm(np.asarray(snapshot(session).position) - historical.position) < .03
    finally:
        if session.env:
            session.env.close()


def test_measured_fault_produces_only_pre_failure_return_candidates(tmp_path):
    import subgoal_planner as sp
    from waypoint_gmm import GMMChain
    from waypoint_gmm_editor import GMMSession
    from waypoint_recovery_images import plan_path
    from waypoint_recovery_benchmark import prepare_scene, run_fault
    from recovery.scene import enable_robot_collisions
    from recovery.return_sim import RecordVisits, snapshot
    from recovery.return_points import Decision

    session = GMMSession(sp.Args(seed=1000))
    spec = dict(suite='libero_goal', task_id=1, scene='original', kind='transition',
                intensity='low', seed=1000, fault_seed=1)
    try:
        session.load(spec['suite'], spec['task_id'], cameras=False)
        chain = GMMChain.load(plan_path(spec['suite'], spec['task_id']), spec['suite'], spec['task_id'])
        prepare_scene(session, chain, spec, tmp_path / 'layouts')
        enable_robot_collisions(session.world)
        with RecordVisits(session) as recorder:
            fault = run_fault(session, chain, spec, 0., 180., tmp_path, capture=lambda *_: None)
        assert fault['observed_failure'], fault
        candidates = recorder.history.candidates(fault['failure_event']['sim_time'], [0., .1, 1.])
        assert candidates
        d = Decision('measured-fault', 'libero_goal:1', 'vla@fixed', 'controller@fixed',
                     fault['failure_event']['sim_time'], snapshot(session), candidates)
        d.validate()
        assert all(c.state.time < d.failure_time for c in candidates)
        assert all(c.state is recorder.history.segments[int(c.id.split(':')[0])]['states'][int(c.id.split(':')[1])]
                   for c in candidates)
    finally:
        if session.env:
            session.env.close()
