"""Return-point causality, conditional labels, shared policy, and safe handoff."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from recovery.return_points import (Decision, Outcome, ReturnPoint, RobotState,
    Reward, SCENE_DIM, VisitedHistory, load_transitions, write_transition)
from recovery.return_model import (ReturnPointNetwork, ReturnPointSelector, SelectionConfig,
    batch_decisions, training_loss)
from recovery.return_train import split_tasks, train


torch.set_num_threads(2)


def state(t, x=0., held=()):
    return RobotState(t, [x, 0., .5], np.eye(3).tolist(), [x] + [0.] * 6,
                      [0.] * 7, [.04, -.04], -1.,
                      scene=[[x] + [0.] * (SCENE_DIM - 1)], held=list(held))


def decision(task='task:a', id='failure:a'):
    points = [ReturnPoint('a', 0, 'approach', .1, 'completed_duration', state(1., .1)),
              ReturnPoint('b', 1, 'grasp', .5, 'visited_prefix_duration', state(2., .2))]
    return Decision(id, task, 'vla@fixed', 'rrt@fixed', 3., state(4., .3), points)


def test_return_points_are_actual_visits_and_exclude_the_failure_boundary():
    history = VisitedHistory()
    samples = [state(float(i), i / 10.) for i in range(11)]
    history.start(2, 'approach', samples[0])
    for sample in samples[1:]:
        history.append(sample)
    history.complete(samples[-1])
    points = history.candidates(12., [0., .1, 1.])
    assert [p.state for p in points] == [samples[0], samples[1], samples[10]]
    assert all(p.phase_basis == 'completed_duration' for p in points)
    prefix = history.candidates(5., [.1, 1.])
    assert all(p.state.time < 5. for p in prefix)
    assert prefix[-1].state is samples[4]
    assert all(p.phase_basis == 'visited_prefix_duration' for p in prefix)


def test_future_or_post_failure_candidate_is_rejected():
    d = decision()
    d.candidates[0].state.time = d.failure_time
    with pytest.raises(ValueError, match='BEFORE'):
        d.validate()


def test_failed_return_and_vla_timeout_are_distinct_and_censored(tmp_path):
    d = decision()
    path = tmp_path / 'transitions.jsonl'
    failed_return = Outcome(False, True, 2.)
    timeout = Outcome(True, False, 1., False, None, 20.)
    write_transition(path, d, 'a', failed_return)
    write_transition(path, d, 'b', timeout)
    rows = load_transitions(path)
    assert rows[0][2].resume_success is None
    assert rows[1][2].resume_success is False
    assert rows[1][2].finish_seconds is None
    assert Reward()(failed_return) < Reward()(Outcome(False, False, 2.))


def test_shared_model_ignores_task_and_primitive_names_and_preserves_permutations():
    torch.manual_seed(2)
    model = ReturnPointNetwork(32).eval()
    original = decision()
    renamed = Decision.from_dict(asdict(original))
    renamed.task_key = 'previously-unseen-task'
    renamed.candidates.reverse()
    for point in renamed.candidates:
        point.primitive = 'new_primitive_name'
        point.waypoint += 100
    expected = model(batch_decisions([original]))
    actual = model(batch_decisions([renamed]))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name].flip(1))


def test_scene_padding_is_ignored_and_candidate_batching_is_invariant():
    model = ReturnPointNetwork(32).eval()
    original, bigger = decision(), decision('task:b', 'failure:b')
    bigger.current.scene.extend([[1.] * SCENE_DIM] * 5)
    bigger.candidates.append(replace(bigger.candidates[0], id='c'))
    standalone = model(batch_decisions([original]))
    combined = model(batch_decisions([original, bigger]))
    for name in standalone:
        torch.testing.assert_close(standalone[name][0], combined[name][0, :2])


def test_missing_vla_outcome_does_not_train_resume_or_duration_heads():
    model = ReturnPointNetwork(32)
    batch = batch_decisions([decision()])
    predicted = model(batch)
    for value in predicted.values():
        value.retain_grad()
    loss, _ = training_loss(predicted, batch, torch.tensor([0]), [Outcome(False, True, 2.)], Reward())
    loss.backward()
    for name in ('resume_logit', 'recovery_log_time', 'finish_log_time', 'total_log_time'):
        assert predicted[name].grad is None or torch.count_nonzero(predicted[name].grad) == 0
    assert torch.count_nonzero(predicted['safe_logit'].grad) == 1


def test_task_split_prevents_scene_or_counterfactual_leakage():
    rows = [(decision(f'task:{t}', f'failure:{t}:{scene}'), 0, Outcome(False, False, 1.))
            for t in range(3) for scene in range(2)]
    training, validation = split_tasks(rows, ['task:1'])
    assert {d.task_key for d, _, _ in training} == {'task:0', 'task:2'}
    assert {d.task_key for d, _, _ in validation} == {'task:1'}
    with pytest.raises(ValueError, match='at least two tasks'):
        split_tasks(rows[:2])


def test_training_checkpoint_ranks_candidates_and_refuses_unsupported_context(tmp_path):
    rows = []
    for task in range(3):
        for repeat in range(5):
            d = decision(f'task:{task}', f'failure:{task}:{repeat}')
            rows.extend([(d, 0, Outcome(True, False, 1., True, 2., 2.)),
                         (d, 1, Outcome(False, True, 3.))])
    checkpoint = train(rows, tmp_path, epochs=12, hidden=32, learning_rate=.01,
                       validation_tasks=['task:2'])
    selector = ReturnPointSelector(checkpoint, SelectionConfig(0., 0., 1, 2.))
    report = selector.select(decision())
    assert report['selected'] == 'a'
    assert report['rankings'][0]['total_seconds_given_success'] >= 0
    assert json.loads((tmp_path / 'evaluation.json').read_text())['tasks'] == ['task:2']
    unseen = decision()
    unseen.current.velocity = [1000.] * 7
    assert selector.select(unseen)['selected'] is None
    assert selector.select(decision(), feasible=lambda _: False)['selected'] is None
    wrong_vla = replace(decision(), vla_id='different_vla')
    with pytest.raises(ValueError, match='identity'):
        selector.select(wrong_vla)


def test_unsafe_return_never_resumes_vla():
    from recovery.return_sim import resume_from_return
    controller = SimpleNamespace(run=lambda _: Outcome(True, True, 1.))
    runner = Mock()
    outcome = resume_from_return(SimpleNamespace(), decision().candidates[0], controller,
                                 'grasp the object', policy_runner=runner)
    assert not outcome.safe_reached
    runner.assert_not_called()


def test_candidate_cannot_request_a_missing_historical_grasp(tmp_path):
    rows = [(decision(f'task:{task}', f'failure:{task}'), 0,
             Outcome(True, False, 1., True, 2., 2.)) for task in range(2)]
    checkpoint = train(rows, tmp_path, epochs=1, hidden=16, validation_tasks=['task:1'])
    selector = ReturnPointSelector(checkpoint, SelectionConfig(0., 0., 1, 100.))
    d = decision()
    for candidate in d.candidates:
        candidate.state.held = ['object_previously_held']
    assert selector.select(d)['selected'] is None


def test_contact_observation_records_exceedance_without_changing_recovery_gate(monkeypatch):
    from recovery.return_sim import ReturnSafety, ReturnViolation, watch_contacts
    import recovery.scene as scene
    import recovery.return_sim as adapter
    sim = SimpleNamespace(step=Mock(), model=SimpleNamespace())
    world = SimpleNamespace(m=object(), d=SimpleNamespace(ncon=0),
        domain=SimpleNamespace(parsed_problem={}))
    grip = SimpleNamespace(important_geoms=dict(left_fingerpad=[], right_fingerpad=[]))
    domain = SimpleNamespace(robots=[SimpleNamespace(gripper=grip)], objects_dict={})
    session = SimpleNamespace(world=world, env=SimpleNamespace(sim=sim, env=domain))
    monitor = SimpleNamespace(measure=lambda: (np.array([11.]), False, 5.), robot=set())
    monkeypatch.setattr(scene, 'ContactMonitor', lambda *args: monitor)
    monkeypatch.setattr(scene, 'geom_ids', lambda *args: set())
    monkeypatch.setattr(adapter, 'snapshot', lambda *args: SimpleNamespace(held=[]))
    with watch_contacts(session, ReturnSafety(), stop=False) as recorded:
        sim.step()
        sim.step()
    assert recorded['limit_exceeded']
    assert recorded['exceeding_physics_steps'] == 2
    assert recorded['peak_bad_force_n'] == 11.
    with pytest.raises(ReturnViolation):
        with watch_contacts(session, ReturnSafety()):
            sim.step()


def test_goal_success_and_force_compliant_success_are_reported_separately():
    outcome = Outcome(True, False, 1., True, 2., 2.,
                      resume_contacts=dict(limit_exceeded=True))
    assert outcome.success
    assert not outcome.force_compliant_success


def test_physical_evaluation_counts_abstentions_and_baseline_successes():
    from recovery.return_evaluate import summarize
    records = [dict(controller='rl', selected=False, success=False,
                    force_compliant_success=False, return_attempted=False, total_seconds=0.),
               dict(controller='resume_now', selected=True, success=True,
                    force_compliant_success=False, return_attempted=False, total_seconds=3.,
                    vla_force_exceeded=True)]
    result = summarize(records)
    assert result['rl']['coverage'] == 0
    assert result['rl']['task_success_rate'] == 0
    assert result['resume_now']['task_success_rate'] == 1
    assert result['resume_now']['force_compliant_successes'] == 0
