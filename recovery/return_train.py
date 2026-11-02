"""Train and evaluate a task-shared, offline return-point contextual bandit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from recovery.return_model import (ReturnPointNetwork, SelectionConfig, batch_decisions,
                                   support_features, training_loss)
from recovery.return_points import SCHEMA, Reward, load_transitions


def split_tasks(rows, validation_tasks=None, fraction=.2, seed=0):
    tasks = sorted({d.task_key for d, _, _ in rows})
    if len(tasks) < 2:
        raise ValueError('Need at least two tasks for a task-held-out validation split')
    if validation_tasks:
        held_out = set(validation_tasks)
        if not held_out <= set(tasks):
            raise ValueError('Validation task keys must occur in the dataset')
    else:
        if not 0 < fraction < 1:
            raise ValueError('Validation fraction must be between 0 and 1')
        rng = np.random.default_rng(seed)
        held_out = set(rng.permutation(tasks)[:max(1, min(len(tasks) - 1, int(np.ceil(len(tasks) * fraction))))])
    train = [r for r in rows if r[0].task_key not in held_out]
    validation = [r for r in rows if r[0].task_key in held_out]
    if not train or not validation:
        raise ValueError('Training and held-out tasks must both be nonempty')
    return train, validation


def evaluate(model, rows, reward, batch_size=64):
    model.eval()
    errors = dict(safe_brier=[], resume_brier=[], recovery_mae_seconds=[],
                  finish_mae_seconds=[], total_mae_seconds=[], q_mae=[])
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            predicted = model(batch_decisions([d for d, _, _ in chunk]))
            for index, (_, action, outcome) in enumerate(chunk):
                safe = float(predicted['safe_logit'][index, action].sigmoid())
                errors['safe_brier'].append((safe - outcome.safe_reached) ** 2)
                errors['q_mae'].append(abs(float(predicted['q'][index, action]) - reward(outcome)))
                if outcome.resume_success is not None:
                    resume = float(predicted['resume_logit'][index, action].sigmoid())
                    errors['resume_brier'].append((resume - outcome.resume_success) ** 2)
                if outcome.safe_reached:
                    estimate = float(predicted['recovery_log_time'][index, action].clamp_max(12).expm1())
                    errors['recovery_mae_seconds'].append(abs(estimate - outcome.recovery_seconds))
                if outcome.resume_success:
                    estimate = float(predicted['finish_log_time'][index, action].clamp_max(12).expm1())
                    errors['finish_mae_seconds'].append(abs(estimate - outcome.finish_seconds))
                if outcome.success:
                    estimate = float(predicted['total_log_time'][index, action].clamp_max(12).expm1())
                    errors['total_mae_seconds'].append(abs(estimate - outcome.recovery_seconds - outcome.rollout_seconds))
    result = {key: dict(mean=float(np.mean(values)) if values else None, count=len(values))
              for key, values in errors.items()}
    result.update(rollouts=len(rows), tasks=sorted({d.task_key for d, _, _ in rows}),
                  observed_safe_arrival_rate=float(np.mean([o.safe_reached for _, _, o in rows])),
                  observed_joint_success_rate=float(np.mean([o.success for _, _, o in rows])),
                  observed_force_compliant_success_rate=float(np.mean([o.force_compliant_success for _, _, o in rows])))
    # Ranking quality is observable only with a measured outcome for every action.
    groups = {}
    for decision, action, outcome in rows:
        group = groups.setdefault(decision.id, dict(decision=decision, returns={}))
        group['returns'].setdefault(action, []).append(reward(outcome))
    regrets = []
    with torch.no_grad():
        for group in groups.values():
            decision, returns = group['decision'], group['returns']
            if len(returns) != len(decision.candidates):
                continue
            chosen = int(model(batch_decisions([decision]))['q'][0].argmax())
            means = {action: float(np.mean(values)) for action, values in returns.items()}
            regrets.append(max(means.values()) - means[chosen])
    result['unconstrained_ranking_regret'] = dict(
        mean=float(np.mean(regrets)) if regrets else None, complete_decisions=len(regrets))
    return result


def train(rows, output, epochs=50, batch_size=64, hidden=128, learning_rate=3e-4,
          cql_alpha=.1, seed=0, validation_tasks=None, reward=None, selection=None, threads=2):
    if epochs < 1 or batch_size < 1 or hidden < 1 or learning_rate <= 0 or cql_alpha < 0:
        raise ValueError('Invalid training configuration')
    identities = {(d.vla_id, d.controller_id, d.max_policy_steps, d.stable_seconds,
                   d.resume_contact_mode) for d, _, _ in rows}
    if len(identities) != 1:
        raise ValueError('Train separately for each VLA, recovery controller, and rollout horizon/stability contract')
    # Counterfactual copies must describe exactly the same failure decision.
    decisions = {}
    for decision, _, _ in rows:
        serialized = json.dumps(asdict(decision), sort_keys=True)
        previous = decisions.setdefault(decision.id, serialized)
        if previous != serialized:
            raise ValueError('A decision ID describes different failure states/candidates')
    training, validation = split_tasks(rows, validation_tasks, seed=seed)
    if not any(o.safe_reached for _, _, o in training) or not any(o.success for _, _, o in training):
        raise ValueError('Need safely reached and completed training rollouts to learn the time heads')
    reward, selection = reward or Reward(), selection or SelectionConfig()
    if threads < 1:
        raise ValueError('CPU thread count must be positive')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ReturnPointNetwork(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best_score = [], float('inf')
    vla_id, controller_id, max_policy_steps, stable_seconds, resume_contact_mode = next(iter(identities))
    support = []
    with torch.no_grad():
        for start in range(0, len(training), batch_size):
            chunk = training[start:start + batch_size]
            features = support_features(batch_decisions([d for d, _, _ in chunk]))
            support.extend(features[i, action] for i, (_, action, _) in enumerate(chunk))
    support = torch.stack(support)
    mean = support.mean(0)
    scale = support.std(0, unbiased=False).clamp_min(.05)
    normalized_support = (support - mean) / scale
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(len(training))
        for start in range(0, len(order), batch_size):
            chunk = [training[i] for i in order[start:start + batch_size]]
            batch = batch_decisions([d for d, _, _ in chunk])
            actions = torch.tensor([action for _, action, _ in chunk], dtype=torch.long)
            loss, terms = training_loss(model(batch), batch, actions,
                [o for _, _, o in chunk], reward, cql_alpha)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; checkpoint was not saved')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation, reward, batch_size)
        score = metrics['q_mae']['mean'] + metrics['safe_brier']['mean']
        record = dict(epoch=epoch + 1, training_loss=float(np.mean(losses)), validation=metrics)
        history.append(record)
        if score < best_score:
            best_score = score
            checkpoint = dict(schema=SCHEMA, trained=True, hidden=hidden, model=model.state_dict(),
                optimizer=optimizer.state_dict(), epoch=epoch + 1, vla_id=vla_id,
                controller_id=controller_id, reward=asdict(reward), selection=asdict(selection),
                max_policy_steps=max_policy_steps, stable_seconds=stable_seconds,
                resume_contact_mode=resume_contact_mode,
                support=normalized_support, support_mean=mean, support_scale=scale,
                training_tasks=sorted({d.task_key for d, _, _ in training}),
                validation_tasks=metrics['tasks'], validation=metrics,
                training_config=dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                     cql_alpha=cql_alpha, seed=seed))
            torch.save(checkpoint, output / 'checkpoint.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"epoch {epoch + 1}/{epochs}: loss={record['training_loss']:.4f}, "
                  f"held-out Q MAE={metrics['q_mae']['mean']:.4f}", flush=True)
    saved = torch.load(output / 'checkpoint.pt', weights_only=True)
    (output / 'evaluation.json').write_text(json.dumps(saved['validation'], indent=2) + '\n')
    return output / 'checkpoint.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Measured return-point transition JSONL')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--cql-alpha', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--validation-tasks', nargs='+', help='e.g. libero_goal:1 libero_object:3')
    parser.add_argument('--success-bonus', type=float, default=10.)
    parser.add_argument('--failure-cost', type=float, default=2.)
    parser.add_argument('--violation-cost', type=float, default=10.)
    parser.add_argument('--seconds-cost', type=float, default=.1)
    args = parser.parse_args()
    options = vars(args).copy()
    data = options.pop('data')
    reward = Reward(**{key: options.pop(key) for key in
                       ('success_bonus', 'failure_cost', 'violation_cost', 'seconds_cost')})
    path = train(load_transitions(data), reward=reward, **options)
    print(f'Saved {path}; see evaluation.json for task-held-out prediction metrics')


if __name__ == '__main__':
    main()
