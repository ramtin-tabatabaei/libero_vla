"""Shared set encoder and conservative one-decision offline RL selector."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recovery.return_points import (CORE_DIM, OPTION_DIM, SCENE_DIM, SCHEMA,
                                    core, option_features)


def batch_decisions(decisions, device='cpu'):
    batch = len(decisions)
    options = max(len(d.candidates) for d in decisions)
    slots = max(1, max(len(s.scene) for d in decisions
                       for s in [d.current] + [c.state for c in d.candidates]))
    current = np.zeros((batch, CORE_DIM), np.float32)
    scene = np.zeros((batch, slots, SCENE_DIM), np.float32)
    scene_mask = np.zeros((batch, slots), bool)
    historical = np.zeros((batch, options, slots, SCENE_DIM), np.float32)
    historical_mask = np.zeros((batch, options, slots), bool)
    features = np.zeros((batch, options, OPTION_DIM), np.float32)
    mask = np.zeros((batch, options), bool)
    for row, decision in enumerate(decisions):
        decision.validate()
        current[row] = core(decision.current)
        count = len(decision.current.scene)
        scene[row, :count] = np.asarray(decision.current.scene).reshape(count, SCENE_DIM)
        scene_mask[row, :count] = True
        for column, candidate in enumerate(decision.candidates):
            features[row, column] = option_features(decision.current, candidate)
            count = len(candidate.state.scene)
            historical[row, column, :count] = np.asarray(candidate.state.scene).reshape(count, SCENE_DIM)
            historical_mask[row, column, :count] = True
            mask[row, column] = True
    arrays = dict(current=current, scene=scene, scene_mask=scene_mask,
                  historical=historical, historical_mask=historical_mask,
                  features=features, mask=mask)
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def pool(values, mask):
    """Masked mean and max; an absent set yields zeros, including the max."""
    expanded = mask.unsqueeze(-1)
    mean = (values * expanded).sum(-2) / expanded.sum(-2).clamp_min(1)
    maximum = values.masked_fill(~expanded, -torch.inf).amax(-2)
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat([mean, maximum], -1)


def support_features(batch):
    scene = pool(batch['scene'], batch['scene_mask'])
    past = pool(batch['historical'], batch['historical_mask'])
    count = batch['features'].shape[1]
    return torch.cat([batch['current'][:, None].expand(-1, count, -1),
                      scene[:, None].expand(-1, count, -1),
                      batch['features'], past], -1)


class ReturnPointNetwork(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.scene_encoder = nn.Sequential(nn.Linear(SCENE_DIM, 32), nn.SiLU(),
                                            nn.Linear(32, 32), nn.SiLU())
        self.pair = nn.Sequential(nn.Linear(CORE_DIM + OPTION_DIM + 128, hidden),
                                  nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        # Q, safe arrival, conditional VLA success, recovery/finish/total ETAs.
        self.heads = nn.Linear(hidden, 6)

    def forward(self, batch):
        current = pool(self.scene_encoder(batch['scene']), batch['scene_mask'])
        history = pool(self.scene_encoder(batch['historical']), batch['historical_mask'])
        count = batch['features'].shape[1]
        pair = torch.cat([batch['current'][:, None].expand(-1, count, -1),
                          current[:, None].expand(-1, count, -1),
                          batch['features'], history], -1)
        output = self.heads(self.pair(pair))
        return dict(q=output[..., 0], safe_logit=output[..., 1],
                    resume_logit=output[..., 2],
                    recovery_log_time=F.softplus(output[..., 3]),
                    finish_log_time=F.softplus(output[..., 4]),
                    total_log_time=F.softplus(output[..., 5]))


def training_loss(predicted, batch, actions, outcomes, reward, cql_alpha=.1):
    """Horizon-one Q regression, CQL, and observable auxiliary targets.

    Missing VLA outcomes and right-censored ETAs are masked, never set to zero.
    There is no next-state bootstrap after this terminal recovery/VLA episode.
    """
    indices = torch.arange(len(actions), device=actions.device)
    selected = {key: value[indices, actions] for key, value in predicted.items()}
    dtype, device = selected['q'].dtype, actions.device
    tensor = lambda values: torch.tensor(values, dtype=dtype, device=device)
    returns = tensor([reward(o) for o in outcomes])
    arrival = tensor([o.safe_reached for o in outcomes])
    resumed = torch.tensor([o.resume_success is not None for o in outcomes], device=device)
    completed = torch.tensor([bool(o.resume_success) for o in outcomes], device=device)
    successful = torch.tensor([o.success for o in outcomes], device=device)
    q_loss = F.smooth_l1_loss(selected['q'], returns)
    # Count normalization removes a candidate-count-dependent additive constant.
    logits = predicted['q'].masked_fill(~batch['mask'], -torch.inf)
    cql = (torch.logsumexp(logits, -1) - batch['mask'].sum(-1).float().log()
           - selected['q']).mean()
    safe_loss = F.binary_cross_entropy_with_logits(selected['safe_logit'], arrival)
    zero = selected['q'].sum() * 0
    resume_loss = (F.binary_cross_entropy_with_logits(selected['resume_logit'][resumed],
        tensor([bool(o.resume_success) for o in outcomes])[resumed]) if resumed.any() else zero)
    recovery_loss = (F.smooth_l1_loss(selected['recovery_log_time'][arrival.bool()],
        tensor([np.log1p(o.recovery_seconds) for o in outcomes])[arrival.bool()])
        if arrival.bool().any() else zero)
    finish_loss = (F.smooth_l1_loss(selected['finish_log_time'][completed],
        tensor([np.log1p(o.finish_seconds or 0.) for o in outcomes])[completed])
        if completed.any() else zero)
    total_loss = (F.smooth_l1_loss(selected['total_log_time'][successful],
        tensor([np.log1p(o.recovery_seconds + o.rollout_seconds) for o in outcomes])[successful])
        if successful.any() else zero)
    loss = q_loss + cql_alpha * cql + safe_loss + resume_loss + recovery_loss + finish_loss + total_loss
    return loss, dict(q=float(q_loss.detach()), cql=float(cql.detach()),
                      safe=float(safe_loss.detach()), resume=float(resume_loss.detach()),
                      recovery_time=float(recovery_loss.detach()), finish_time=float(finish_loss.detach()),
                      total_time=float(total_loss.detach()))


@dataclass
class SelectionConfig:
    min_safe_probability: float = .9
    min_resume_probability: float = .5
    min_support: int = 5
    support_radius: float = 2.

    def __post_init__(self):
        for probability in (self.min_safe_probability, self.min_resume_probability):
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Probability thresholds must lie between 0 and 1')
        if self.min_support < 1 or not np.isfinite(self.support_radius) or self.support_radius <= 0:
            raise ValueError('Support count and radius must be positive')


class ReturnPointSelector:
    def __init__(self, checkpoint, config=None, device='cpu'):
        self.device = device
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get('schema') != SCHEMA or not saved.get('trained'):
            raise ValueError('Need a trained visited-return-point checkpoint')
        self.model = ReturnPointNetwork(saved['hidden']).to(device)
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        self.vla_id, self.controller_id = saved['vla_id'], saved['controller_id']
        self.max_policy_steps, self.stable_seconds = saved['max_policy_steps'], saved['stable_seconds']
        self.resume_contact_mode = saved.get('resume_contact_mode', 'stop')
        self.config = config or SelectionConfig(**saved['selection'])
        self.support = saved['support'].to(device)
        self.mean = saved['support_mean'].to(device)
        self.scale = saved['support_scale'].to(device)

    def select(self, decision, feasible=None):
        decision.validate()
        if (decision.vla_id, decision.controller_id) != (self.vla_id, self.controller_id):
            raise ValueError('VLA/controller identity differs from the training rollouts')
        if (decision.max_policy_steps, decision.stable_seconds) != (self.max_policy_steps, self.stable_seconds):
            raise ValueError('VLA rollout horizon/stability check differs from training')
        if decision.resume_contact_mode != self.resume_contact_mode:
            raise ValueError('VLA contact-monitor contract differs from training')
        batch = batch_decisions([decision], self.device)
        with torch.no_grad():
            predicted = self.model(batch)
            query = (support_features(batch)[0] - self.mean) / self.scale
            distances = torch.cdist(query, self.support) / np.sqrt(query.shape[-1])
            counts = (distances <= self.config.support_radius).sum(-1)
            safe = predicted['safe_logit'][0].sigmoid()
            resume = predicted['resume_logit'][0].sigmoid()
        rankings = []
        for index, candidate in enumerate(decision.candidates):
            reasons = []
            if counts[index] < self.config.min_support:
                reasons.append('insufficient similar training rollouts')
            if safe[index] < self.config.min_safe_probability:
                reasons.append('safe-arrival estimate below threshold')
            if resume[index] < self.config.min_resume_probability:
                reasons.append('VLA-success estimate below threshold')
            # Historical contact/grasp cannot be restored by moving the arm alone.
            if set(candidate.state.held) != set(decision.current.held):
                reasons.append('historical held-object state cannot be reproduced')
            recovery_time = float(torch.expm1(predicted['recovery_log_time'][0, index].clamp_max(12)))
            finish_time = float(torch.expm1(predicted['finish_log_time'][0, index].clamp_max(12)))
            rankings.append(dict(id=candidate.id, waypoint=candidate.waypoint,
                primitive=candidate.primitive, phase=candidate.phase, phase_basis=candidate.phase_basis,
                recorded_sim_time=candidate.state.time, position=candidate.state.position,
                expected_return=float(predicted['q'][0, index]),
                safe_arrival_probability=float(safe[index]),
                vla_success_given_arrival_probability=float(resume[index]),
                joint_success_probability=float(safe[index] * resume[index]),
                recovery_seconds_given_safe_arrival=recovery_time,
                finish_seconds_given_vla_success=finish_time,
                total_seconds_given_success=float(torch.expm1(predicted['total_log_time'][0, index].clamp_max(12))),
                nearby_training_rollouts=int(counts[index]), rejected=reasons))
        # Expensive physical checks run in value order until a feasible option wins.
        rankings.sort(key=lambda row: row['expected_return'], reverse=True)
        selected = None
        for row in rankings:
            if row['rejected']:
                continue
            candidate = next(c for c in decision.candidates if c.id == row['id'])
            if feasible is not None and not feasible(candidate):
                row['rejected'].append('current-scene planner rejected the return path')
                continue
            selected = row['id']
            break
        return dict(selected=selected,
                    reason='selected' if selected else 'no supported, acceptable return point',
                    rankings=rankings, thresholds=asdict(self.config))
