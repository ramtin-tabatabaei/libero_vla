"""The staged pick-up reward, and the scene measurements it is built from.

Kept apart from the environment so every term can be tested on plain numbers,
with no simulator in the way.

    r = P(d_prev) - P(d)                         integrated distance progress
      - lambda_R . sigmoid((d_align-d)/tau) . e_R smooth wrist alignment
      + 2 . 1[first valid contact]                once per episode
      + 5 . 1[first stable grasp]                 once per episode
      + 10 . 1[held-lift success]                  terminal task reward
      - lambda_c . sum_i min(F_i, F_max)          force on undesired contacts
      - lambda_s . 1[F_contact > F_safe]          severe collision
      - lambda_a . ||a||^2                        effort

Contact/grasp bonuses are latched by default. The final bonus is paid on held
success by the environment. Legacy per-step bonuses and hard alignment gating
remain explicit ablations for comparisons with earlier runs.
"""
import dataclasses

import numpy as np

GRASP, ARTICULATE, REACH = "grasp", "articulate", "reach"


@dataclasses.dataclass
class RewardConfig:
    # --- progress: w(d) = w_far + (w_near - w_far) exp(-d / w_reference) ---
    w_far: float = 10.0
    """Gain on distance closed when the target is still far off. The old flat
    weight was 10, and it took the arm from 38 cm to contact reliably."""
    w_near: float = 30.0
    """Gain as the distance goes to zero. Higher than w_far because the last
    two centimetres decide whether the fingers straddle the object or clip its
    shoulder, and a flat gain makes that stretch worth almost nothing."""
    w_reference: float = 0.10
    """Length scale over which the gain climbs from w_far to w_near."""

    # --- alignment ---
    align_distance: float = 0.10
    """Midpoint of the smooth orientation gate, in metres."""
    align_temperature: float = 0.01
    """tau in metres; controls the gate transition width, must be positive."""
    smooth_alignment: bool = True
    """Disable only for comparison with the former hard distance cutoff."""
    w_align: float = 0.5
    """lambda_R, per radian of geodesic error against the top-down grasp pose."""

    # --- staged bonuses ---
    bonus_contact: float = 2.0
    bonus_grasp: float = 5.0
    bonus_lift: float = 10.0
    latch_bonuses: bool = True
    """Pay each stage at most once per episode, even after release/regrasp."""
    terminal_stage_bonus: bool = True
    """Pay the final stage on held success; disable for the old height indicator."""
    integrated_progress: bool = True
    """Use the antiderivative difference; disable only for the old weighted-step ablation."""

    # --- collisions ---
    w_collision: float = 0.02
    """Initial lambda_c, per newton of capped undesired force. Calibrate using
    raw force/reward distributions and held-out behavior comparisons."""
    force_cap: float = 50.0
    """F_max. Caps any single contact so one bad spike cannot swamp the batch."""
    w_severe: float = 1.0
    """lambda_s, flat penalty for a contact above F_safe."""
    force_safe: float = 100.0
    """F_safe, in newtons."""

    # --- effort and time ---
    w_action: float = 0.01
    """lambda_a on ||a||^2."""
    w_time: float = 0.0
    """Flat per-step cost, so finishing early beats finishing late."""

    def validate(self):
        if not all(np.isfinite(v) for v in dataclasses.asdict(self).values()):
            raise ValueError("Reward parameters must be finite.")
        if self.w_reference <= 0:
            raise ValueError("Progress reference length must be greater than zero.")
        if self.align_temperature <= 0:
            raise ValueError("Alignment temperature must be greater than zero.")
        if self.align_distance < 0:
            raise ValueError("Alignment radius must not be negative.")
        if self.force_cap <= 0 or self.force_safe <= 0:
            raise ValueError("Force limits must be greater than zero.")
        weights = (self.w_far, self.w_near, self.w_align, self.bonus_contact,
                   self.bonus_grasp, self.bonus_lift, self.w_collision,
                   self.w_severe, self.w_action, self.w_time)
        if min(weights) < 0:
            raise ValueError("Reward weights must not be negative.")


# ==================================================================== terms


def progress_gain(distance, config):
    """w(d) -- high close in, lower far out."""
    spread = config.w_near - config.w_far
    return config.w_far + spread * np.exp(-np.asarray(distance, dtype=float) / config.w_reference)


def progress_potential(distance, config):
    """-integral_0^d w(s) ds, the exact antiderivative of `progress_gain`.

    Paying its difference between steps is the same number as w(d)(d_prev - d)
    for any step small enough that w barely changes across it, and unlike that
    product it cannot be farmed by driving back and forth.
    """
    distance = float(distance)
    spread = config.w_near - config.w_far
    return -(config.w_far * distance
             + spread * config.w_reference * (1.0 - np.exp(-distance / config.w_reference)))


def alignment_gate(distance, config):
    """Stable logistic gate, with g(d_align)=0.5 and no hard boundary."""
    if not config.smooth_alignment:
        return float(distance < config.align_distance)
    x = (float(distance) - config.align_distance) / config.align_temperature
    return float(np.exp(-np.logaddexp(0.0, x)))


def alignment_error(grip_mat, target_mat):
    """Geodesic angle between two rotations, in radians.

    arccos((tr(R_des^T R) - 1) / 2) -- the rotation angle of the residual, which
    is the only rotation metric that does not depend on how either frame was
    parameterised.
    """
    residual = np.asarray(target_mat, dtype=float).T @ np.asarray(grip_mat, dtype=float)
    return float(np.arccos(np.clip((np.trace(residual) - 1.0) / 2.0, -1.0, 1.0)))


def collision_cost(forces, config, contact_force=None):
    """(continuous penalty, severe penalty) for one step's undesired contacts."""
    forces = np.asarray(forces, dtype=float)
    forces = np.abs(forces)
    continuous = config.w_collision * float(np.sum(np.minimum(forces, config.force_cap)))
    peak = float(forces.max()) if forces.size else 0.0
    if contact_force is not None:
        peak = max(peak, abs(float(contact_force)))
    severe = config.w_severe * float(peak > config.force_safe)
    return continuous, severe


def stage_bonus(stages, awarded, config):
    """What the staged indicators pay this step, and the updated latch.

    `stages` is (contact, grasp, lift) as booleans; `awarded` is the same shape
    and records what has already been paid this episode.
    """
    values = (config.bonus_contact, config.bonus_grasp, config.bonus_lift)
    total = 0.0
    updated = list(awarded)
    for i, (reached, value) in enumerate(zip(stages, values)):
        if not reached:
            continue
        if config.latch_bonuses and awarded[i]:
            continue
        total += value
        updated[i] = True
    return total, tuple(updated)


def step_reward(previous_distance, distance, grip_mat, target_mat, stages, awarded,
                forces, action, config, contact_force=None, return_terms=False):
    """The whole reward for one control step, and the updated bonus latch."""
    progress = (progress_potential(distance, config) - progress_potential(previous_distance, config)
                if config.integrated_progress else
                float(progress_gain(distance, config)) * (previous_distance - distance))
    align = (config.w_align * alignment_gate(distance, config)
             * alignment_error(grip_mat, target_mat) if target_mat is not None else 0.0)
    bonus, awarded = stage_bonus(stages, awarded, config)
    continuous, severe = collision_cost(forces, config, contact_force)
    terms = dict(progress=float(progress), alignment=-align, stages=bonus,
                 collision=-continuous, severe=-severe,
                 action=-config.w_action * float(np.sum(np.square(action))),
                 time=-config.w_time)
    result = (float(sum(terms.values())), awarded)
    return result + (terms,) if return_terms else result
