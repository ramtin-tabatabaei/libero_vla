"""Task-scoped waypoint data and pose geometry. No simulator or GUI dependencies."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import List

import numpy as np

WORLD = "World"
PREVIOUS = "Previous waypoint"
MAX_WAYPOINTS = 9
DOWN = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])


def vector(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("Position must contain three finite numbers.")
    return a


def rotation(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or not np.allclose(a.T @ a, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(a), 1):
        raise ValueError("Orientation must be a rotation matrix.")
    return a


@dataclass
class Waypoint:
    name: str = "Waypoint"
    reference: str = WORLD
    offset: list = field(default_factory=lambda: [0., 0., 0.])
    orientation: list = field(default_factory=lambda: DOWN.tolist())
    orientation_relative_to_previous: bool = False
    motion: str = "reach"
    gripper: str = "keep"
    gripper_after: str = "keep"
    grasp: str = ""
    ignore: list = field(default_factory=list)
    carry: bool = False
    seat: bool = False
    hold_xy: bool = False
    joint: str = ""
    sweep: float = 0.

    def validate(self):
        vector(self.offset)
        rotation(self.orientation)
        if not isinstance(self.orientation_relative_to_previous, bool):
            raise ValueError("Previous-waypoint orientation flag must be true or false.")
        if self.motion not in ("reach", "line", "grip", "arc"):
            raise ValueError("Unknown motion: " + self.motion)
        if self.gripper not in ("keep", "open", "close"):
            raise ValueError("Unknown gripper setting.")
        if self.gripper_after not in ("keep", "open", "close"):
            raise ValueError("Unknown gripper-after setting.")
        if not isinstance(self.reference, str) or not self.reference:
            raise ValueError("Choose a reference.")
        if not np.isfinite(self.sweep):
            raise ValueError("Invalid joint sweep.")


@dataclass
class Chain:
    suite: str
    task_id: int
    language: str
    waypoints: List[Waypoint] = field(default_factory=list)
    source: str = "manual"

    def validate(self):
        for w in self.waypoints:
            w.validate()

    def save(self, path):
        self.validate()
        if len(self.waypoints) > MAX_WAYPOINTS:
            raise ValueError("A task can have at most 9 waypoints. Remove or combine a step before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(version=1, **asdict(self)), indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path, suite, task_id):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.pop("version", None) != 1:
            raise ValueError("Expected a version 1 task waypoint chain (legacy subgoal lists use a different format).")
        if (data.get("suite"), data.get("task_id")) != (suite, task_id):
            raise ValueError("This chain belongs to a different task. Load that task first.")
        data["waypoints"] = [Waypoint(**w) for w in data["waypoints"]]
        chain = cls(**data)
        chain.validate()
        return chain


def resolve(waypoint, poses, previous):
    """Resolve a single anchor; new previous-waypoint rotations use its wrist frame.

    Older saved waypoints lack ``orientation_relative_to_previous`` and retain
    their original world-axis orientation semantics on load.
    """
    waypoint.validate()
    if waypoint.reference == PREVIOUS:
        origin, position_basis = previous[0], np.eye(3)
        orientation_basis = previous[1] if waypoint.orientation_relative_to_previous else np.eye(3)
    elif waypoint.reference == WORLD:
        origin, position_basis = np.zeros(3), np.eye(3)
        orientation_basis = np.eye(3)
    else:
        if waypoint.reference not in poses:
            raise ValueError("Reference missing from scene: " + waypoint.reference)
        origin, position_basis = poses[waypoint.reference]
        orientation_basis = position_basis
    return (origin + position_basis @ vector(waypoint.offset),
            orientation_basis @ rotation(waypoint.orientation))


def relative_waypoint(name, position, orientation, reference, poses, previous, **kwargs):
    relative_previous = kwargs.pop('orientation_relative_to_previous', reference == PREVIOUS)
    if reference == PREVIOUS:
        origin, position_basis = previous[0], np.eye(3)
        orientation_basis = previous[1] if relative_previous else np.eye(3)
    elif reference == WORLD:
        origin, position_basis = np.zeros(3), np.eye(3)
        orientation_basis = np.eye(3)
    else:
        origin, position_basis = poses[reference]
        orientation_basis = position_basis
    return Waypoint(name, reference, (position_basis.T @ (position - origin)).tolist(),
                    (orientation_basis.T @ orientation).tolist(),
                    orientation_relative_to_previous=relative_previous, **kwargs)
