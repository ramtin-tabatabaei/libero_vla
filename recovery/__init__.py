"""PPO recovery skill: drive the Panda back onto the target object.

The repo is a flat pile of modules imported by bare name from its root, and
this is the only directory in it, so importing `subgoal_planner` or
`waypoint_rrt` from in here needs the root on the path. Doing it once, at
package import, keeps every module below free of sys.path juggling and lets the
trainer be launched either as `python -m recovery.train` from the root or as a
bare subprocess from anywhere.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
