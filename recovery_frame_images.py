"""Lossless scene and wrist frames at every LIBERO control step."""
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

CAMERAS = (('scene', 'agentview'), ('wrist', 'robot0_eye_in_hand'))
FIELDS = ('control_step', 'sim_time', 'controller', 'camera', 'image')


class FrameImages:
    """Write an initial image pair and one pair per actual env.step, with an index."""
    def __init__(self, session, trial_folder, image_folder):
        self.session = session
        self.trial_folder, self.folder = Path(trial_folder), Path(image_folder)
        self.counts = Counter(waypoints=0, policy=0)
        self.stream = None

    def start(self):
        # A worker can be interrupted before it commits its trial result. Keep
        # that capture as an earlier attempt, never mix its frames with a retry.
        if self.folder.exists() and any(self.folder.iterdir()):
            self.folder.rename(self.folder.with_name(
                self.folder.name + '.interrupted_' + str(time.time_ns())))
        self.folder.mkdir(parents=True, exist_ok=True)
        self.trial_folder.mkdir(parents=True, exist_ok=True)
        link = self.trial_folder / 'photos'
        if link.is_symlink():
            if link.resolve() != self.folder.resolve():
                raise ValueError('Existing photos link points to another capture')
        elif link.exists():
            if link.resolve() != self.folder.resolve():
                raise ValueError('Existing photos directory must be preserved')
        else:
            link.symlink_to(self.folder.resolve(), target_is_directory=True)
        self.stream = (self.folder / 'frames.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        try:
            self.capture(0, -1)
        except BaseException:
            self.close(False)
            raise

    def capture(self, phase, control_step):
        from PIL import Image
        controller = 'waypoints' if phase == 0 else 'policy'
        now = float(self.session.env.sim.data.time)
        for name, camera in CAMERAS:
            relative = Path(controller) / name / ('frame_%06d_t_%012.3f.png' % (control_step+1, now))
            path = self.folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = self.session.env.sim.render(width=256, height=256, camera_name=camera)
            pixels = np.ascontiguousarray(pixels[::-1])
            temporary = path.with_suffix('.tmp')
            Image.fromarray(pixels).save(str(temporary), format='PNG', compress_level=1)
            temporary.replace(path)
            self.writer.writerow(dict(control_step=control_step, sim_time=repr(now),
                controller=controller, camera=name, image=str(relative)))
            self.counts[controller] += 1
        self.stream.flush()

    def summary(self):
        return dict(enabled=True, capture='live_after_each_control_step', initial_pair=True,
                    cameras=dict(CAMERAS), width=256, height=256, format='PNG',
                    sampling='every env.step; no frame interval or skipped control steps',
                    waypoint_images=self.counts['waypoints'], vla_images=self.counts['policy'],
                    image_index=str(self.trial_folder / 'photos/frames.csv'))

    def close(self, completed):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / 'capture.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(self.summary(), completed=completed), indent=2)+'\n')
        temporary.replace(path)
