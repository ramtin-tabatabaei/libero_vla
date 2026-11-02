"""Resume Grasp failures, High grasp loss in Transport and baselines on DATA.

Additional benchmark arguments, such as --dry-run or --output, override defaults.
"""
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parent
PYTHON = Path('/home/ramtin/miniconda3/envs/libero/bin/python')
OUTPUT = Path('/media/ramtin/DATA/Ramtin/pi05_recovery_images_20261008')


def main():
    environment = os.environ.copy()
    environment.setdefault('NUMBA_CACHE_DIR', '/tmp/libero_vla_numba')
    environment.setdefault('MPLCONFIGDIR', '/tmp/libero_vla_matplotlib')
    environment.setdefault('MUJOCO_GL', 'glx')
    arguments = [
        str(PYTHON), str(PROJECT/'waypoint_recovery_benchmark.py'),
        '--images-only', '--workers', '5', '--port', '8000',
        '--output', str(OUTPUT), '--handoff-delays', '0', '5',
        '--stable-seconds', '4', '--max-policy-steps', '800',
        '--photo-every', '0.5', '--resume',
    ]
    os.chdir(PROJECT)
    os.execve(str(PYTHON), arguments + sys.argv[1:], environment)


if __name__ == '__main__':
    main()
