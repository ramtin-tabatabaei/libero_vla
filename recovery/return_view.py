"""Show which visited primitive/state a trained selector would return to."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from recovery.return_model import ReturnPointSelector, SelectionConfig
from recovery.return_points import Decision


def plot_return_points(decision, report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection='3d')
    groups = {}
    for candidate in decision.candidates:
        groups.setdefault((candidate.waypoint, candidate.primitive), []).append(candidate)
    for (waypoint, primitive), points in groups.items():
        points.sort(key=lambda point: point.state.time)
        xyz = np.array([p.state.position for p in points])
        axes.plot(*xyz.T, marker='.', label=f'{waypoint + 1}: {primitive}')
    axes.scatter(*decision.current.position, color='red', marker='x', s=100, label='current failure/handoff pose')
    selected = next((c for c in decision.candidates if c.id == report['selected']), None)
    if selected:
        axes.scatter(*selected.state.position, color='limegreen', edgecolors='black', s=180,
                     label=f'return: {selected.primitive} at {selected.phase:.1%}')
        title = f'Waypoint {selected.waypoint + 1}, {selected.primitive}, {selected.phase:.1%}'
        if selected.phase_basis == 'visited_prefix_duration':
            title += ' of visited prefix'
        actual = report.get('actual_outcome')
        if actual is not None and not (actual.get('reached') and not actual.get('recovery_violation')):
            title += '\nExecution stopped before safe arrival'
    else:
        title = 'No acceptable return point'
    axes.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)', title=title)
    axes.legend(loc='best', fontsize=8)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--decision', required=True, help='decision.json produced by return_collect')
    parser.add_argument('--output', required=True, help='Output image; ranking JSON is saved beside it')
    args = parser.parse_args()
    decision = Decision.from_dict(json.loads(Path(args.decision).read_text()))
    selector = ReturnPointSelector(args.checkpoint)
    report = selector.select(decision)
    report['planner_checked'] = False
    plot_return_points(decision, report, args.output)
    Path(args.output).with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(selected=report['selected'], reason=report['reason'], image=args.output)))


if __name__ == '__main__':
    main()
