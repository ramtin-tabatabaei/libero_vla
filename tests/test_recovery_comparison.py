"""Compare the same conditions without mixing different eligibility sets."""
import csv
import json

from compare_recovery_runs import compare


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_reports_common_cases_and_full_denominators(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir(); new.mkdir()
    rows = [dict(trial_id=str(i), scene='original', kind='collision', intensity='low',
                 detection_latency='off', status=status) for i, status in enumerate(
        ('recovered', 'recovery_failed', 'failure_not_observed', 'recovered'))]
    write_csv(old/'results.csv', rows)
    (new/'manifest.json').write_text(json.dumps({'conditions': [{'trial_id':str(i)} for i in range(3)]}))
    changed = [dict(row) for row in rows[:3]]
    changed[0]['status'] = 'recovery_failed'
    changed[2]['status'] = 'recovered'
    write_csv(new/'results.csv', changed)
    result = compare(old, new)
    overall = result['overall']
    assert result['conditions'] == 3
    assert result['complete']
    assert overall['reference_evaluable'] == 2
    assert overall['repeat_evaluable'] == 3
    assert overall['common_evaluable'] == 2
    assert overall['common_reference_percent'] == 50.
    assert overall['common_repeat_percent'] == 0.
    assert overall['repeat_recovered'] == 1
