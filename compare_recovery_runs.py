"""Compare recovery policies on matching condition IDs and eligibility."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

ELIGIBLE = {"recovered", "recovery_failed"}
GROUPS = ("scene", "kind", "intensity", "detection_latency")


def compare(reference, output):
    reference, output = Path(reference), Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    ids = {c["trial_id"] for c in manifest["conditions"]}
    with (reference / "results.csv").open(newline="") as stream:
        baseline = {r["trial_id"]: r for r in csv.DictReader(stream) if r["trial_id"] in ids}
    with (output / "results.csv").open(newline="") as stream:
        repeated = {r["trial_id"]: r for r in csv.DictReader(stream)}
    grouped = defaultdict(list)
    for trial_id in ids:
        if trial_id not in baseline:
            raise ValueError("Reference result missing for " + trial_id)
        original, repeat = baseline[trial_id], repeated.get(trial_id)
        if repeat and any(original[k] != repeat[k] for k in GROUPS):
            raise ValueError("Condition metadata differs for " + trial_id)
        grouped[tuple(original[k] for k in GROUPS)].append((original, repeat))
    def summarize(pairs):
        counts = dict(reference_evaluable=0, reference_recovered=0,
                      repeat_recorded=0, repeat_evaluable=0, repeat_recovered=0,
                      common_evaluable=0, common_reference_recovered=0, common_repeat_recovered=0)
        for original, repeat in pairs:
            a = original["status"] in ELIGIBLE
            b = repeat is not None and repeat["status"] in ELIGIBLE
            counts["reference_evaluable"] += a
            counts["reference_recovered"] += original["status"] == "recovered"
            counts["repeat_recorded"] += repeat is not None
            counts["repeat_evaluable"] += b
            counts["repeat_recovered"] += bool(repeat and repeat["status"] == "recovered")
            counts["common_evaluable"] += a and b
            counts["common_reference_recovered"] += a and b and original["status"] == "recovered"
            counts["common_repeat_recovered"] += a and b and repeat["status"] == "recovered"
        for prefix in ("reference", "repeat", "common_reference", "common_repeat"):
            n = counts["common_evaluable"] if prefix.startswith("common_") else counts[prefix+"_evaluable"]
            counts[prefix+"_percent"] = 100*counts[prefix+"_recovered"]/n if n else None
        return counts
    groups = [dict(zip(GROUPS, key), **summarize(pairs)) for key, pairs in sorted(grouped.items())]
    result = dict(reference=str(reference.resolve()), conditions=len(ids),
        complete=len(repeated) == len(ids),
        overall=summarize([pair for pairs in grouped.values() for pair in pairs]), groups=groups,
        note="Common percentages use only IDs evaluable for both models; full-run denominators are reported separately.")
    from waypoint_recovery_batch import atomic_json
    atomic_json(output / "comparison.json", result)
    temp = output / "comparison.csv.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    temp.replace(output / "comparison.csv")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.output)["overall"], indent=2))


if __name__ == "__main__":
    main()
