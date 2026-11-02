"""Read-only status of the actual saved plan, including edits after validation."""
import json
from pathlib import Path


def plan_summary(store, suite, task_id):
    path = Path(store) / suite / f"task_{task_id:03d}.json"
    try:
        data = json.loads(path.read_text())
        count, source = len(data["waypoints"]), data.get("source", "manual")
        if count > 9:
            return "Legacy plan", count, False
        if "pi0.5" not in source:
            return "Manual / legacy", count, False
        if "replay passed" in source and "edited" not in source:
            return "Ready · replay passed", count, True
        if "rollout success=False" in source:
            return "Demo did not succeed", count, False
        if "replay FAILED" in source:
            return "Replay needs work", count, False
        return "Needs replay", count, False
    except FileNotFoundError:
        report_path = Path(store) / "_reports" / suite / path.name
        if report_path.exists():
            return "Build needs work", 0, False
        return "Not built yet", 0, False
    except (ValueError, KeyError, TypeError):
        return "Cannot read plan", 0, False
