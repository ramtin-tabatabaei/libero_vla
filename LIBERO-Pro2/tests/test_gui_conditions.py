import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from libero_gui_catalog import TaskCatalog, MaxCatalog, MAX_ROOT, RunSettings
from libero_conditions import PRO2_CONDITIONS, MAX_EVENTS, condition_info


def test_every_published_pro2_condition_has_a_human_name_and_category():
    catalog = TaskCatalog()
    published = {e["case_id"] for e in catalog.entries}
    assert published <= set(PRO2_CONDITIONS)
    for code in published:
        category, title, description = PRO2_CONDITIONS[code]
        assert category and title and description
        assert "_" not in title and not title.startswith(code)
    info = condition_info(catalog, "libero_spatial", 0, "S15")
    assert info["category"] == "Cameras & images"
    assert "camera" in info["title"].lower()
    assert "Before the first action" in info["when"]
    runtime = condition_info(catalog, "libero_spatial", 0, "D01")
    assert runtime["timing"] == "During task"
    assert "before grasping" in runtime["when"]


@pytest.fixture(scope="module")
def max_catalog():
    if not MAX_ROOT.exists():
        pytest.skip("The optional LIBERO-Max checkout is not present")
    return MaxCatalog("lite")


def test_max_selector_preserves_all_real_source_case_identities(max_catalog):
    catalog = max_catalog
    assert len(catalog.entries) == 800
    assert len(catalog.index) == 800
    assert {e["raw_case"]["scenario"]["change_type"] for e in catalog.entries} == set(MAX_EVENTS)
    for e in catalog.entries:
        required = catalog.require(e["suite"], e["base_task_id"], e["case_id"])
        assert required["raw_case"]["case_id"] == e["case_id"]
    for is_pro in (True, False):
        entry = next(e for e in catalog.entries if bool(e["raw_case"].get("substrate_variant")) == is_pro)
        info = condition_info(catalog, entry["suite"], entry["base_task_id"], entry["case_id"])
        assert info["category"] != "Other changes"
        assert info["when"] and info["description"]
        assert catalog.details(entry["suite"], entry["base_task_id"], entry["case_id"])["path"].exists()


def test_full_max_and_lite_are_distinct_manifests(max_catalog):
    full = MaxCatalog("full")
    assert len(full.entries) == 8000
    assert full.benchmark_id == "max"
    assert max_catalog.benchmark_id == "max-lite"
    assert {e["case_id"] for e in max_catalog.entries} <= {e["case_id"] for e in full.entries}


def test_max_command_uses_real_case_id_and_selected_arm(max_catalog):
    e = max_catalog.entries[0]
    settings = RunSettings(e["suite"], e["base_task_id"], e["case_id"], 195, 0, 220, 5,
        benchmark="max-lite", arm="control").validate(max_catalog)
    command = settings.command()
    assert "--profile lite" in command and "--arm control" in command
    assert e["case_id"] in command
    assert "run_max_task.py" in command
