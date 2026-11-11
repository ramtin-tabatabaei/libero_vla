"""Ensure grouped navigation exposes every published condition and real case."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from libero_condition_browser import ConditionBrowser
from libero_conditions import MAX_EVENTS, PRO2_CONDITIONS, PRO2_LABELS
from libero_gui_catalog import MAX_ROOT, SUITES, MaxCatalog, TaskCatalog


def leaves(browser, suite, **kwargs):
    return [item for group in browser.groups(suite, **kwargs) for item in group["conditions"]]


def test_pro2_groups_explain_all_conditions_and_search_by_code_or_meaning():
    browser = ConditionBrowser(TaskCatalog())
    assert set(browser.definitions) == set(PRO2_CONDITIONS)
    assert set(PRO2_LABELS) == set(browser.definitions)
    assert len(browser.groups("libero_spatial")) == 8
    assert {i["key"] for i in leaves(browser, "libero_spatial", query="S15")} == {"S15"}
    assert {i["category"] for i in leaves(browser, "libero_spatial", query="camera")} == {"Cameras & images"}
    assert len(leaves(browser, "libero_spatial", timing="At start")) == 22
    assert len(leaves(browser, "libero_spatial", timing="During task")) == 20
    assert browser.groups("libero_spatial", query="no-such-condition") == []


def test_pro2_condition_first_navigation_excludes_unsupported_tasks():
    browser = ConditionBrowser(TaskCatalog())
    assert 0 not in browser.tasks("libero_goal", "D02")
    assert browser.cases("libero_goal", 0, "D02") == []
    reached = []
    for suite in SUITES:
        for key in browser.definitions:
            for task in browser.tasks(suite, key):
                cases = browser.cases(suite, task, key)
                assert cases
                assert all(e["case_id"] == key for e in cases)
                reached.extend((e["suite"], e["base_task_id"], e["case_id"]) for e in cases)
    assert len(reached) == len(set(reached)) == 1688


@pytest.mark.parametrize("profile,total", [("lite", 800), ("full", 8000)])
def test_max_all_eight_events_remain_visible_and_every_published_setup_is_reachable(profile, total):
    if not MAX_ROOT.exists():
        pytest.skip("The optional LIBERO-Max checkout is not present")
    catalog = MaxCatalog(profile)
    browser = ConditionBrowser(catalog)
    assert set(browser.definitions) == set(MAX_EVENTS)
    reached = []
    for suite in SUITES:
        groups = browser.groups(suite)
        assert len(groups) == 4
        assert {i["key"] for i in leaves(browser, suite)} == set(MAX_EVENTS)
        for item in leaves(browser, suite):
            key = item["key"]
            tasks = browser.tasks(suite, key)
            assert len(tasks) == item["task_count"]
            case_count = 0
            for task in tasks:
                for entry in browser.cases(suite, task, key):
                    assert entry["raw_case"]["scenario"]["change_type"] == key
                    catalog.require(suite, task, entry["case_id"])
                    reached.append(entry["case_id"])
                    case_count += 1
            assert case_count == item["case_count"]
    assert len(reached) == len(set(reached)) == total
