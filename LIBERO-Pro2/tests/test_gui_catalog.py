import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from libero_gui_catalog import TaskCatalog, RunSettings, parse_bddl, section, read_client_config


@pytest.fixture(scope="module")
def catalog():
    return TaskCatalog()


def test_task_and_condition_selection_matches_the_released_manifest(catalog):
    assert len(catalog.entries) == 1688
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        assert catalog.tasks(suite) == list(range(10))
        for task in catalog.tasks(suite):
            expected = {e["case_id"] for e in catalog.entries if e["suite"] == suite and e["base_task_id"] == task}
            assert {e["case_id"] for e in catalog.cases(suite, task)} == expected
            for group in ("static", "dynamic", "base"):
                assert all(e["case_kind"] == group for e in catalog.cases(suite, task, group))
            assert catalog.details(suite, task)["instruction"]
    assert "D02" not in {e["case_id"] for e in catalog.cases("libero_goal", 0)}


def test_task_condition_details_come_from_the_selected_file(catalog):
    original = catalog.details("libero_spatial", 0)
    shifted = catalog.details("libero_spatial", 0, "S01")
    dynamic = catalog.details("libero_spatial", 0, "D20")
    assert not original["condition"]
    assert shifted["condition"]["operator"] == ["pose_transform"]
    assert dynamic["condition"]["trigger"] == ["robot_in_motion"]
    assert original["goal"] == shifted["goal"] == dynamic["goal"]
    assert catalog.split("libero_spatial", 0) == "Held-out test"
    assert catalog.split("libero_spatial", 1) == "Training task"


def test_bddl_quotes_comments_and_invalid_parentheses():
    tree = parse_bddl('(define ; ignored (comment)\n (:language "put (cup) here"))')
    assert section(tree, ":language") == [":language", "put (cup) here"]
    with pytest.raises(ValueError):
        parse_bddl("(define (:language broken)")


@pytest.mark.parametrize("field,value", [("seed", -1), ("init_state_id", 50), ("max_steps", 0), ("replan_steps", 51)])
def test_invalid_settings_cannot_be_run(catalog, field, value):
    values = dict(suite="libero_spatial", task_id=0, case="BASE", seed=142, init_state_id=0, max_steps=220, replan_steps=5)
    values[field] = value
    with pytest.raises(ValueError):
        RunSettings(**values).validate(catalog)


def test_generated_command_preserves_selection_and_does_not_embed_credentials(catalog):
    settings = RunSettings("libero_spatial", 0, "S01", 321, 2, 220, 5).validate(catalog)
    command = settings.command()
    for token in ("--variant S01", "--seed 321", "--init-state-id 2", "--live", "client.env"):
        assert token in command
    assert "LIBERO_REMOTE_API_KEY=" not in command


def test_credentials_are_read_without_shell_execution(tmp_path):
    path = tmp_path / "client.env"
    target = tmp_path / "must-not-exist"
    path.write_text("LIBERO_REMOTE_URL='http://127.0.0.1:8000'\n" + f"LIBERO_REMOTE_API_KEY='$(touch {target})'\n")
    config = read_client_config(path)
    assert config["LIBERO_REMOTE_URL"] == "http://127.0.0.1:8000"
    assert config["LIBERO_REMOTE_API_KEY"].startswith("$(touch ")
    assert not target.exists()
