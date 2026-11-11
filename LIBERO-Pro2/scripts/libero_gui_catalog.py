"""Local task catalogue and run settings shared by the desktop GUI and its checks."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import shlex
import os

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "libero-pro-install/LIBERO-Pro-evaluation"
ASSETS = BUNDLE / "service-source/assets"
SIM_PYTHON = BUNDLE / ".envs/libero-pro/bin/python"
MAX_ROOT = Path(os.environ.get("LIBERO_MAX_ROOT", ROOT.parent / "LIBERO-MAX"))
SUITES = {"libero_spatial": "Spatial", "libero_object": "Object",
          "libero_goal": "Goal", "libero_10": "Long horizon"}


def parse_bddl(text):
    """Parse parentheses and quoted strings without importing the simulator."""
    tokens = re.findall(r'"(?:[^"\\]|\\.)*"|;[^\n]*|[()]|[^\s();]+', text)
    stack, root = [], []
    for token in tokens:
        if token.startswith(";"):
            continue
        if token == "(":
            node = []
            (stack[-1] if stack else root).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced task definition")
            stack.pop()
        else:
            if not stack:
                raise ValueError("Unexpected token outside task definition")
            stack[-1].append(token.strip('"'))
    if stack:
        raise ValueError("Unclosed task definition")
    return root


def section(tree, name):
    for node in tree:
        if isinstance(node, list):
            if node and node[0] == name:
                return node
            found = section(node, name)
            if found is not None:
                return found
    return None


def expression(node):
    return "(" + " ".join(expression(x) if isinstance(x, list) else str(x) for x in node) + ")"


def readable(value):
    return str(value).replace("_", " ")


class TaskCatalog:
    benchmark_id = "pro2"
    display_name = "LIBERO-Pro2"
    def __init__(self, assets=ASSETS):
        self.assets = Path(assets)
        self.entries = json.loads((self.assets / "task_manifest.json").read_text())["tasks"]
        self.index = {(e["suite"], e["base_task_id"], e["case_id"]): e for e in self.entries}
        self.splits = json.loads((self.assets / "benchmark_split_v1.json").read_text())["splits"]

    def tasks(self, suite):
        return sorted({e["base_task_id"] for e in self.entries if e["suite"] == suite})

    def task_label(self, suite, task_id):
        return self.details(suite, task_id)["instruction"]

    def cases(self, suite, task_id, group="all"):
        matches = [e for e in self.entries if e["suite"] == suite and e["base_task_id"] == task_id
                   and (group == "all" or e["case_kind"] == group)]
        return sorted(matches, key=lambda e: (e["case_id"] != "BASE", e["case_id"].startswith("D"), e["case_id"]))

    def require(self, suite, task_id, case):
        return self.index[(suite, task_id, case.upper())]

    def split(self, suite, task_id):
        return "Held-out test" if task_id in self.splits["test"][suite] else "Training task"

    @lru_cache(maxsize=256)
    def details(self, suite, task_id, case="BASE"):
        entry = self.require(suite, task_id, case)
        path = (self.assets / entry["bddl_path"]).resolve()
        path.relative_to(self.assets.resolve())
        tree = parse_bddl(path.read_text())
        language = section(tree, ":language")
        goal = section(tree, ":goal")
        node = section(tree, case.upper()) if case.upper() != "BASE" else None
        fields = {}
        if node:
            for field in node[1:]:
                if isinstance(field, list) and field:
                    fields[field[0].lstrip(":")] = field[1:]
        return {"instruction": " ".join(language[1:]) if language else "Instruction not declared",
                "goal": " ".join(expression(n) if isinstance(n, list) else str(n) for n in (goal or [])[1:]),
                "condition": fields, "path": path, "entry": entry}

    def case_label(self, suite, task_id, case):
        if case == "BASE":
            return "BASE  /  Original task"
        # The base file includes the per-task condition catalogue, so loading menus
        # does not require parsing all 1,688 condition files.
        tree = self._base_tree(suite, task_id)
        node = section(tree, case)
        variant = section([node], ":variant") if node else None
        name = readable(" ".join(variant[1:])) if variant else ("Static condition" if case.startswith("S") else "Runtime condition")
        return f"{case}  /  {name}"

    @lru_cache(maxsize=40)
    def _base_tree(self, suite, task_id):
        return parse_bddl(self.details(suite, task_id)["path"].read_text())


class MaxCatalog(TaskCatalog):
    """Read the real MAX manifests, preserving source-task and scenario identity."""

    def __init__(self, profile="lite", root=MAX_ROOT):
        self.root = Path(root)
        self.profile = profile
        self.benchmark_id = "max-lite" if profile == "lite" else "max"
        self.display_name = "LIBERO-Max Lite" if profile == "lite" else "LIBERO-Max"
        self.manifest_path = self.root / ("benchmark/lite/libero_max_lite.json" if profile == "lite"
                                          else "benchmark/max8000/libero_max_8000.json")
        payload = json.loads(self.manifest_path.read_text())
        self.raw_cases = payload["cases"]
        limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
        self.entries, self.index, self.task_records = [], {}, {}
        for suite in SUITES:
            records = [c for c in self.raw_cases if c["task_suite_name"] == suite]
            keys = sorted({(c["substrate_category"], c["task_index"], c["task_name"]) for c in records})
            ids = {key: i for i, key in enumerate(keys)}
            for record in records:
                task_id = ids[(record["substrate_category"], record["task_index"], record["task_name"])]
                entry = dict(suite=suite, base_task_id=task_id, case_id=record["case_id"],
                             case_kind="dynamic", max_steps=limits[suite], raw_case=record,
                             source_task_id=record["task_index"])
                self.entries.append(entry)
                self.index[(suite, task_id, record["case_id"])] = entry
                self.task_records[(suite, task_id)] = record

    def require(self, suite, task_id, case):
        return self.index[(suite, task_id, case)]

    def split(self, suite, task_id):
        return self.task_records[(suite, task_id)]["substrate_category"].replace("LIBERO-PRO/", "PRO · ").replace("_", " ")

    def task_label(self, suite, task_id):
        record = self.task_records[(suite, task_id)]
        if record.get("substrate_variant"):
            return record["substrate_variant"]["language"]
        name = re.sub(r"^[A-Z_]+SCENE\d+_", "", record["task_name"])
        name = re.split(r"_(?:tb|table)_\d+|_view_[-\d.]|_language_", name, maxsplit=1)[0]
        return name.replace("_", " ")

    @lru_cache(maxsize=256)
    def details(self, suite, task_id, case=None):
        if case is None:
            case = self.cases(suite, task_id)[0]["case_id"]
        entry = self.require(suite, task_id, case)
        record = entry["raw_case"]
        variant = record.get("substrate_variant")
        path = (self.root / ".deps/libero-pro-data/bddl_files" / variant["bddl_file"] if variant else
                self.root / ".deps/LIBERO-plus/libero/libero/bddl_files" / suite / (record["task_name"] + ".bddl"))
        instruction, goal = self.task_label(suite, task_id), "Complete the original source task."
        if path.exists():
            tree = parse_bddl(path.read_text())
            language, declared_goal = section(tree, ":language"), section(tree, ":goal")
            if language:
                instruction = " ".join(language[1:])
            if declared_goal:
                goal = " ".join(expression(n) if isinstance(n, list) else str(n) for n in declared_goal[1:])
        return dict(instruction=instruction, goal=goal, condition=record["scenario"], path=path, entry=entry)

    def case_label(self, suite, task_id, case):
        from libero_conditions import MAX_EVENTS
        record = self.require(suite, task_id, case)["raw_case"]
        event = record["scenario"]["change_type"]
        title = MAX_EVENTS.get(event, ("", event.replace("_", " "), ""))[1]
        return title + f" · seed {record['policy_seed']}"


@dataclass(frozen=True)
class RunSettings:
    suite: str
    task_id: int
    case: str
    seed: int
    init_state_id: int
    max_steps: int
    replan_steps: int
    benchmark: str = "pro2"
    arm: str = "intervention"

    def validate(self, catalog):
        catalog.require(self.suite, self.task_id, self.case)
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 4294967295")
        if not 0 <= self.init_state_id < 50:
            raise ValueError("Initial state must be between 0 and 49")
        if not 1 <= self.max_steps <= 5000:
            raise ValueError("Step limit must be between 1 and 5000")
        if not 1 <= self.replan_steps <= 50:
            raise ValueError("Replan interval must be between 1 and 50")
        return self

    @property
    def variant(self):
        if self.benchmark != "pro2":
            return self.case
        return "base" if self.case.upper() == "BASE" else self.case.upper()

    def command(self):
        if self.benchmark != "pro2":
            arguments = [str(ROOT / "openpi/.venv/bin/python"), "scripts/run_max_task.py",
                "--profile", "lite" if self.benchmark == "max-lite" else "full", "--case-id", self.case,
                "--arm", self.arm, "--seed", str(self.seed), "--init-state-id", str(self.init_state_id),
                "--max-steps", str(self.max_steps), "--replan-steps", str(self.replan_steps), "--live"]
            return f"cd {shlex.quote(str(ROOT))}\n" + shlex.join(arguments)
        arguments = [str(SIM_PYTHON), "scripts/run_pi05_task.py", "--suite", self.suite,
                     "--task-id", str(self.task_id), "--variant", self.variant,
                     "--seed", str(self.seed), "--init-state-id", str(self.init_state_id),
                     "--max-steps", str(self.max_steps), "--replan-steps", str(self.replan_steps), "--live"]
        return (f"cd {shlex.quote(str(ROOT))}\nset -a\nsource "
                "libero-pro-install/LIBERO-Pro-evaluation/client.env\nset +a\n" + shlex.join(arguments))


def read_client_config(path=BUNDLE / "client.env"):
    """Read the generated key/value file without executing shell commands."""
    result = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if separator and name in ("LIBERO_REMOTE_URL", "LIBERO_REMOTE_API_KEY"):
                tokens = shlex.split(value)
                if len(tokens) != 1:
                    raise ValueError(f"Invalid value for {name}")
                result[name] = tokens[0]
    return result
