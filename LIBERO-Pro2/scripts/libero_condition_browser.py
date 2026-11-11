"""Benchmark-wide grouped conditions and the tasks that support each choice."""
from collections import defaultdict

from libero_conditions import PRO2_CONDITIONS, MAX_EVENTS, CATEGORY_ORDER


class ConditionBrowser:
    def __init__(self, catalog):
        self.catalog = catalog
        self.is_max = catalog.benchmark_id != "pro2"
        definitions = MAX_EVENTS if self.is_max else PRO2_CONDITIONS
        published = {self.key(entry) for entry in catalog.entries}
        self.definitions = {}
        for key, (category, title, description) in definitions.items():
            if key not in published:
                continue
            self.definitions[key] = dict(key=key, category=category, title=title, description=description,
                timing="During task" if self.is_max or key.startswith("D") else "Original" if key == "BASE" else "At start")
        self.by_suite = defaultdict(list)
        self.by_task = defaultdict(list)
        self.task_sets = defaultdict(set)
        for entry in catalog.entries:
            key = self.key(entry)
            suite, task = entry["suite"], entry["base_task_id"]
            self.by_suite[(suite, key)].append(entry)
            self.by_task[(suite, task, key)].append(entry)
            self.task_sets[(suite, key)].add(task)
        for values in self.by_task.values():
            values.sort(key=lambda entry: entry["case_id"])

    def key(self, entry):
        return entry["raw_case"]["scenario"]["change_type"] if self.is_max else entry["case_id"]

    def tasks(self, suite, key):
        return sorted(self.task_sets[(suite, key)])

    def cases(self, suite, task, key):
        return self.by_task[(suite, task, key)]

    def groups(self, suite, query="", timing="All"):
        query = query.strip().lower()
        grouped = defaultdict(list)
        for key, definition in self.definitions.items():
            text = " ".join(str(value) for value in definition.values()).lower()
            if query and query not in text:
                continue
            if timing != "All" and definition["timing"] != timing:
                continue
            item = dict(definition, task_count=len(self.task_sets[(suite, key)]),
                        case_count=len(self.by_suite[(suite, key)]))
            grouped[item["category"]].append(item)
        order = (["Objects & geometry", "Cameras & images", "Lighting & appearance", "Obstacles & clutter"]
                 if self.is_max else CATEGORY_ORDER)
        names = [name for name in order if name in grouped] + sorted(set(grouped).difference(order))
        return [{"name": name, "conditions": grouped[name]} for name in names]
