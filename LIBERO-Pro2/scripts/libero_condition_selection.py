"""Grouped condition selection, independent of the currently selected task."""
import json
from tkinter import messagebox

from libero_condition_browser import ConditionBrowser
from libero_conditions import PRO2_LABELS, condition_info
from libero_gui_catalog import SUITES, MaxCatalog


class GroupedConditionSelection:
    def change_benchmark(self, event=None):
        if self.busy:
            return
        self.browser_updating = True
        label = self.benchmark_var.get()
        try:
            if label not in self.catalogs:
                self.catalogs[label] = MaxCatalog("lite" if "Lite" in label else "full")
            self.catalog = self.catalogs[label]
        except (OSError, ValueError) as error:
            messagebox.showerror("Benchmark unavailable", str(error), parent=self.root)
            self.benchmark_var.set("LIBERO-Pro2")
            self.catalog = self.catalogs["LIBERO-Pro2"]
        self.condition_browser = ConditionBrowser(self.catalog)
        self.selected_condition_key = "BASE" if self.catalog.benchmark_id == "pro2" else "target_relocation"
        self.condition_query_var.set("")
        self.condition_timing_var.set("All")
        self.search_var.set("")
        self.all_task_ids, self.task_ids, self.case_entries = [], [], []
        self.active_index = -1
        self.last_selection = None
        self.browser_updating = False
        self.change_suite(reset_tree=True)
        self.check_health()

    def change_suite(self, event=None, reset_tree=False):
        if self.busy:
            return
        self.refresh_condition_tree(reset=reset_tree, select=False)
        self.filter_tasks()

    def refresh_condition_tree(self, *_, reset=False, select=True):
        if self.busy or getattr(self, "browser_updating", False) or not hasattr(self, "condition_browser"):
            return
        tree = self.conditions
        opened = {tree.item(i, "text").rsplit(" (", 1)[0]: tree.item(i, "open") for i in tree.get_children()}
        groups = self.condition_browser.groups(self.suite, self.condition_query_var.get(),
            "All" if self.condition_browser.is_max else self.condition_timing_var.get())
        self.condition_groups = groups
        self.browser_updating = True
        children = tree.get_children()
        if children:
            tree.delete(*children)
        visible = []
        self.condition_group_keys = {}
        for n, group in enumerate(groups):
            parent = "group:" + str(n)
            keys = [item["key"] for item in group["conditions"]]
            self.condition_group_keys[parent] = keys
            if self.condition_query_var.get():
                expanded = True
            elif reset:
                expanded = self.condition_browser.is_max or self.selected_condition_key in keys
            else:
                expanded = opened.get(group["name"], False)
            tree.insert("", "end", iid=parent, text=f"{group['name']} ({len(keys)})", open=expanded, tags=("group",))
            for item in group["conditions"]:
                key = item["key"]
                visible.append(key)
                title = item["title"] if self.condition_browser.is_max else PRO2_LABELS[key] + " · " + key
                timing = {"At start": "Start", "During task": "During", "Original": "—"}[item["timing"]]
                tree.insert(parent, "end", iid="condition:" + key, text=title,
                    values=(timing, item["task_count"]),
                    tags=("unavailable",) if not item["task_count"] else ())
        self.visible_condition_keys = visible
        count, total = len(visible), len(self.condition_browser.definitions)
        self.condition_count_var.set(f"{count} of {total} condition types · {len(groups)} groups\nTasks = compatible tasks in this suite.")
        if self.condition_browser.is_max:
            self.timing_controls.grid_remove()
            self.condition_help_var.set("Choose a change from the 4 groups below. All 8 Max events are available.")
        else:
            self.timing_controls.grid()
            self.condition_help_var.set("Expand a group to see its conditions. Search by name or code, such as S15.")
        key = self.selected_condition_key
        if key in visible:
            tree.selection_set("condition:" + key)
            tree.focus("condition:" + key)
        self.browser_updating = False
        if select and key not in visible:
            preferred = next((item["key"] for g in groups for item in g["conditions"] if item["task_count"]),
                             visible[0] if visible else None)
            self.select_condition(preferred)

    def expand_condition_groups(self, expanded):
        for item in self.conditions.get_children():
            self.conditions.item(item, open=expanded)

    def condition_tree_selected(self, event=None):
        if self.busy or getattr(self, "browser_updating", False):
            return
        selected = self.conditions.selection()
        if not selected:
            return
        item = selected[0]
        if item.startswith("group:"):
            self.conditions.item(item, open=True)
            current = "condition:" + str(self.selected_condition_key)
            if self.conditions.exists(current):
                self.conditions.selection_set(current)
        elif item.startswith("condition:"):
            key = item[len("condition:"):]
            if key != self.selected_condition_key:
                self.select_condition(key)

    def select_condition(self, key):
        if self.busy:
            return
        if key is not None and key not in self.condition_browser.definitions:
            raise ValueError("This condition is not published by the selected benchmark")
        self.selected_condition_key = key
        if key is not None and not self.conditions.exists("condition:" + key):
            self.browser_updating = True
            self.condition_query_var.set("")
            self.condition_timing_var.set("All")
            self.browser_updating = False
            self.refresh_condition_tree(select=False)
        if key is not None and self.conditions.exists("condition:" + key):
            item = "condition:" + key
            self.conditions.item(self.conditions.parent(item), open=True)
            self.conditions.selection_set(item)
            self.conditions.focus(item)
            self.conditions.see(item)
        query = self.search_var.get().strip().lower()
        compatible = self.condition_browser.tasks(self.suite, key) if key else []
        if query and not any(query in self.catalog.task_label(self.suite, t).lower() or query == str(t) for t in compatible):
            self.browser_updating = True
            self.search_var.set("")
            self.browser_updating = False
        self.filter_tasks()

    def filter_tasks(self, *_):
        if self.busy or getattr(self, "browser_updating", False) or not hasattr(self, "condition_browser"):
            return
        previous = self.task_id if self.task_ids and self.active_index >= 0 else None
        key = self.selected_condition_key
        compatible = self.condition_browser.tasks(self.suite, key) if key else []
        query = self.search_var.get().strip().lower()
        self.task_ids = [t for t in compatible if not query or query in self.catalog.task_label(self.suite, t).lower() or query == str(t)]
        self.task_picker.configure(values=[f"{t:02d}  |  " +
            (self.catalog.split(self.suite, t) + "  |  " if self.condition_browser.is_max else "") +
            self.catalog.task_label(self.suite, t) for t in self.task_ids])
        self.slider.configure(to=max(0, len(self.task_ids)-1))
        self.active_index = -1
        if self.task_ids:
            self.set_task(self.task_ids.index(previous) if previous in self.task_ids else 0)
        else:
            self.task_var.set("No matching task — clear the search or choose another suite")
            self.case_entries = []
            self.scenario_picker.configure(values=[])
            self.scenario_var.set("")
            self.task_count_var.set("0 compatible tasks")
            self.instruction_var.set("Choose another condition or clear the search to select a task.")
            self.condition_var.set("")
            if key:
                definition = self.condition_browser.definitions[key]
                self.change_title_var.set(definition["title"])
                self.change_description_var.set(definition["description"])
                self.change_when_var.set("No published task matches this choice in the current suite/search.")
            else:
                self.change_title_var.set("No matching condition")
                self.change_description_var.set("Clear the condition search to show all groups again.")
                self.change_when_var.set("")
            self.last_frames = self.photos = None
            self.clear_frames()
            self.draw_frames()
            self.set_controls(False)
            self.update_command()

    def set_task(self, index):
        if self.busy or self.updating or not self.task_ids:
            return
        index = min(max(0, index), len(self.task_ids)-1)
        if index == self.active_index:
            return
        self.updating = True
        self.active_index = index
        self.task_picker.current(index)
        self.slider.set(index)
        self.updating = False
        self.change_group()

    def list_changed(self, event=None):
        index = self.task_picker.current()
        if index >= 0:
            self.set_task(index)

    @property
    def case(self):
        if not self.task_ids or not self.case_entries:
            return None
        index = self.scenario_picker.current()
        return self.case_entries[index]["case_id"] if 0 <= index < len(self.case_entries) else None

    def change_group(self):
        """Keep the master groups intact; synchronize only the concrete scenario."""
        previous = self.case
        self.case_entries = self.condition_browser.cases(self.suite, self.task_id, self.selected_condition_key)
        labels = []
        for n, entry in enumerate(self.case_entries, 1):
            if self.condition_browser.is_max:
                row = entry["raw_case"]
                scenario = row["scenario"]
                draw = scenario.get("randomization", {}).get("draw_id", 0) + 1
                labels.append(f"{n}: sample {draw}, {scenario.get('severity', 'standard')}")
            else:
                labels.append(entry["case_id"])
        self.scenario_picker.configure(values=labels)
        if self.case_entries:
            ids = [entry["case_id"] for entry in self.case_entries]
            self.scenario_picker.current(ids.index(previous) if previous in ids else 0)
        self.selection_changed()

    def selection_changed(self):
        if self.busy or self.case is None:
            return
        details = self.catalog.details(self.suite, self.task_id, self.case)
        info = condition_info(self.catalog, self.suite, self.task_id, self.case)
        self.task_count_var.set(f"{self.active_index+1} / {len(self.task_ids)} compatible tasks")
        self.title_var.set(f"{self.catalog.display_name} / {SUITES[self.suite]} / Task {self.task_id:02d}")
        self.instruction_var.set(details["instruction"])
        self.change_title_var.set(info["title"])
        self.change_description_var.set(info["description"])
        timing = info["when"]
        if timing.startswith("When "):
            timing = timing[5:6].upper() + timing[6:]
        self.change_when_var.set("When: " + timing + ("   Affects: " + info["targets"] if info["targets"] else ""))
        self.condition_var.set(info["category"] + ("  ·  Source scene: " if self.condition_browser.is_max else "  ·  ") +
            self.catalog.split(self.suite, self.task_id))
        identity = (self.catalog.benchmark_id, self.suite, self.task_id, self.case)
        if identity != self.last_selection:
            self.limit_var.set(str(details["entry"]["max_steps"]))
            if self.condition_browser.is_max:
                record = details["entry"]["raw_case"]
                self.seed_var.set(str(record["policy_seed"]))
                self.init_var.set(str(record["init_state_index"]))
            self.last_selection = identity
        if self.condition_browser.is_max and self.arm_var.get() == "Without mid-task change":
            self.change_title_var.set("Base episode · no mid-task change")
            self.change_description_var.set("Keep this source scene unchanged during execution. The selected event is not injected in this mode.")
            self.change_when_var.set("Switch to ‘With selected change’ to apply: " + info["title"])
        self.set_text(self.detail_text, "Condition reference: " + self.case + "\n\n" + info["description"] +
            "\n\nWhen\n" + info["when"] + "\n\nTask goal\n" + details["goal"] +
            "\n\nPublished parameters\n" + json.dumps(info["technical"], indent=2) + "\n\nSource\n" + str(details["path"]))
        self.last_frames = self.photos = None
        self.clear_frames()
        self.draw_frames()
        self.status_var.set("Ready")
        self.step_var.set("— / —")
        self.elapsed_var.set("0.0 s")
        self.progress["value"] = 0
        self.result_var.set("The task selector contains tasks supporting the selected condition. Preview or run one.")
        self.set_controls(False)
        self.update_command()
