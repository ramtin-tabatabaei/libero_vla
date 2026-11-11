"""Plain-language condition names, categories and timing for the task browser."""

PRO2_CONDITIONS = {
    "BASE": ("Original task", "No added changes", "Use the original task scene and instruction."),
    "S01": ("Objects & geometry", "Move the target object", "Change the target object's starting position and orientation."),
    "S02": ("Objects & geometry", "Move the destination", "Change where the destination or receiving object starts."),
    "S03": ("Objects & geometry", "Change the working height", "Change the height of the receiving object or its supporting surface."),
    "S04": ("Objects & geometry", "Swap object positions", "Exchange the positions of objects on the same supporting surface."),
    "S05": ("Obstacles & clutter", "Restrict the approach path", "Add geometry that makes the approach or transport path more restricted."),
    "S06": ("Objects & geometry", "Use a different object instance", "Replace an object with another instance while preserving the intended task."),
    "S07": ("Objects & geometry", "Change functional geometry", "Change the geometry of a part the robot must interact with."),
    "S08": ("Objects & geometry", "Change object size", "Change an object's scale or proportions."),
    "S09": ("Objects & geometry", "Change opening or clearance", "Change container openings, clearances, or related pose constraints."),
    "S10": ("Obstacles & clutter", "Add irrelevant objects", "Add clutter that is unrelated to the task goal."),
    "S11": ("Obstacles & clutter", "Add look-alike distractors", "Add objects that resemble the intended target."),
    "S12": ("Cameras & images", "Partly hide the scene", "Keep an occluder in front of part of the camera view."),
    "S13": ("Lighting & appearance", "Change object surfaces", "Change surface color, texture, reflection, or appearance."),
    "S14": ("Lighting & appearance", "Change scene appearance", "Change scene context or the contrast between foreground and background."),
    "S15": ("Cameras & images", "Change camera viewpoint", "Change the fixed camera position or its relationship to the other camera."),
    "S16": ("Cameras & images", "Change camera optics", "Change field of view, resolution, crop, or image distortion."),
    "S17": ("Robot & calibration", "Change the robot's starting pose", "Change initial joints, end-effector orientation, or gripper opening."),
    "S18": ("Robot & calibration", "Change robot calibration", "Apply a calibration offset to the robot base, tool, joints, or grasp center."),
    "S19": ("Cameras & images", "Reduce image quality", "Apply the camera degradation configured for this task, such as noise or blur."),
    "S20": ("Action delivery", "Change action or sensor timing", "Use the task's fixed action rate, delay, or observation timing condition."),
    "S21": ("Language", "Reword the instruction", "Use a different wording for the same task goal."),
    "S22": ("Lighting & appearance", "Change the visual theme", "Apply the task's additional scene appearance variation."),
    "D01": ("Objects & geometry", "Move the target during approach", "Move or rotate the target while the robot is approaching it."),
    "D02": ("Objects & geometry", "Move the destination during execution", "Move the receiving object after grasping or before placement."),
    "D03": ("Obstacles & clutter", "Change destination occupancy", "Occupy or release the receiving area before placement."),
    "D04": ("Cameras & images", "Temporarily block the camera", "Move an occluder into the view, hold it there, then remove it."),
    "D05": ("Obstacles & clutter", "Insert an obstacle in the path", "Move an obstacle into the transport path during the task."),
    "D06": ("Objects & geometry", "Rearrange nearby objects", "Move a distractor or rearrange the local scene during execution."),
    "D07": ("Cameras & images", "Move the camera during execution", "Introduce camera shake, drift, or a lasting viewpoint offset."),
    "D08": ("Cameras & images", "Interrupt a camera feed", "Temporarily drop out the configured camera and then restore it."),
    "D09": ("Lighting & appearance", "Change light or exposure", "Change illumination, shadow, or exposure during the task."),
    "D10": ("Cameras & images", "Drop or freeze video frames", "Drop, repeat, or freeze frames in the observation stream."),
    "D11": ("Cameras & images", "Misalign observation timing", "Offset timestamps or temporarily freeze part of the observed state."),
    "D12": ("Action delivery", "Introduce a delay spike", "Temporarily increase command or observation latency."),
    "D13": ("Action delivery", "Desynchronize arm and gripper", "Introduce timing jitter or delay between arm and gripper commands."),
    "D14": ("Action delivery", "Drop or repeat a command", "Skip or repeat selected robot action commands."),
    "D15": ("Action delivery", "Interrupt an action plan", "Introduce stale or late action chunks, or interrupt replanning."),
    "D16": ("Robot & calibration", "Briefly alter robot control", "Apply a bounded action takeover or reachable robot-state offset."),
    "D17": ("Robot & calibration", "Make a grasp fail", "Cause a recoverable grasp failure so the robot must retry."),
    "D18": ("Robot & calibration", "Drop the held object", "Release the object into a reachable, collision-free recovery state."),
    "D20": ("Robot & calibration", "Displace the robot joints", "Apply the task's bounded robot-joint displacement during motion."),
    "D21": ("Language", "Reword the instruction mid-task", "Restate or clarify the instruction without changing the goal."),
}

MAX_EVENTS = {
    "target_relocation": ("Objects & geometry", "Move the target", "Relocate the target object after execution has started."),
    "receptacle_relocation": ("Objects & geometry", "Move the destination", "Relocate the receiving object during the task."),
    "camera_shift": ("Cameras & images", "Shift the camera", "Change the camera viewpoint while the robot is working."),
    "sensor_noise_onset": ("Cameras & images", "Introduce sensor noise", "Turn on the selected image-noise condition during execution."),
    "illumination_switch": ("Lighting & appearance", "Switch the lighting", "Change scene lighting after the task has begun."),
    "visual_theme_switch": ("Lighting & appearance", "Switch the visual theme", "Change the configured colors or textures during execution."),
    "distractor_burst": ("Obstacles & clutter", "Add distractor objects", "Introduce extra objects into the scene during execution."),
    "obstacle_insertion": ("Obstacles & clutter", "Insert an obstacle", "Add an obstacle to the robot's path during execution."),
}

# Compact navigation labels keep the full explanation on the selection card.
PRO2_LABELS = {'BASE': 'Original task',
 'S01': 'Move target',
 'S02': 'Move destination',
 'S03': 'Working height',
 'S04': 'Swap positions',
 'S05': 'Restrict approach',
 'S06': 'Object instance',
 'S07': 'Functional geometry',
 'S08': 'Object size',
 'S09': 'Opening & clearance',
 'S10': 'Extra objects',
 'S11': 'Look-alike objects',
 'S12': 'Scene occlusion',
 'S13': 'Object surfaces',
 'S14': 'Scene appearance',
 'S15': 'Camera viewpoint',
 'S16': 'Camera optics',
 'S17': 'Robot starting pose',
 'S18': 'Robot calibration',
 'S19': 'Image quality',
 'S20': 'Action/sensor timing',
 'S21': 'Reword instruction',
 'S22': 'Visual theme',
 'D01': 'Move target',
 'D02': 'Move destination',
 'D03': 'Destination occupancy',
 'D04': 'Temporary occlusion',
 'D05': 'Path obstacle',
 'D06': 'Rearrange objects',
 'D07': 'Camera motion',
 'D08': 'Camera interruption',
 'D09': 'Light & exposure',
 'D10': 'Drop/freeze frames',
 'D11': 'Observation timing',
 'D12': 'Delay spike',
 'D13': 'Arm/gripper timing',
 'D14': 'Drop/repeat command',
 'D15': 'Interrupt action plan',
 'D16': 'Robot control',
 'D17': 'Grasp failure',
 'D18': 'Drop held object',
 'D20': 'Joint displacement',
 'D21': 'Reword instruction'}

CATEGORY_ORDER = ["Original task", "Objects & geometry", "Obstacles & clutter", "Cameras & images",
                  "Lighting & appearance", "Robot & calibration", "Action delivery", "Language"]


def friendly_target(value):
    import re
    names = {"agentview": "main camera", "robot0_eye_in_hand": "wrist camera", "task_instruction": "task instruction",
             "robot0": "robot", "action_chunk": "planned actions", "replan": "replanning"}
    if str(value) in names:
        return names[str(value)]
    return re.sub(r"_\d+$", "", str(value)).replace("robot0", "robot").replace("_", " ")


def trigger_description(trigger):
    if isinstance(trigger, dict):
        kind, target = trigger.get("type", ""), friendly_target(trigger.get("value", ""))
        if kind == "on_proximity":
            distance = trigger.get("distance_m")
            return f"When the gripper is within {distance:g} m of {target}." if isinstance(distance, (float, int)) else f"When the robot approaches {target}."
        if kind in ("at_step", "step", "on_step"):
            return f"At execution step {trigger.get('value')}."
        return "During execution, when the selected scenario's trigger is reached."
    text = " ".join(trigger) if isinstance(trigger, list) else str(trigger)
    return {
        "approach_or_pregrasp": "While approaching the target, before grasping.",
        "postgrasp_or_preplace": "After grasping, or just before placement.",
        "preplace": "Just before the robot places the object.",
        "transport": "While the robot is transporting the object.",
        "grasp_attempt": "When the robot attempts a grasp.",
        "place_attempt": "When the robot attempts placement.",
        "robot_in_motion": "After the robot starts moving.",
        "state_change": "When the task reaches the configured state change.",
        "task_phase": "At the task phase configured by the benchmark.",
    }.get(text, "During execution, at the trigger configured for this task.")


def condition_info(catalog, suite, task_id, case):
    details = catalog.details(suite, task_id, case)
    if getattr(catalog, "benchmark_id", "pro2") != "pro2":
        record = details["entry"]["raw_case"]
        scenario = record["scenario"]
        category, title, description = MAX_EVENTS.get(scenario["change_type"],
            ("Other changes", scenario["change_type"].replace("_", " "), "Apply this published LIBERO-Max event."))
        return dict(category=category, title=title, description=description,
                    when=trigger_description(scenario["trigger"]),
                    targets=friendly_target(scenario["change"].get("object", scenario["change"].get("camera", "scene"))),
                    code=case, timing="During task", technical=scenario)
    category, title, description = PRO2_CONDITIONS[case]
    fields = details["condition"]
    timing = "Original" if case == "BASE" else ("At start" if case.startswith("S") else "During task")
    when = ("Throughout the episode; no additional change is applied." if case == "BASE" else
            "Before the first action, when the environment resets." if case.startswith("S") else
            trigger_description(fields.get("trigger", [])))
    return dict(category=category, title=title, description=description, when=when,
                targets=", ".join(friendly_target(t) for t in fields.get("targets", [])),
                code=case, timing=timing, technical=fields)
