"""Metric-specific judge instructions. One call evaluates one secondary metric only."""

RUBRICS = {
    "D1.1": "One question only: do the characters and objects required by the text appear? Ignore counts, attributes, identity, style, and image quality. Score 1 if missing, 3 if only some appear, 5 if every required entity appears. Do not use 2 or 4.",
    "D1.2": "One question only: are the counts required by the text correct? Ignore attributes and identity. Score 1 if the counts are wrong, 3 if only some counts are correct, 5 if every required count is correct. Do not use 2 or 4.",
    "D1.3": "One question only: is each requested attribute attached to the subject the text names, with no attribute swap? Ignore whether a reference image matches. Score 1 if attributes are on the wrong subjects, 3 if only some bindings are correct, 5 if every attribute is on the right subject. Do not use 2 or 4.",
    "D1.5": "One question only: does the single-subject action required by the text actually happen, rather than a merely similar pose? Do not grade trajectory, speed, or reference-motion fidelity. Score 1 if it does not occur, 3 if it only partly occurs, 5 if it occurs. Do not use 2 or 4.",
    "D1.6": "Extract requested actions in order, locate each visible event, then grade omissions and reversals. Evidence must contain event timestamps.",
    "D1.7": "Extract only scene constraints: location, environment, weather, time of day, and indoor/outdoor status. Ignore subject motion, style, and camera instructions unless they change the scene itself.",
    "D1.8": "Check only requested visual style. Distinguish concrete styles from generic quality adjectives; do not grade prompt content.",
    "D1.9": "Check only requested shot size, viewpoint, camera movement, and lens language. Use motion evidence and do not confuse subject motion with camera motion.",
    "D1.10": "Extract every explicit negative constraint and report only visible violations. Absence is uncertain when visibility is poor.",
    "D1.11": "Decompose the prompt into atomic constraints, grade each separately, identify omissions, and fail critical constraints explicitly.",
    "D2.4": "Judge only whether requested or visible materials such as metal, glass, water, skin, cloth, and wood look physically and visually credible.",
    "D2.5": "Judge light direction, shadows, reflections, and illumination consistency using the supplied exposure statistics; do not double-count general aesthetics.",
    "D2.6": "Find only visual artifacts: ghosting, duplication, melting, stretching, impossible anatomy, or corrupted texture. Give timestamped examples.",
    "D4.3": "Track the same character's hairstyle, color, and length over time; report only unexplained drift, accounting for pose and occlusion.",
    "D4.5": "Track requested or clearly visible accessories on the same character and report disappearance, replacement, or drift after accounting for occlusion.",
    "D5.1": "One question only: does the action required by the text run through to completion, not merely begin or show a similar pose? Do not grade trajectory or speed. Score 1 if it does not complete, 3 if it only partly completes, 5 if it completes. Do not use 2 or 4.",
    "D5.3": "Judge only human motion naturalness, using pose evidence when present. Identify implausible joint motion or biomechanics with timestamps.",
    "D5.7": "Compare visible initial and final states with the prompt. Grade only endpoint correctness, not overall smoothness.",
    "D6.2": "One question only: does the hand actually grasp the object required by the text? Two-dimensional overlap alone is not a grasp. Ignore object appearance. Score 1 if there is no grasp, 3 if the grasp is incomplete, 5 if the grasp happens. Do not use 2 or 4.",
    "D6.3": "One question only: does the take, move, or place action required by the text happen, including the object's resulting state? Ignore object appearance. Score 1 if it does not happen, 3 if only part of it happens, 5 if it happens. Do not use 2 or 4.",
    "D6.4": "One question only: does the requested open, close, switch, push, or similar object operation happen, with the object's state changed? Ignore object appearance. Score 1 if it does not happen, 3 if the operation is incomplete, 5 if it happens. Do not use 2 or 4.",
    "D6.5": "One question only: does the requested eating, drinking, pouring, wearing, or riding action happen? Ignore object appearance. Score 1 if it does not happen, 3 if it only partly happens, 5 if it happens. Do not use 2 or 4.",
    "D6.6": "One question only: does the two-person contact, confrontation, cooperation, or other relational action required by the text happen? Do not decide who matches a reference identity. Score 1 if it does not happen, 3 if it only partly happens, 5 if it happens. Do not use 2 or 4.",
    "D6.7": "One question only: does the multi-subject interaction required by the text happen among the participating people? Do not decide who matches a reference identity. Score 1 if it does not happen, 3 if it only partly happens, 5 if it happens. Do not use 2 or 4.",
}


def rubric(metric_id: str) -> str:
    return RUBRICS.get(metric_id, "Evaluate only this metric's definition and ignore all unrelated prompt requirements.")
