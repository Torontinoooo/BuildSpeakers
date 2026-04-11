from __future__ import annotations

"""Small set of readable experiment presets."""

from dataclasses import replace

from .experiment import ExperimentCase, default_experiment_case
from .params import LinkwitzTransformParams, PIDParams, SpeakerParams


def baseline_case() -> ExperimentCase:
    return default_experiment_case("baseline")


def stable_pi_case() -> ExperimentCase:
    case = default_experiment_case("stable_pi")
    cfg = replace(
        case.config,
        pid=PIDParams(controller_type="PI", Kp=1.4, Ki=0.08, Kd=0.0),
    )
    return ExperimentCase(name=case.name, config=cfg)


def small_box_to_large_box_case(actual_box_l: float = 3.0, target_box_l: float = 27.5) -> ExperimentCase:
    case = default_experiment_case("small_to_large_box")
    speaker = replace(case.config.speaker, actual_box_volume_L=actual_box_l)
    linkwitz = replace(case.config.linkwitz, target_box_L=target_box_l)
    cfg = replace(case.config, speaker=speaker, linkwitz=linkwitz)
    return ExperimentCase(name=case.name, config=cfg)
