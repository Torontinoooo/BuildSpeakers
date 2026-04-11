from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
from scipy import signal

from .filters import FeedbackMeasurementChain, PIDController, RepairFilterChain
from .params import MFBConfiguration, SensorParams
from .physics import AccelerometerModel, AcousticRadiationModel, AmplifierModel, LoudspeakerModel
from .transfer_helpers import TFMath, phase_margin_and_gain_margin, poles


@dataclass(frozen=True, slots=True)
class PlantBlocks:
    amplifier: signal.TransferFunction
    loudspeaker_small: signal.TransferFunction
    loudspeaker_target: signal.TransferFunction
    accelerometer: signal.TransferFunction
    acoustic: signal.TransferFunction


@dataclass(frozen=True, slots=True)
class ControlBlocks:
    measurement_filter: signal.TransferFunction
    controller: signal.TransferFunction
    repair: signal.TransferFunction


@dataclass(frozen=True, slots=True)
class LoopBlocks:
    plant: PlantBlocks
    control: ControlBlocks
    plant_accel_small: signal.TransferFunction
    plant_accel_target: signal.TransferFunction
    plant_sensor: signal.TransferFunction
    sensor_path: signal.TransferFunction
    forward_path_no_controller: signal.TransferFunction
    forward_path: signal.TransferFunction
    open_loop_no_controller: signal.TransferFunction
    open_loop: signal.TransferFunction
    closed_loop_accel: signal.TransferFunction
    sensitivity: signal.TransferFunction
    open_acoustic_small: signal.TransferFunction
    open_acoustic_target: signal.TransferFunction
    closed_acoustic: signal.TransferFunction


class LoopAssembler:
    """Assemble the MFB loop in readable stages.

    The staged build mirrors the control block structure used in the references:
    plant, sensor path, command shaping, controller, and then closed-loop closure.
    """

    def __init__(self, config: MFBConfiguration) -> None:
        self.config = config

    def build_plant(self) -> PlantBlocks:
        cfg = self.config
        target_box_l = cfg.linkwitz.resolve_target_box_L(cfg.speaker)
        return PlantBlocks(
            amplifier=AmplifierModel(cfg.amplifier).transfer_function(),
            loudspeaker_small=LoudspeakerModel(cfg.speaker).transfer_function(),
            loudspeaker_target=LoudspeakerModel(cfg.speaker, box_volume_L=target_box_l).transfer_function(),
            accelerometer=AccelerometerModel(cfg.sensor).transfer_function(),
            acoustic=AcousticRadiationModel(cfg.speaker, cfg.analysis.rho_air, cfg.analysis.reference_distance_m).transfer_function(),
        )

    def build_control(self) -> ControlBlocks:
        cfg = self.config
        return ControlBlocks(
            measurement_filter=FeedbackMeasurementChain(cfg.feedback_filters).transfer_function(),
            controller=PIDController(cfg.pid).transfer_function(),
            repair=RepairFilterChain(cfg.speaker, cfg.linkwitz, cfg.output_filter).transfer_function(),
        )

    def build(self) -> LoopBlocks:
        plant = self.build_plant()
        control = self.build_control()

        plant_accel_small = TFMath.series(plant.amplifier, plant.loudspeaker_small)
        plant_accel_target = TFMath.series(plant.amplifier, plant.loudspeaker_target)
        plant_sensor = TFMath.series(plant.amplifier, plant.loudspeaker_small, plant.accelerometer)
        sensor_path = TFMath.series(plant.accelerometer, control.measurement_filter)
        forward_path_no_controller = TFMath.series(control.repair, plant.amplifier, plant.loudspeaker_small)
        forward_path = TFMath.series(control.repair, control.controller, plant.amplifier, plant.loudspeaker_small)
        open_loop_no_controller = TFMath.series(forward_path_no_controller, sensor_path)
        open_loop = TFMath.series(forward_path, sensor_path)
        closed_loop_accel = TFMath.negative_feedback(forward_path, sensor_path)
        sensitivity = TFMath.one_over_one_plus(open_loop)
        open_acoustic_small = TFMath.series(control.repair, plant.amplifier, plant.loudspeaker_small, plant.acoustic)
        open_acoustic_target = TFMath.series(plant.amplifier, plant.loudspeaker_target, plant.acoustic)
        closed_acoustic = TFMath.series(closed_loop_accel, plant.acoustic)

        return LoopBlocks(
            plant=plant,
            control=control,
            plant_accel_small=plant_accel_small,
            plant_accel_target=plant_accel_target,
            plant_sensor=plant_sensor,
            sensor_path=sensor_path,
            forward_path_no_controller=forward_path_no_controller,
            forward_path=forward_path,
            open_loop_no_controller=open_loop_no_controller,
            open_loop=open_loop,
            closed_loop_accel=closed_loop_accel,
            sensitivity=sensitivity,
            open_acoustic_small=open_acoustic_small,
            open_acoustic_target=open_acoustic_target,
            closed_acoustic=closed_acoustic,
        )


class MFBLoop:
    def __init__(self, config: MFBConfiguration) -> None:
        self.config = config
        self._blocks: LoopBlocks | None = None

    @property
    def blocks(self) -> LoopBlocks:
        if self._blocks is None:
            self._blocks = LoopAssembler(self.config).build()
        return self._blocks

    def clone_with_sensor(self, sensor: SensorParams) -> "MFBLoop":
        return MFBLoop(replace(self.config, sensor=sensor))

    def block_dictionary(self) -> Dict[str, signal.TransferFunction]:
        blocks = self.blocks
        return {
            "AMP(s)": blocks.plant.amplifier,
            "SPK_small(s)": blocks.plant.loudspeaker_small,
            "SPK_target(s)": blocks.plant.loudspeaker_target,
            "AccV(s)": blocks.plant.accelerometer,
            "F(s)": blocks.control.measurement_filter,
            "C(s)": blocks.control.controller,
            "R(s)": blocks.control.repair,
            "Plant_small(s)": blocks.plant_accel_small,
            "Plant_target(s)": blocks.plant_accel_target,
            "Plant_sensor(s)": blocks.plant_sensor,
            "Forward_noC(s)": blocks.forward_path_no_controller,
            "Forward(s)": blocks.forward_path,
            "OL0(s)": blocks.open_loop_no_controller,
            "OL(s)": blocks.open_loop,
            "CL(s)": blocks.closed_loop_accel,
            "S(s)": blocks.sensitivity,
        }

    def stability_summary(self) -> tuple[bool, float, list[complex]]:
        loop_poles = poles(self.blocks.closed_loop_accel)
        max_real = float(loop_poles.real.max()) if len(loop_poles) else float("nan")
        return max_real < 0.0, max_real, list(loop_poles)

    def stability_metrics(self) -> dict[str, list[float]]:
        analysis = self.config.analysis
        freq_hz = np.logspace(np.log10(analysis.f_min_hz), np.log10(analysis.f_max_hz), int(analysis.n_points))
        ol0_u, ol0_pm, ol0_p, ol0_gm = phase_margin_and_gain_margin(self.blocks.open_loop_no_controller, freq_hz)
        ol_u, ol_pm, ol_p, ol_gm = phase_margin_and_gain_margin(self.blocks.open_loop, freq_hz)
        return {
            "ol0_unity_hz": ol0_u,
            "ol0_pm_deg": ol0_pm,
            "ol0_phase_cross_hz": ol0_p,
            "ol0_gm_db": ol0_gm,
            "ol_unity_hz": ol_u,
            "ol_pm_deg": ol_pm,
            "ol_phase_cross_hz": ol_p,
            "ol_gm_db": ol_gm,
        }
