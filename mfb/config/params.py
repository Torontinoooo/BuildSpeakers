from __future__ import annotations

"""Configuration objects for the refactored motional-feedback study.

The plant model stays deliberately close to the classical lumped electro-mechanical
loudspeaker equations used by Schneider et al. (AES 138, 2015), where the voltage
-to-acceleration transfer follows the usual Re, Le, Bl, Mms, Rms, Cms small-signal
model. The control structure also follows the practical MFB partition used in the
RMS white papers: prefilter / repair path, measurement conditioning, controller,
plant, and sensor.
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal
import math


SignalKind = Literal["sine", "multisine", "chirp", "step", "square", "prbs", "white_noise", "random_sine"]
LevelMode = Literal["nominal", "max", "custom", "random"]
ControllerType = Literal["P", "PI", "PID"]
PIDMode = Literal["gain", "analog_components"]


@dataclass(frozen=True, slots=True)
class SpeakerParams:
    """Linear sealed-box loudspeaker parameters.

    The voltage-to-acceleration coefficients match the standard electrodynamic
    loudspeaker equations used in Schneider et al., AES 138, Eq. (3)-(4).
    """

    Re: float = 6.08
    Le: float = 1e-9
    Fs: float = 31.0
    Qms: float = 2.15
    Qes: float = 0.36
    Mms: float = 14.51e-3
    Cms: float = 1.82e-3
    Sd: float = 104e-4
    Bl: float = 6.9
    Vas_L: float = 27.5
    actual_box_volume_L: float = 3.0
    Rms: float = field(default=0.0)

    def __post_init__(self) -> None:
        for name in ("Re", "Le", "Fs", "Qms", "Qes", "Mms", "Cms", "Sd", "Bl", "Vas_L", "actual_box_volume_L"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.Rms <= 0.0:
            object.__setattr__(self, "Rms", 2.0 * math.pi * self.Fs * self.Mms / self.Qms)

    @property
    def Qts(self) -> float:
        return (self.Qms * self.Qes) / (self.Qms + self.Qes)

    def alpha(self, box_volume_L: float | None = None) -> float:
        volume = self.actual_box_volume_L if box_volume_L is None else box_volume_L
        if volume <= 0.0:
            raise ValueError("box volume must be > 0")
        return self.Vas_L / volume

    def total_compliance(self, box_volume_L: float | None = None) -> float:
        return self.Cms / (1.0 + self.alpha(box_volume_L))

    def sealed_box_resonance(self, box_volume_L: float | None = None) -> float:
        return self.Fs * math.sqrt(1.0 + self.alpha(box_volume_L))

    def sealed_box_q(self, box_volume_L: float | None = None) -> float:
        return self.Qts * math.sqrt(1.0 + self.alpha(box_volume_L))

    def voltage_to_acceleration_coefficients(self, box_volume_L: float | None = None) -> tuple[list[float], list[float]]:
        c_total = self.total_compliance(box_volume_L)
        numerator = [self.Bl, 0.0, 0.0]
        denominator = [
            self.Le * self.Mms,
            self.Le * self.Rms + self.Re * self.Mms,
            self.Le / c_total + self.Re * self.Rms + self.Bl**2,
            self.Re / c_total,
        ]
        return numerator, denominator


@dataclass(frozen=True, slots=True)
class AmplifierParams:
    gain_db: float = 26.3
    lowpass_enabled: bool = False
    lowpass_family: str = "linkwitz_riley"
    lowpass_hz: float = 500.0
    lowpass_order: int = 4
    dsp_delay_enabled: bool = True
    dsp_fixed_latency_s: float = 2.5e-3
    output_alignment_delay_s: float = 0.0
    pade_order: int = 2

    def __post_init__(self) -> None:
        if self.lowpass_family not in {"linkwitz_riley", "butterworth", "bessel"}:
            raise ValueError("Unsupported low-pass family")
        if self.lowpass_hz <= 0.0 or self.lowpass_order <= 0 or self.pade_order <= 0:
            raise ValueError("Amplifier low-pass and delay parameters must be positive")
        if self.dsp_fixed_latency_s < 0.0 or self.output_alignment_delay_s < 0.0:
            raise ValueError("Delays must be non-negative")


@dataclass(frozen=True, slots=True)
class SensorParams:
    """Simple accelerometer abstraction.

    The control-oriented sensor view follows the AES 138 paper: acceleration is
    measured, then limited in bandwidth before it closes the loop. The higher-band
    pole for ADXL1001 also reflects its output amplifier behaviour.
    """

    sensor_model: str = "adxl327"
    axis: str = "x"
    supply_voltage_v: float | None = None
    bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        model = self.sensor_model.lower()
        axis = self.axis.lower()
        object.__setattr__(self, "sensor_model", model)
        object.__setattr__(self, "axis", axis)
        if model not in {"adxl1001", "adxl327", "adxl206"}:
            raise ValueError("Unsupported sensor model")
        if axis not in {"x", "y", "z"}:
            raise ValueError("Unsupported sensor axis")
        if model == "adxl1001" and axis != "x":
            raise ValueError("ADXL1001 is single-axis")
        if model == "adxl206" and axis == "z":
            raise ValueError("ADXL206 has x/y only")
        if self.supply_voltage_v is None:
            object.__setattr__(self, "supply_voltage_v", self.default_supply_voltage_v)
        if self.bandwidth_hz is None:
            object.__setattr__(self, "bandwidth_hz", self.default_bandwidth_hz)
        vmin, vmax = self.allowed_supply_voltage_range_v
        if not (vmin <= float(self.supply_voltage_v) <= vmax):
            raise ValueError("Sensor supply voltage is outside the supported range")
        if float(self.bandwidth_hz) <= 0.0 or float(self.bandwidth_hz) > self.maximum_supported_bandwidth_hz:
            raise ValueError("Sensor bandwidth is outside the supported range")

    @property
    def default_supply_voltage_v(self) -> float:
        return 3.0 if self.sensor_model == "adxl327" else 5.0

    @property
    def default_bandwidth_hz(self) -> float:
        return 7000.0 if self.sensor_model == "adxl1001" else 1500.0

    @property
    def allowed_supply_voltage_range_v(self) -> tuple[float, float]:
        if self.sensor_model == "adxl327":
            return 1.8, 3.6
        if self.sensor_model == "adxl206":
            return 3.0, 6.0
        return 3.0, 5.25

    @property
    def maximum_supported_bandwidth_hz(self) -> float:
        if self.sensor_model == "adxl327":
            return 550.0 if self.axis == "z" else 1600.0
        if self.sensor_model == "adxl206":
            return 2500.0
        return 11000.0

    @property
    def nominal_supply_voltage_v(self) -> float:
        return 3.0 if self.sensor_model == "adxl327" else 5.0

    @property
    def nominal_sensitivity_v_per_g(self) -> float:
        if self.sensor_model == "adxl327":
            return 420e-3
        if self.sensor_model == "adxl206":
            return 312e-3
        return 20e-3

    @property
    def sensitivity_v_per_g(self) -> float:
        return self.nominal_sensitivity_v_per_g * (float(self.supply_voltage_v) / self.nominal_supply_voltage_v)

    @property
    def dc_sensitivity_v_per_mps2(self) -> float:
        return self.sensitivity_v_per_g / 9.81

    @property
    def zero_g_bias_v(self) -> float:
        return float(self.supply_voltage_v) / 2.0

    @property
    def filter_cap_uF(self) -> float | None:
        if self.sensor_model not in {"adxl327", "adxl206"}:
            return None
        rfilt = 32000.0
        return 1e6 / (2.0 * math.pi * rfilt * float(self.bandwidth_hz))

    @property
    def output_amplifier_bandwidth_hz(self) -> float | None:
        return 70000.0 if self.sensor_model == "adxl1001" else None

    @property
    def model_label(self) -> str:
        return self.sensor_model.upper()


@dataclass(frozen=True, slots=True)
class SourceParams:
    source_name: str = "Tascam X4"
    nominal_vrms: float = 0.316
    nominal_level_dbv: float = -10.0
    max_vrms: float = 2.0
    max_level_dbv: float = 6.0

    def validate(self) -> None:
        if self.nominal_vrms <= 0.0 or self.max_vrms <= 0.0:
            raise ValueError("Source voltage must be > 0")
        if self.nominal_vrms > self.max_vrms:
            raise ValueError("nominal_vrms must not exceed max_vrms")


@dataclass(frozen=True, slots=True)
class SourceSignalParams:
    kind: SignalKind = "sine"
    level_mode: LevelMode = "nominal"
    custom_vrms: float | None = None
    random_vrms_min: float = 0.05
    random_vrms_max: float = 1.0
    duration_s: float = 20.0
    sample_rate_hz: float = 48_000.0
    frequency_hz: float = 50.0
    phase_rad: float = 0.0
    chirp_f0_hz: float = 10.0
    chirp_f1_hz: float = 500.0
    frequencies_hz: tuple[float, ...] = (20.0, 40.0, 80.0)
    random_num_tones: int = 5
    random_freq_min_hz: float = 20.0
    random_freq_max_hz: float = 200.0
    step_time_s: float = 0.1
    seed: int | None = 1234

    def validate(self) -> None:
        if self.duration_s <= 0.0 or self.sample_rate_hz <= 0.0:
            raise ValueError("Signal duration and sample rate must be > 0")
        if self.level_mode == "custom" and (self.custom_vrms is None or self.custom_vrms <= 0.0):
            raise ValueError("custom_vrms must be provided and > 0 in custom mode")


@dataclass(frozen=True, slots=True)
class FeedbackFilterParams:
    """Sensor-path conditioning.

    The offset high-pass mirrors the practical need to remove DC bias and drift before
    subtraction. The optional measurement low-pass follows the MFB practice in the
    AES 138 paper and the RMS design note: measurement bandwidth is intentionally
    limited to avoid noise and cone breakup entering the loop.
    """

    offset_hp_enabled: bool = False
    offset_hp_hz: float =1.0
    offset_hp_order: int = 1
    offset_hp_family: str = "butterworth"
    noise_lp_enabled: bool = False
    noise_lp_hz: float = 500.0
    noise_lp_order: int = 2
    noise_lp_family: str = "butterworth"

    def __post_init__(self) -> None:
        if self.offset_hp_family not in {"bessel", "butterworth"} or self.noise_lp_family not in {"bessel", "butterworth"}:
            raise ValueError("Unsupported measurement-filter family")
        if self.offset_hp_hz <= 0.0 or self.noise_lp_hz <= 0.0:
            raise ValueError("Measurement-filter corner frequencies must be > 0")
        if self.offset_hp_order <= 0 or self.noise_lp_order <= 0:
            raise ValueError("Measurement-filter orders must be >= 1")


@dataclass(frozen=True, slots=True)
class PIDParams:
    controller_type: ControllerType = "PI"
    mode: PIDMode = "gain"
    Kp: float = 3.96
    Ki: float = 0.6
    Kd: float = 1.0e-9
    derivative_hz: float = 30000.0
    integral_leak_hz: float = 20.0
    r_p_in_ohm: float = 10e3
    r_p_fb_ohm: float = 14e3
    r_i_in_ohm: float = 100e3
    c_i_f: float = 200e-6
    r_i_leak_ohm: float = 1.59e6
    r_d_in_ohm: float = 10e3
    c_d_f: float = 10e-9
    r_d_rolloff_ohm: float = 22e3
    c_d_rolloff_f: float = 10e-9
    controller_gain: float = 1.1
    corr_lp_enabled: bool = False
    corr_lp_hz: float = 500.0
    corr_lp_order: int = 1
    corr_lp_family: str = "bessel"

    def __post_init__(self) -> None:
        if self.controller_type not in {"P", "PI", "PID"} or self.mode not in {"gain", "analog_components"}:
            raise ValueError("Unsupported PID configuration")
        if any(value < 0.0 for value in (self.Kp, self.Ki, self.Kd)):
            raise ValueError("Kp, Ki and Kd must be >= 0")
        if self.derivative_hz <= 0.0 or self.integral_leak_hz <= 0.0:
            raise ValueError("PID corner frequencies must be > 0")
        if self.corr_lp_hz <= 0.0 or self.corr_lp_order <= 0:
            raise ValueError("CorrLP(s) corner frequency and order must be > 0")
        if self.corr_lp_family not in {"bessel", "butterworth", "linkwitz_riley"}:
            raise ValueError("Unsupported CorrLP(s) family")

    @property
    def resolved_kp(self) -> float:
        if self.mode == "gain":
            return self.Kp
        return self.controller_gain * (self.r_p_fb_ohm / self.r_p_in_ohm)

    @property
    def resolved_ki(self) -> float:
        if self.mode == "gain":
            return self.Ki
        return self.controller_gain * (1.0 / (self.r_i_in_ohm * self.c_i_f))

    @property
    def resolved_integral_leak_hz(self) -> float:
        if self.mode == "gain":
            return self.integral_leak_hz
        return 1.0 / (2.0 * math.pi * self.r_i_leak_ohm * self.c_i_f)

    @property
    def resolved_kd(self) -> float:
        if self.mode == "gain":
            return self.Kd
        return self.controller_gain * (self.r_d_in_ohm * self.c_d_f)

    @property
    def resolved_derivative_hz(self) -> float:
        if self.mode == "gain":
            return self.derivative_hz
        return 1.0 / (2.0 * math.pi * self.r_d_rolloff_ohm * self.c_d_rolloff_f)


@dataclass(frozen=True, slots=True)
class LinkwitzTransformParams:
    """Command-path Linkwitz transform.

    The transform keeps the command path readable: it does not change the plant,
    it asks the loop to make a small sealed box behave like a target closed-box
    alignment. That interpretation follows both the original Linkwitz-transform view
    and the practical repair-path decomposition used in the current project.
    """

    enabled: bool = False
    target_box_L: float | None = None
    target_qtc: float = 0.0001

    def __post_init__(self) -> None:
        if self.target_qtc <= 0.0:
            raise ValueError("target_qtc must be > 0")

    def resolve_target_box_L(self, speaker: SpeakerParams) -> float:
        return speaker.Vas_L if self.target_box_L is None else self.target_box_L

    def feasibility_k(self, speaker: SpeakerParams) -> float:
        fo = speaker.sealed_box_resonance()
        qo = speaker.sealed_box_q()
        fp = speaker.sealed_box_resonance(self.resolve_target_box_L(speaker))
        qp = self.target_qtc
        return (fo / fp - qo / qp) / (qo / qp - fp / fo)


@dataclass(frozen=True, slots=True)
class OutputFilterParams:
    enabled: bool = False
    lowpass_hz: float = 500.0
    lowpass_order: int = 1
    lowpass_q: float = 0.6
    family: str = "bessel"


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    results_root: Path = Path("results")
    show_plots: bool = False
    save_plots: bool = True
    f_min_hz: float = 1.0
    f_max_hz: float = 22_000.0
    n_points: int = 100_000
    reference_distance_m: float = 2.0
    rho_air: float = 1.20
    run_single_case: bool = True
    run_sensor_sweep: bool = True
    sweep_sensor_models: tuple[str, ...] = ("adxl1001", "adxl206", "adxl327")


@dataclass(frozen=True, slots=True)
class MFBConfiguration:
    speaker: SpeakerParams
    amplifier: AmplifierParams
    sensor: SensorParams
    source: SourceParams
    signal: SourceSignalParams
    feedback_filters: FeedbackFilterParams
    pid: PIDParams
    linkwitz: LinkwitzTransformParams
    output_filter: OutputFilterParams
    analysis: AnalysisParams

    def replaced(self, **changes: object) -> "MFBConfiguration":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["analysis"]["results_root"] = str(self.analysis.results_root)
        return data


def default_configuration() -> MFBConfiguration:
    speaker = SpeakerParams(actual_box_volume_L=3.0)
    return MFBConfiguration(
        speaker=speaker,
        amplifier=AmplifierParams(),
        sensor=SensorParams(sensor_model="adxl1001", axis="x", supply_voltage_v=5.0, bandwidth_hz=7000.0),
        source=SourceParams(),
        signal=SourceSignalParams(),
        feedback_filters=FeedbackFilterParams(),
        pid=PIDParams(),
        linkwitz=LinkwitzTransformParams(target_box_L=speaker.Vas_L),
        output_filter=OutputFilterParams(),
        analysis=AnalysisParams(),
    )
