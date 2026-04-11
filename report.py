from __future__ import annotations

from .helpers import pretty_box_comment
from .loop import MFBLoop


def build_summary_text(loop: MFBLoop) -> str:
    cfg = loop.config
    stable, max_real, loop_poles = loop.stability_summary()
    actual_box_l = cfg.speaker.actual_box_volume_L
    target_box_l = cfg.linkwitz.resolve_target_box_L(cfg.speaker)
    fc_actual = cfg.speaker.sealed_box_resonance(actual_box_l)
    qtc_actual = cfg.speaker.sealed_box_q(actual_box_l)
    fc_target = cfg.speaker.sealed_box_resonance(target_box_l)
    qtc_target = cfg.linkwitz.target_qtc

    sensor_lines = [
        f"Sensor: {cfg.sensor.model_label}, axis={cfg.sensor.axis}, supply={cfg.sensor.supply_voltage_v:.2f} V, bandwidth={cfg.sensor.bandwidth_hz:.1f} Hz",
        f"Sensor DC sensitivity ≈ {cfg.sensor.sensitivity_v_per_g * 1e3:.2f} mV/g, zero-g bias ≈ {cfg.sensor.zero_g_bias_v:.3f} V",
    ]
    if cfg.sensor.filter_cap_uF is not None:
        sensor_lines.append(f"Equivalent sensor output capacitor ≈ {cfg.sensor.filter_cap_uF:.6f} µF")
    if cfg.sensor.output_amplifier_bandwidth_hz is not None:
        sensor_lines.append(f"Sensor output amplifier pole ≈ {cfg.sensor.output_amplifier_bandwidth_hz:.1f} Hz")

    lines = [
        "MFB refactored study",
        "=" * 60,
        "Readable structure chosen from the paper block diagrams:",
        "AMP(s) -> SPK(s) -> AccV(s) with F(s), C(s), and R(s) kept separate.",
        f"Source: {cfg.source.source_name}, nominal line level = {cfg.source.nominal_vrms:.3f} Vrms ({cfg.source.nominal_level_dbv:.1f} dBV)",
        f"Actual sealed box: {actual_box_l:.2f} L ({pretty_box_comment(actual_box_l, cfg.speaker.Vas_L)})",
        f"Actual box fc ≈ {fc_actual:.2f} Hz, Qtc ≈ {qtc_actual:.3f}",
        f"Target box ≈ {target_box_l:.2f} L, target fc ≈ {fc_target:.2f} Hz, target Qtc ≈ {qtc_target:.3f}",
        f"Amplifier LP: enabled={cfg.amplifier.lowpass_enabled}, family={cfg.amplifier.lowpass_family}, order={cfg.amplifier.lowpass_order}, fc={cfg.amplifier.lowpass_hz:.1f} Hz",
        f"Amplifier DSP delay: enabled={cfg.amplifier.dsp_delay_enabled}, fixed={cfg.amplifier.dsp_fixed_latency_s * 1e3:.2f} ms, align={cfg.amplifier.output_alignment_delay_s * 1e3:.2f} ms",
        *sensor_lines,
        f"Feedback OFFHP: enabled={cfg.feedback_filters.offset_hp_enabled}, family={cfg.feedback_filters.offset_hp_family}, order={cfg.feedback_filters.offset_hp_order}, fc={cfg.feedback_filters.offset_hp_hz:.2f} Hz",
        f"Feedback NoiseLP: enabled={cfg.feedback_filters.noise_lp_enabled}, family={cfg.feedback_filters.noise_lp_family}, order={cfg.feedback_filters.noise_lp_order}, fc={cfg.feedback_filters.noise_lp_hz:.1f} Hz",
        f"Controller: type={cfg.pid.controller_type}, mode={cfg.pid.mode}",
        f"Resolved controller values: Kp={cfg.pid.resolved_kp:.4g}, Ki={cfg.pid.resolved_ki:.4g}, Kd={cfg.pid.resolved_kd:.4g}, f_leak={cfg.pid.resolved_integral_leak_hz:.3f} Hz, f_d={cfg.pid.resolved_derivative_hz:.1f} Hz",
        f"Linkwitz transform enabled={cfg.linkwitz.enabled}, feasibility k ≈ {cfg.linkwitz.feasibility_k(cfg.speaker):.4f}",
        f"Output LP: enabled={cfg.output_filter.enabled}, family={cfg.output_filter.family}, order={cfg.output_filter.lowpass_order}, fc={cfg.output_filter.lowpass_hz:.1f} Hz, Q={cfg.output_filter.lowpass_q:.3f}",
        f"Closed-loop stable: {stable}, largest pole real part = {max_real:.4f}",
        "Closed-loop poles:",
    ]
    lines.extend(f"  {pole.real:+.5f} {pole.imag:+.5f}j" for pole in loop_poles)
    return "\n".join(lines)
