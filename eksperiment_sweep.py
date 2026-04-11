from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv

import numpy as np

from .analyses import MFBAnalyser
from .experiment import ExperimentCase, default_experiment_case
from .params import PIDParams
from .transfer_helpers import frequency_response, phase_margin_and_gain_margin
from .sweep_compare import plot_sweep_side_by_side

def _safe_first(values):
    return values[0] if values else None


def _extract_metrics(analyser: MFBAnalyser) -> dict[str, float | str | None]:
    freq = analyser.freq_hz
    op = frequency_response(analyser.loop.blocks.open_loop, freq)
    cl, s = analyser._control_closed_loop_from_open_loop(op)

    cl_abs = np.abs(cl)
    s_db = 20.0 * np.log10(np.maximum(np.abs(s), 1e-20))

    unity_f, pm, _, _ = phase_margin_and_gain_margin(analyser.loop.blocks.open_loop, freq)

    i_cl_peak = int(np.argmax(cl_abs))
    i_s_min = int(np.argmin(s_db))

    return {
        "case": analyser.case.slug(),
        "Kp": analyser.case.config.pid.resolved_kp,
        "Ki": analyser.case.config.pid.resolved_ki,
        "Kd": analyser.case.config.pid.resolved_kd,
        "Qtc_target": analyser.case.config.linkwitz.target_qtc,
        "target_box_L": analyser.case.config.linkwitz.resolve_target_box_L(analyser.case.config.speaker),
        "actual_box_L": analyser.case.config.speaker.actual_box_volume_L,
        "op_unity_hz": _safe_first(unity_f),
        "phase_margin_deg": _safe_first(pm),
        "cl_peak_abs": float(cl_abs[i_cl_peak]),
        "cl_peak_hz": float(freq[i_cl_peak]),
        "s_min_db": float(s_db[i_s_min]),
        "s_min_hz": float(freq[i_s_min]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _base_case(results_root: Path) -> ExperimentCase:
    case = default_experiment_case("sweep_base")
    analysis = replace(
        case.config.analysis,
        results_root=results_root,
        run_single_case=False,
        run_sensor_sweep=False,
        show_plots=False,
        save_plots=True,
    )
    cfg = replace(case.config, analysis=analysis)
    return ExperimentCase(name=case.name, config=cfg)


def run_pi_sweep(
    kp_values: list[float],
    ki_values: list[float],
    results_root: Path = Path("results/pi_sweep"),
) -> Path:
    """
    Sweep PI values with only essential plots + one side-by-side comparison figure.
    """
    base = _base_case(results_root)
    rows: list[dict[str, object]] = []
    cases: list[ExperimentCase] = []

    for kp in kp_values:
        for ki in ki_values:
            pid = PIDParams(
                controller_type="PI",
                mode=base.config.pid.mode,
                Kp=kp,
                Ki=ki,
                Kd=0.0,
                derivative_hz=base.config.pid.derivative_hz,
                integral_leak_hz=base.config.pid.integral_leak_hz,
                r_p_in_ohm=base.config.pid.r_p_in_ohm,
                r_p_fb_ohm=base.config.pid.r_p_fb_ohm,
                r_i_in_ohm=base.config.pid.r_i_in_ohm,
                c_i_f=base.config.pid.c_i_f,
                r_i_leak_ohm=base.config.pid.r_i_leak_ohm,
                controller_gain=base.config.pid.controller_gain,
            )
            cfg = replace(base.config, pid=pid)
            case = ExperimentCase(name=f"pi_kp{kp:g}_ki{ki:g}", config=cfg)

            analyser = MFBAnalyser(case)
            analyser.run_essential_case()
            rows.append(_extract_metrics(analyser))
            cases.append(case)

    csv_path = _write_csv(results_root / "pi_sweep_summary.csv", rows)

    plot_sweep_side_by_side(
        cases=cases,
        output_path=results_root / "pi_sweep_side_by_side.png",
        figure_title="PI sweep comparison",
    )

    return csv_path

def run_linkwitz_q_sweep(
    q_values: list[float],
    target_box_values_l: list[float],
    results_root: Path = Path("results/linkwitz_q_sweep"),
) -> Path:
    """
    Sweep Linkwitz target Qtc and target box size, with a side-by-side comparison figure.
    """
    base = _base_case(results_root)
    rows: list[dict[str, object]] = []
    cases: list[ExperimentCase] = []

    for qtc in q_values:
        for target_box_l in target_box_values_l:
            linkwitz = replace(
                base.config.linkwitz,
                enabled=True,
                target_qtc=qtc,
                target_box_L=target_box_l,
            )
            cfg = replace(base.config, linkwitz=linkwitz)
            case = ExperimentCase(
                name=f"lt_q{qtc:g}_target{target_box_l:g}L",
                config=cfg,
            )

            analyser = MFBAnalyser(case)
            analyser.run_essential_case()
            rows.append(_extract_metrics(analyser))
            cases.append(case)

    csv_path = _write_csv(results_root / "linkwitz_q_sweep_summary.csv", rows)

    plot_sweep_side_by_side(
        cases=cases,
        output_path=results_root / "linkwitz_q_sweep_side_by_side.png",
        figure_title="Linkwitz Q / target-box sweep comparison",
    )

    return csv_path


def main() -> None:
    pi_csv = run_pi_sweep(
        kp_values=[3.25, 3.3, 3.45, 3.57, 4.1, 4.5],
        ki_values=[0.0001,  0.1, 1, 10],
    )
    print(f"Saved PI sweep summary to: {pi_csv.resolve()}")

if __name__ == "__main__":
    main()