from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

from mfb.analysis.analyses import MFBAnalyser
from mfb.config.experiment import ExperimentCase
from mfb.utils.helpers import ensure_directory
from mfb.utils.transfer_helpers import frequency_response, normalized_magnitude_db, phase_margin_and_gain_margin


def _short_case_label(case: ExperimentCase) -> str:
    pid = case.config.pid
    lt = case.config.linkwitz
    return (
        f"Kp={pid.resolved_kp:.3g}\n"
        f"Ki={pid.resolved_ki:.3g}\n"
        f"Q={lt.target_qtc:.2f}"
    )


def _aligned_cl_s_limits(cl_abs: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Make the CL absolute axis and S dB axis line up so that:
      |CL| = 1.0  <->  0 dB
      |CL| = 10^(-3/20) <-> -3 dB
    """
    cl_ymin = 0.08
    cl_ymax = max(3.0, float(np.nanmax(cl_abs)) * 1.15)
    s_ymin = 20.0 * np.log10(cl_ymin)
    s_ymax = 20.0 * np.log10(cl_ymax)
    return (cl_ymin, cl_ymax), (s_ymin, s_ymax)


def plot_sweep_side_by_side(
    cases: list[ExperimentCase],
    output_path: Path,
    figure_title: str,
) -> Path:
    """
    One comparison figure for a sweep.

    Layout:
    row 1: passive small box vs target vs closed loop
    row 2: open-loop OP(s) magnitude
    row 3: CL(s) and S(s) combined
    """
    if not cases:
        raise ValueError("No cases were provided for side-by-side plotting.")

    n = len(cases)
    fig, axes = plt.subplots(
        3,
        n,
        figsize=(4.6 * n, 10.0),
        sharex="col",
        squeeze=False,
    )

    for col, case in enumerate(cases):
        analyser = MFBAnalyser(case)
        freq = analyser.freq_hz
        blocks = analyser.loop.blocks

        # ---------------------------
        # Row 1: passive/target/CL
        # ---------------------------
        ax = axes[0, col]
        ax.semilogx(
            freq,
            normalized_magnitude_db(blocks.open_acoustic_small, freq, 100.0),
            linewidth=2.0,
            label="small box",
        )
        ax.semilogx(
            freq,
            normalized_magnitude_db(blocks.open_acoustic_target, freq, 100.0),
            linewidth=2.0,
            label="target",
        )
        ax.semilogx(
            freq,
            normalized_magnitude_db(blocks.closed_acoustic, freq, 100.0),
            linewidth=2.4,
            label="closed loop",
        )
        ax.set_title(_short_case_label(case), fontsize=10)
        ax.set_ylabel("Acoustic mag [dB]" if col == 0 else "")
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        if col == 0:
            ax.legend(fontsize=8, loc="best")

        # ---------------------------
        # Row 2: OP(s) magnitude only
        # ---------------------------
        ax = axes[1, col]
        op = frequency_response(blocks.open_loop, freq)
        op_db = 20.0 * np.log10(np.maximum(np.abs(op), 1e-20))
        ax.semilogx(freq, op_db, linewidth=2.3, label="OP(s)")
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)

        unity_f, pm, _, _ = phase_margin_and_gain_margin(blocks.open_loop, freq)
        for i, fc in enumerate(unity_f[:2]):
            ax.axvline(fc, color="0.55", linestyle=":", linewidth=0.9, alpha=0.8)
            if i < len(pm):
                ax.text(
                    fc,
                    np.nanmax(op_db) - 4.0 - 8.0 * i,
                    f"{fc:.0f} Hz\nPM {pm[i]:.0f}°",
                    ha="center",
                    va="top",
                    fontsize=8,
                )

        ax.set_ylabel("OP(s) [dB]" if col == 0 else "")
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        if col == 0:
            ax.legend(fontsize=8, loc="best")

        # ---------------------------
        # Row 3: CL(s) + S(s)
        # ---------------------------
        ax_cl = axes[2, col]
        ax_s = ax_cl.twinx()

        CL = op / (1.0 + op)
        S = 1.0 / (1.0 + op)

        cl_abs = np.abs(CL)
        s_db = 20.0 * np.log10(np.maximum(np.abs(S), 1e-20))

        level_m3db_abs = 10.0 ** (-3.0 / 20.0)

        ax_cl.semilogx(freq, cl_abs, linewidth=2.4, label="CL(s)")
        ax_s.semilogx(freq, s_db, linewidth=2.1, linestyle="--", label="S(s)")

        ax_cl.set_yscale("log")
        (cl_ymin, cl_ymax), (s_ymin, s_ymax) = _aligned_cl_s_limits(cl_abs)
        ax_cl.set_ylim(cl_ymin, cl_ymax)
        ax_s.set_ylim(s_ymin, s_ymax)

        # show only 0 dB and negative ticks on S-axis
        tick_start = int(np.floor(s_ymin / 5.0) * 5.0)
        s_ticks = list(np.arange(tick_start, 1, 5))
        if 0.0 not in s_ticks:
            s_ticks.append(0.0)
        s_ticks = sorted(set(float(t) for t in s_ticks if t <= 0.0))
        ax_s.set_yticks(s_ticks)

        ax_cl.axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
        ax_s.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        ax_cl.axhline(level_m3db_abs, color="C3", linestyle="--", linewidth=1.0)
        ax_s.axhline(-3.0, color="C3", linestyle="--", linewidth=1.0)

        # left boundary = rising CL(s), right boundary = falling S(s)
        def interp_crossing_logx(x0, y0, x1, y1, y_target):
            if np.isclose(y1, y0):
                return float(x0)
            lx0 = np.log10(x0)
            lx1 = np.log10(x1)
            lxt = lx0 + (y_target - y0) * (lx1 - lx0) / (y1 - y0)
            return float(10.0 ** lxt)

        def first_rising_crossing_abs(x, y, level):
            i_peak = int(np.argmax(y))
            for i in range(i_peak):
                if y[i] < level <= y[i + 1]:
                    return interp_crossing_logx(x[i], y[i], x[i + 1], y[i + 1], level)
            return float(x[0])

        def falling_edge_crossing_db(x, y_db, level_db):
            for i in range(len(y_db) - 1):
                if y_db[i] >= level_db > y_db[i + 1]:
                    return interp_crossing_logx(x[i], y_db[i], x[i + 1], y_db[i + 1], level_db)
            return float(x[-1])

        f_bw_low = first_rising_crossing_abs(freq, cl_abs, level_m3db_abs)
        f_bw_high = falling_edge_crossing_db(freq, s_db, -3.0)

        ax_cl.axvline(f_bw_low, color="0.35", linestyle="--", linewidth=1.0)
        ax_cl.axvline(f_bw_high, color="0.35", linestyle="--", linewidth=1.0)
        ax_cl.plot(f_bw_low, level_m3db_abs, marker="o", color="C3", markersize=4.0)
        ax_s.plot(f_bw_high, -3.0, marker="o", color="C3", markersize=4.0)

        ax_cl.set_xlabel("Frequency [Hz]")
        ax_cl.set_ylabel("|CL| [abs]" if col == 0 else "")
        ax_s.set_ylabel("S [dB]" if col == n - 1 else "")
        ax_cl.grid(True, which="major", alpha=0.35)
        ax_cl.grid(True, which="minor", alpha=0.12)

        if col == 0:
            lines = [ax_cl.get_lines()[0], ax_s.get_lines()[0]]
            labels = ["CL(s)", "S(s)"]
            ax_cl.legend(lines, labels, fontsize=8, loc="best")

    fig.suptitle(figure_title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path