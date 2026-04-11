from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .experiment import ExperimentCase
from .helpers import ensure_directory, log_frequency_grid
from .loop import MFBLoop
from .params import SensorParams
from .report import build_summary_text
from .transfer_helpers import frequency_response, normalized_magnitude_db, phase_margin_and_gain_margin, poles


class MFBAnalyser:
    def __init__(self, case: ExperimentCase) -> None:
        self.case = case
        self.loop = MFBLoop(case.config)
        self.paths = case.paths()
        self.case.write_metadata()

    @property
    def freq_hz(self) -> np.ndarray:
        analysis = self.case.config.analysis
        return log_frequency_grid(analysis.f_min_hz, analysis.f_max_hz, analysis.n_points)

    @property
    def output_dir(self) -> Path:
        return ensure_directory(self.paths.plots_dir)

    def _prepare_plot_backend(self) -> None:
        if not self.case.config.analysis.show_plots:
            plt.switch_backend("Agg")

    def _finalize(self, figure: plt.Figure, filename: str) -> Path:
        figure.tight_layout()
        out_path = self.output_dir / filename
        if self.case.config.analysis.save_plots:
            figure.savefig(out_path, dpi=180)
        if self.case.config.analysis.show_plots:
            plt.show()
        else:
            plt.close(figure)
        return out_path

    def _mag_db(self, system: signal.TransferFunction) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.abs(frequency_response(system, self.freq_hz)), 1e-20))

    def _phase_deg(self, system: signal.TransferFunction) -> np.ndarray:
        return np.unwrap(np.angle(frequency_response(system, self.freq_hz))) * 180.0 / np.pi

    def _annotate_margin(self, ax_mag, ax_phase, system: signal.TransferFunction) -> None:
        unity_f, pm, phase_f, gm = phase_margin_and_gain_margin(system, self.freq_hz)
        for index, fc in enumerate(unity_f):
            ax_mag.axvline(fc, color="gray", linestyle="--", linewidth=1)
            ax_phase.axvline(fc, color="gray", linestyle="--", linewidth=1)
            mag_here = float(np.interp(np.log10(fc), np.log10(self.freq_hz), self._mag_db(system)))
            phase_here = float(np.interp(np.log10(fc), np.log10(self.freq_hz), self._phase_deg(system)))
            ax_mag.plot(fc, mag_here, marker="o", color="C3")
            ax_phase.plot(fc, phase_here, marker="o", color="C3")
            ax_phase.annotate(f"PM{index + 1} ≈ {pm[index]:.1f}°\n@ {fc:.1f} Hz", xy=(fc, phase_here), xytext=(5, 8), textcoords="offset points", fontsize=8)
        for index, fc in enumerate(phase_f):
            ax_mag.axvline(fc, color="C2", linestyle=":", linewidth=1)
            mag_here = float(np.interp(np.log10(fc), np.log10(self.freq_hz), self._mag_db(system)))
            ax_mag.plot(fc, mag_here, marker="s", color="C2")
            ax_mag.annotate(f"GM{index + 1} ≈ {gm[index]:.1f} dB\n@ {fc:.1f} Hz", xy=(fc, mag_here), xytext=(5, -18), textcoords="offset points", fontsize=8)

    def _sensor_cases(self) -> list[tuple[str, MFBLoop]]:
        cfg = self.case.config
        loops: list[tuple[str, MFBLoop]] = []
        for model in cfg.analysis.sweep_sensor_models:
            supply = 3.0 if model == "adxl327" else 5.0
            bandwidth = 7000.0 if model == "adxl1001" else 1500.0
            sensor = SensorParams(sensor_model=model, axis="x", supply_voltage_v=supply, bandwidth_hz=bandwidth)
            loops.append((sensor.model_label, self.loop.clone_with_sensor(sensor)))
        return loops

    def save_summary(self) -> Path:
        self.paths.summary_txt.write_text(build_summary_text(self.loop), encoding="utf-8")
        return self.paths.summary_txt
    
    def run_essential_case(self) -> dict[str, Path]:
        """
        Run only the essential plots for controller and Linkwitz experiments.

        Saved:
        - summary.txt
        - 02_passive_target_cl.png
        - 04_ol_margin.png
        - 05_cl_sensitivity.png
        """
        return {
            "summary": self.save_summary(),
            "passive_vs_target": self.plot_passive_small_target_cl(),
            "ol_margin": self.plot_ol_phase_margin(),
            "cl_sensitivity": self.plot_cl_and_sensitivity(),
        }

    @staticmethod
    def _control_closed_loop_from_open_loop(open_loop_response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized control response T and sensitivity S.

        For the RMS-style control figure we use the standard loop definitions
        T = L / (1 + L) and S = 1 / (1 + L). That keeps the bandwidth marker
        at |T| = 0.7 meaningful and separate from the physical loudspeaker units.
        """
        sensitivity = 1.0 / (1.0 + open_loop_response)
        closed_loop = open_loop_response * sensitivity
        return closed_loop, sensitivity

    @staticmethod
    def _find_bandwidth_region(freq_hz: np.ndarray, magnitude_abs: np.ndarray, level: float = 0.7) -> tuple[float | None, float | None]:
        above = np.flatnonzero(magnitude_abs >= level)
        if len(above) == 0:
            return None, None
        return float(freq_hz[above[0]]), float(freq_hz[above[-1]])

    @staticmethod
    def _fit_rolloff_slope_db_per_dec(freq_hz: np.ndarray, magnitude_abs: np.ndarray, start_hz: float, stop_hz: float) -> tuple[float | None, np.ndarray | None]:
        if start_hz <= 0.0 or stop_hz <= 0.0 or stop_hz <= start_hz:
            return None, None
        magnitude_db = 20.0 * np.log10(np.maximum(magnitude_abs, 1e-20))
        mask = (freq_hz >= start_hz) & (freq_hz <= stop_hz)
        if np.count_nonzero(mask) < 3:
            return None, None
        x = np.log10(freq_hz[mask])
        y = magnitude_db[mask]
        coeffs = np.polyfit(x, y, 1)
        fit = np.polyval(coeffs, x)
        return float(coeffs[0]), np.column_stack((10.0**x, fit))

    @staticmethod
    def _format_slope(slope_db_per_dec: float | None) -> str:
        if slope_db_per_dec is None or not np.isfinite(slope_db_per_dec):
            return "n/a"
        return f"{slope_db_per_dec:+.0f} dB/dec"

    def _style_control_axes(self, ax: plt.Axes, *, y_label: str) -> None:
        ax.set_xscale("log")
        ax.set_ylabel(y_label)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)

    def plot_transfer_blocks(self) -> Path:
        self._prepare_plot_backend()
        blocks = self.loop.block_dictionary()
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for name, system in blocks.items():
            axes[0].semilogx(self.freq_hz, self._mag_db(system), label=name)
            axes[1].semilogx(self.freq_hz, self._phase_deg(system), label=name)
        axes[0].set_title("Transfer blocks")
        axes[0].set_ylabel("Magnitude [dB]")
        axes[0].grid(True, which="both")
        axes[0].legend(fontsize=8, ncol=2)
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Phase [deg]")
        axes[1].grid(True, which="both")
        return self._finalize(figure, "01_transfer_blocks.png")

    def plot_passive_small_target_cl(self) -> Path:
        self._prepare_plot_backend()
        b = self.loop.blocks
        figure = plt.figure(figsize=(10, 6))
        ax = figure.add_subplot(1, 1, 1)
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.plant_accel_small, self.freq_hz, 200.0), label="Passive small box")
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.plant_accel_target, self.freq_hz, 200.0), label="Passive target box")
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.closed_loop_accel, self.freq_hz, 200.0), label="Closed-loop")
        ax.set_title("Small box, target box, and closed-loop acceleration")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB], normalized at 200 Hz")
        ax.grid(True, which="both")
        ax.legend()
        return self._finalize(figure, "02_passive_target_cl.png")

    def plot_ol_bode(self) -> Path:
        self._prepare_plot_backend()
        b = self.loop.blocks

        figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

        axes[0].semilogx(self.freq_hz, self._mag_db(b.open_loop_no_controller), linewidth=2.0, label="OP0(s)")
        axes[0].semilogx(self.freq_hz, self._mag_db(b.open_loop), linewidth=2.3, label="OP(s)")
        axes[0].axhline(0.0, color="0.35", linestyle=":", linewidth=1.1)
        axes[0].set_title("Open-loop OP(s)")
        axes[0].set_ylabel("Magnitude [dB]")
        axes[0].grid(True, which="major", alpha=0.35)
        axes[0].grid(True, which="minor", alpha=0.12)
        axes[0].legend()

        axes[1].semilogx(self.freq_hz, self._phase_deg(b.open_loop_no_controller), linewidth=2.0, label="OP0(s)")
        axes[1].semilogx(self.freq_hz, self._phase_deg(b.open_loop), linewidth=2.3, label="OP(s)")
        axes[1].axhline(-180.0, color="0.35", linestyle=":", linewidth=1.1)
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Phase [deg]")
        axes[1].grid(True, which="major", alpha=0.35)
        axes[1].grid(True, which="minor", alpha=0.12)
        axes[1].legend()

        return self._finalize(figure, "03_ol_bode.png")

    def plot_ol_phase_margin(self) -> Path:
        self._prepare_plot_backend()
        system = self.loop.blocks.open_loop

        figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

        axes[0].semilogx(self.freq_hz, self._mag_db(system), linewidth=2.3, label="OP(s)")
        axes[1].semilogx(self.freq_hz, self._phase_deg(system), linewidth=2.3, label="OP(s)")

        axes[0].axhline(0.0, color="0.35", linestyle=":", linewidth=1.1)
        axes[1].axhline(-180.0, color="0.35", linestyle=":", linewidth=1.1)

        self._annotate_margin(axes[0], axes[1], system)

        axes[0].set_title("Open-loop OP(s): gain and phase margins")
        axes[0].set_ylabel("Magnitude [dB]")
        axes[0].grid(True, which="major", alpha=0.35)
        axes[0].grid(True, which="minor", alpha=0.12)
        axes[0].legend()

        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Phase [deg]")
        axes[1].grid(True, which="major", alpha=0.35)
        axes[1].grid(True, which="minor", alpha=0.12)
        axes[1].legend()

        return self._finalize(figure, "04_ol_margin.png")

    def plot_cl_and_sensitivity(self) -> Path:
        """
        Combined control plot using:

            OP(s) = open-loop transfer
            CL(s) = OP(s) / (1 + OP(s))
            S(s)  = 1 / (1 + OP(s))

        Display:
        - CL(s) on left y-axis as absolute magnitude
        - S(s)  on right y-axis in dB

        Required alignment:
        - |CL| = 1.0 aligns with   0 dB
        - |CL| = 10^(-3/20) aligns with -3 dB

        Bandwidth definition:
        - left boundary  = rising edge of CL(s) at -3 dB / 0.7079
        - right boundary = falling edge of S(s) at -3 dB
        """
        self._prepare_plot_backend()

        OP = frequency_response(self.loop.blocks.open_loop, self.freq_hz)
        CL = OP / (1.0 + OP)
        S = 1.0 / (1.0 + OP)

        cl_abs = np.abs(CL)
        s_abs = np.abs(S)
        s_db = 20.0 * np.log10(np.maximum(s_abs, 1e-20))

        level_0db_abs = 1.0
        level_m3db_abs = 10.0 ** (-3.0 / 20.0)   # exact abs value for -3 dB

        unity_f, pm, _, _ = phase_margin_and_gain_margin(self.loop.blocks.open_loop, self.freq_hz)

        def interp_crossing_logx(x0: float, y0: float, x1: float, y1: float, y_target: float) -> float:
            if np.isclose(y1, y0):
                return float(x0)
            lx0 = np.log10(x0)
            lx1 = np.log10(x1)
            lxt = lx0 + (y_target - y0) * (lx1 - lx0) / (y1 - y0)
            return float(10.0 ** lxt)

        def first_rising_crossing_abs(x: np.ndarray, y: np.ndarray, level: float) -> float:
            """First rising crossing of CL(s) before the main peak."""
            i_peak = int(np.argmax(y))
            for i in range(i_peak):
                if y[i] < level <= y[i + 1]:
                    return interp_crossing_logx(
                        float(x[i]),
                        float(y[i]),
                        float(x[i + 1]),
                        float(y[i + 1]),
                        level,
                    )
            return float(x[0])

        def falling_edge_crossing_db(x: np.ndarray, y_db: np.ndarray, level_db: float) -> float:
            """
            First falling-edge crossing of S(s).
            This is the right bandwidth boundary you asked for.
            """
            for i in range(len(y_db) - 1):
                if y_db[i] >= level_db > y_db[i + 1]:
                    return interp_crossing_logx(
                        float(x[i]),
                        float(y_db[i]),
                        float(x[i + 1]),
                        float(y_db[i + 1]),
                        level_db,
                    )
            return float(x[-1])

        def fit_slope_db_per_dec_from_abs(
            x: np.ndarray,
            y_abs: np.ndarray,
            f_ref: float,
            side: str,
            decades: float = 0.35,
        ) -> float | None:
            y_db_local = 20.0 * np.log10(np.maximum(y_abs, 1e-20))
            if side == "left":
                f0 = f_ref / (10.0 ** decades)
                f1 = f_ref
            else:
                f0 = f_ref
                f1 = f_ref * (10.0 ** decades)

            mask = (x >= f0) & (x <= f1)
            if np.count_nonzero(mask) < 3:
                return None

            p = np.polyfit(np.log10(x[mask]), y_db_local[mask], 1)
            return float(p[0])

        def fmt_slope(slope: float | None) -> str:
            if slope is None or not np.isfinite(slope):
                return "n/a"
            return f"{slope:+.1f} dB/dec"

        # ------------------------------------------------------------------
        # Bandwidth boundaries
        # ------------------------------------------------------------------
        f_bw_low = first_rising_crossing_abs(self.freq_hz, cl_abs, level_m3db_abs)
        f_bw_high = falling_edge_crossing_db(self.freq_hz, s_db, -3.0)

        # Local slopes
        cl_low_slope = fit_slope_db_per_dec_from_abs(self.freq_hz, cl_abs, f_bw_low, side="left")
        s_high_slope = fit_slope_db_per_dec_from_abs(self.freq_hz, s_abs, f_bw_high, side="right")

        # Sensitivity most-negative point
        i_s_min = int(np.argmin(s_db))
        f_s_min = float(self.freq_hz[i_s_min])
        s_min_db = float(s_db[i_s_min])

        # ------------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------------
        figure, ax_cl = plt.subplots(figsize=(11.2, 6.8))
        ax_s = ax_cl.twinx()

        cl_line = ax_cl.semilogx(
            self.freq_hz,
            cl_abs,
            linewidth=2.6,
            label=r"CL(s) = OP(s)/(1 + OP(s))",
        )[0]

        s_line = ax_s.semilogx(
            self.freq_hz,
            s_db,
            linewidth=2.3,
            linestyle="--",
            label=r"S(s) = 1/(1 + OP(s))",
        )[0]

        # Left axis: CL(s)
        ax_cl.set_yscale("log")
        ax_cl.set_xlabel("Frequency [Hz]")
        ax_cl.set_ylabel("|CL(s)| [abs]")

        cl_ymin = 0.08
        cl_ymax = max(3.0, float(np.nanmax(cl_abs)) * 1.20)
        ax_cl.set_ylim(cl_ymin, cl_ymax)

        # Right axis: aligned dB transform of the CL axis
        s_ymin = 20.0 * np.log10(cl_ymin)
        s_ymax = 20.0 * np.log10(cl_ymax)
        ax_s.set_ylim(s_ymin, s_ymax)
        ax_s.set_ylabel("S(s) [dB]")

        # ------------------------------------------------------------------
        # Show only 0 dB and negative tick labels on S axis
        # even though the aligned axis may mathematically extend above 0 dB.
        # ------------------------------------------------------------------
        tick_start = int(np.floor(s_ymin / 5.0) * 5.0)
        s_ticks = list(np.arange(tick_start, 1, 5))
        if 0.0 not in s_ticks:
            s_ticks.append(0.0)
        s_ticks = sorted(set(float(t) for t in s_ticks if t <= 0.0))
        ax_s.set_yticks(s_ticks)

        # ------------------------------------------------------------------
        # Aligned reference lines
        # ------------------------------------------------------------------
        ax_cl.axhline(level_0db_abs, color="0.45", linestyle=":", linewidth=1.0)
        ax_s.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)

        ax_cl.axhline(level_m3db_abs, color="C3", linestyle="--", linewidth=1.25)
        ax_s.axhline(-3.0, color="C3", linestyle="--", linewidth=1.25)

        # Bandwidth markers
        ax_cl.axvline(f_bw_low, color="0.35", linestyle="--", linewidth=1.1)
        ax_cl.axvline(f_bw_high, color="0.35", linestyle="--", linewidth=1.1)

        # Boundary points
        ax_cl.plot(f_bw_low, level_m3db_abs, marker="o", color="C3", markersize=5)
        ax_s.plot(f_bw_high, -3.0, marker="o", color="C3", markersize=5)

        # Bandwidth arrow
        y_arrow = 0.11
        ax_cl.annotate(
            "",
            xy=(f_bw_low, y_arrow),
            xytext=(f_bw_high, y_arrow),
            arrowprops=dict(arrowstyle="<->", lw=1.3, color="0.35"),
        )
        ax_cl.text(
            np.sqrt(f_bw_low * f_bw_high),
            y_arrow * 1.08,
            f"Bandwidth ≈ {f_bw_low:.1f} Hz to {f_bw_high:.1f} Hz",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        # Level labels
        ax_cl.text(
            self.freq_hz[1],
            level_0db_abs * 1.03,
            "1.0  ↔  0 dB",
            fontsize=9,
            ha="left",
            va="bottom",
        )
        ax_cl.text(
            self.freq_hz[1],
            level_m3db_abs * 1.03,
            f"{level_m3db_abs:.3f}  ↔  -3 dB",
            fontsize=9,
            ha="left",
            va="bottom",
        )

        # Slope labels
        if cl_low_slope is not None and np.isfinite(cl_low_slope):
            ax_cl.text(
                f_bw_low * 1.10,
                level_m3db_abs * 1.35,
                f"CL rising-edge slope ≈ {fmt_slope(cl_low_slope)}",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        if s_high_slope is not None and np.isfinite(s_high_slope):
            ax_s.text(
                f_bw_high * 1.08,
                -5.0,
                f"S falling-edge slope ≈ {fmt_slope(s_high_slope)}",
                fontsize=9,
                ha="left",
                va="top",
            )

        # Most-negative S point
        ax_s.plot(f_s_min, s_min_db, marker="o", color="C1", markersize=5)
        ax_s.annotate(
            f"S min ≈ {s_min_db:.1f} dB @ {f_s_min:.1f} Hz",
            xy=(f_s_min, s_min_db),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=9,
        )

        # Open-loop unity / phase-margin labels
        for i, fc in enumerate(unity_f[:2]):
            ax_cl.axvline(fc, color="0.55", linestyle=":", linewidth=0.9, alpha=0.8)
            if i < len(pm):
                ax_cl.text(
                    fc,
                    2.0,
                    f"OP unity @ {fc:.1f} Hz\nPM ≈ {pm[i]:.1f}°",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

        ax_cl.grid(True, which="major", alpha=0.35)
        ax_cl.grid(True, which="minor", alpha=0.12)
        ax_cl.set_title("Combined CL(s) and S(s) from OP(s)")

        ax_cl.legend([cl_line, s_line], [cl_line.get_label(), s_line.get_label()], loc="best")

        return self._finalize(figure, "05_cl_sensitivity.png")

    def plot_nyquist(self) -> Path:
        self._prepare_plot_backend()
        response_ol0 = frequency_response(self.loop.blocks.open_loop_no_controller, self.freq_hz)
        response_ol = frequency_response(self.loop.blocks.open_loop, self.freq_hz)
        figure = plt.figure(figsize=(7, 7))
        ax = figure.add_subplot(1, 1, 1)
        ax.plot(response_ol0.real, response_ol0.imag, label="OL0")
        ax.plot(response_ol.real, response_ol.imag, label="OL")
        ax.plot(-1.0, 0.0, marker="x", color="C3", markersize=10, label="-1")
        ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1)
        ax.set_title("Nyquist plot")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.grid(True)
        ax.legend()
        ax.axis("equal")
        return self._finalize(figure, "06_nyquist.png")

    def plot_pole_map(self) -> Path:
        self._prepare_plot_backend()
        cl_poles = poles(self.loop.blocks.closed_loop_accel)
        figure = plt.figure(figsize=(7, 6))
        ax = figure.add_subplot(1, 1, 1)
        ax.scatter(cl_poles.real, cl_poles.imag, marker="x", s=80, color="C3")
        ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1)
        ax.set_title("Closed-loop pole map")
        ax.set_xlabel("Real part")
        ax.set_ylabel("Imag part")
        ax.grid(True)
        return self._finalize(figure, "07_cl_poles.png")

    def plot_acoustic_comparison(self) -> Path:
        self._prepare_plot_backend()
        b = self.loop.blocks
        figure = plt.figure(figsize=(10, 6))
        ax = figure.add_subplot(1, 1, 1)
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.open_acoustic_small, self.freq_hz, 200.0), label="Passive small acoustic")
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.open_acoustic_target, self.freq_hz, 200.0), label="Passive target acoustic")
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.closed_acoustic, self.freq_hz, 200.0), label="Closed-loop acoustic")
        ax.set_title("Acoustic comparison")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB], normalized at 200 Hz")
        ax.grid(True, which="both")
        ax.legend()
        return self._finalize(figure, "08_acoustic.png")

    def plot_time_tracking(self, test_freq_hz: float = 40.0, duration_s: float = 0.2, accel_ref_mps2: float = 1.0) -> Path:
        self._prepare_plot_backend()
        time = np.linspace(0.0, duration_s, 6000)
        accel_ref = accel_ref_mps2 * np.sin(2.0 * np.pi * test_freq_hz * time)
        reference_voltage = self.case.config.sensor.dc_sensitivity_v_per_mps2 * accel_ref
        _, accel_out, _ = signal.lsim(self.loop.blocks.closed_loop_accel, U=reference_voltage, T=time)
        figure = plt.figure(figsize=(10, 5))
        ax = figure.add_subplot(1, 1, 1)
        ax.plot(time, accel_ref, label="reference acceleration")
        ax.plot(time, accel_out, label="closed-loop acceleration")
        ax.set_title("Closed-loop acceleration tracking")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [m/s²]")
        ax.grid(True)
        ax.legend()
        return self._finalize(figure, "09_tracking.png")

    def plot_sensor_sweep_ol(self) -> Path:
        self._prepare_plot_backend()
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for label, loop in self._sensor_cases():
            axes[0].semilogx(self.freq_hz, self._mag_db(loop.blocks.open_loop), label=label)
            axes[1].semilogx(self.freq_hz, self._phase_deg(loop.blocks.open_loop), label=label)
        axes[0].axhline(0.0, color="k", linestyle=":", linewidth=1)
        axes[0].set_title("Sensor sweep: open loop")
        axes[0].set_ylabel("Magnitude [dB]")
        axes[0].grid(True, which="both")
        axes[0].legend()
        axes[1].axhline(-180.0, color="k", linestyle=":", linewidth=1)
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Phase [deg]")
        axes[1].grid(True, which="both")
        axes[1].legend()
        return self._finalize(figure, "10_sensor_ol.png")

    def plot_sensor_sweep_cl(self) -> Path:
        self._prepare_plot_backend()
        b = self.loop.blocks
        figure = plt.figure(figsize=(10, 6))
        ax = figure.add_subplot(1, 1, 1)
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.plant_accel_small, self.freq_hz, 200.0), label="Passive small box", linewidth=2)
        ax.semilogx(self.freq_hz, normalized_magnitude_db(b.plant_accel_target, self.freq_hz, 200.0), label="Passive target box", linewidth=2)
        for label, loop in self._sensor_cases():
            ax.semilogx(self.freq_hz, normalized_magnitude_db(loop.blocks.closed_loop_accel, self.freq_hz, 200.0), label=f"CL {label}")
        ax.set_title("Sensor sweep: closed loop")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB], normalized at 200 Hz")
        ax.grid(True, which="both")
        ax.legend(fontsize=8)
        return self._finalize(figure, "11_sensor_cl.png")

    def plot_sensor_sweep_nyquist(self) -> Path:
        self._prepare_plot_backend()
        figure = plt.figure(figsize=(7, 7))
        ax = figure.add_subplot(1, 1, 1)
        for label, loop in self._sensor_cases():
            response = frequency_response(loop.blocks.open_loop, self.freq_hz)
            ax.plot(response.real, response.imag, label=label)
        ax.plot(-1.0, 0.0, marker="x", color="C3", markersize=10, label="-1")
        ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1)
        ax.set_title("Sensor sweep: Nyquist")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.grid(True)
        ax.legend()
        ax.axis("equal")
        return self._finalize(figure, "12_sensor_nyquist.png")

    def save_sensor_sweep_summary(self) -> Path:
        lines = ["Sensor sweep summary", "=" * 60]
        for label, loop in self._sensor_cases():
            unity_f, pm, phase_f, gm = phase_margin_and_gain_margin(loop.blocks.open_loop, self.freq_hz)
            lines.append(f"{label}:")
            lines.append(f"  OL crossover ≈ {unity_f[0]:.2f} Hz, PM ≈ {pm[0]:.2f}°" if unity_f else "  OL crossover: none in scan range")
            lines.append(f"  OL gain margin ≈ {gm[0]:.2f} dB at {phase_f[0]:.2f} Hz" if phase_f else "  OL gain margin: none in scan range")
        out_path = self.paths.case_dir / "sensor_sweep_summary.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def run_single_case(self) -> dict[str, Path]:
        return {
            "summary": self.save_summary(),
            "blocks": self.plot_transfer_blocks(),
            "passive_vs_target": self.plot_passive_small_target_cl(),
            "ol_bode": self.plot_ol_bode(),
            "ol_margin": self.plot_ol_phase_margin(),
            "cl": self.plot_cl_and_sensitivity(),
            "nyquist": self.plot_nyquist(),
            "poles": self.plot_pole_map(),
            "acoustic": self.plot_acoustic_comparison(),
            "tracking": self.plot_time_tracking(),
        }

    def run_sensor_sweep(self) -> dict[str, Path]:
        return {
            "sensor_sweep_summary": self.save_sensor_sweep_summary(),
            "sensor_ol": self.plot_sensor_sweep_ol(),
            "sensor_cl": self.plot_sensor_sweep_cl(),
            "sensor_nyquist": self.plot_sensor_sweep_nyquist(),
        }

    def run_all(self) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        if self.case.config.analysis.run_single_case:
            outputs.update(self.run_single_case())
        if self.case.config.analysis.run_sensor_sweep:
            outputs.update(self.run_sensor_sweep())
        return outputs
