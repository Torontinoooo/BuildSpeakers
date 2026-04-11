from __future__ import annotations

import math
import numpy as np

from mfb.config.params import SourceParams, SourceSignalParams


class SourceModel:
    """Generate a real input waveform from the source and signal settings."""

    def __init__(self, source: SourceParams, signal: SourceSignalParams) -> None:
        self.source = source
        self.signal = signal
        self.source.validate()
        self.signal.validate()

    @property
    def dt(self) -> float:
        return 1.0 / self.signal.sample_rate_hz

    def time_vector(self) -> np.ndarray:
        n_samples = int(round(self.signal.duration_s * self.signal.sample_rate_hz))
        return np.arange(n_samples, dtype=float) * self.dt

    def selected_vrms(self) -> float:
        mode = self.signal.level_mode
        if mode == "nominal":
            vrms = self.source.nominal_vrms
        elif mode == "max":
            vrms = self.source.max_vrms
        elif mode == "custom":
            vrms = float(self.signal.custom_vrms)
        else:
            rng = np.random.default_rng(self.signal.seed)
            vrms = float(rng.uniform(self.signal.random_vrms_min, self.signal.random_vrms_max))
        if vrms > self.source.max_vrms:
            raise ValueError("Requested source level exceeds the source maximum")
        return vrms

    def _normalize_to_vrms(self, waveform: np.ndarray, target_vrms: float) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        if rms <= 0.0:
            raise ValueError("Cannot scale a zero waveform to a target Vrms")
        return waveform * (target_vrms / rms)

    def _clip_to_source_limit(self, waveform: np.ndarray) -> np.ndarray:
        peak_limit = math.sqrt(2.0) * self.source.max_vrms
        return np.clip(waveform, -peak_limit, peak_limit)

    def _unit_waveform(self, time: np.ndarray) -> np.ndarray:
        kind = self.signal.kind
        rng = np.random.default_rng(self.signal.seed)
        if kind == "sine":
            return np.sin(2.0 * math.pi * self.signal.frequency_hz * time + self.signal.phase_rad)
        if kind == "square":
            return np.where(np.sin(2.0 * math.pi * self.signal.frequency_hz * time + self.signal.phase_rad) >= 0.0, 1.0, -1.0)
        if kind == "step":
            return np.where(time >= self.signal.step_time_s, 1.0, 0.0)
        if kind == "chirp":
            f0 = self.signal.chirp_f0_hz
            f1 = self.signal.chirp_f1_hz
            k = (f1 - f0) / self.signal.duration_s
            phase = 2.0 * math.pi * (f0 * time + 0.5 * k * time**2)
            return np.sin(phase)
        if kind == "white_noise":
            return rng.standard_normal(size=time.shape)
        if kind == "prbs":
            bit_rate = max(1.0, self.signal.frequency_hz)
            samples_per_bit = max(1, int(round(self.signal.sample_rate_hz / bit_rate)))
            n_bits = int(np.ceil(len(time) / samples_per_bit))
            bits = rng.choice([-1.0, 1.0], size=n_bits)
            return np.repeat(bits, samples_per_bit)[: len(time)]
        if kind == "multisine":
            waveform = np.zeros_like(time)
            for frequency in self.signal.frequencies_hz:
                waveform += np.sin(2.0 * math.pi * frequency * time + rng.uniform(0.0, 2.0 * math.pi))
            return waveform
        waveform = np.zeros_like(time)
        freqs = rng.uniform(self.signal.random_freq_min_hz, self.signal.random_freq_max_hz, size=max(1, self.signal.random_num_tones))
        for frequency in freqs:
            waveform += np.sin(2.0 * math.pi * frequency * time + rng.uniform(0.0, 2.0 * math.pi))
        return waveform

    def generate_voltage(self) -> tuple[np.ndarray, np.ndarray]:
        time = self.time_vector()
        raw = self._unit_waveform(time)
        voltage = self._normalize_to_vrms(raw, self.selected_vrms())
        return time, self._clip_to_source_limit(voltage)
