from __future__ import annotations

"""Experiment identity, naming, and per-run folders."""

from dataclasses import dataclass
from pathlib import Path

from mfb.utils.helpers import ensure_directory, slugify, write_json
from mfb.config.params import MFBConfiguration, default_configuration


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    root: Path
    slug: str

    @property
    def case_dir(self) -> Path:
        return ensure_directory(self.root / self.slug)

    @property
    def plots_dir(self) -> Path:
        return ensure_directory(self.case_dir / "plots")

    @property
    def config_json(self) -> Path:
        return self.case_dir / "config.json"

    @property
    def notes_txt(self) -> Path:
        return self.case_dir / "references.txt"

    @property
    def summary_txt(self) -> Path:
        return self.case_dir / "summary.txt"


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    name: str
    config: MFBConfiguration
    slug_text: str | None = None

    def slug(self) -> str:
        if self.slug_text:
            return slugify(self.slug_text)
        speaker = self.config.speaker
        pid = self.config.pid
        target_box_l = self.config.linkwitz.resolve_target_box_L(speaker)
        slug = (
            f"{self.name}"
            f"_vb{speaker.actual_box_volume_L:.1f}L"
            f"_lt{target_box_l:.1f}L"
            f"_q{self.config.linkwitz.target_qtc:.2f}"
            f"_sens-{self.config.sensor.sensor_model}"
            f"_kp{pid.resolved_kp:.3g}"
            f"_ki{pid.resolved_ki:.3g}"
            f"_kd{pid.resolved_kd:.3g}"
        )
        return slugify(slug)

    def paths(self) -> ExperimentPaths:
        return ExperimentPaths(self.config.analysis.results_root, self.slug())

    def write_metadata(self) -> dict[str, Path]:
        paths = self.paths()
        config_path = write_json(paths.config_json, self.config.to_dict())
        notes = (
            "Theory references used in the refactor:\n"
            "- Schneider et al., AES 138 (2015): plant block diagram, voltage-to-acceleration plant, and practical controller shaping.\n"
            "- Munnig Schmidt, Motional Feedback Theory in a Nutshell: loop gain, sensitivity, phase margin, and virtual mass interpretation.\n"
            "- Munnig Schmidt, Acceleration Feedback Design: module partition, sensor-path conditioning, and practical loop-shaping workflow.\n"
        )
        paths.notes_txt.write_text(notes, encoding="utf-8")
        return {"config": config_path, "references": paths.notes_txt}


def default_experiment_case(name: str = "default_case") -> ExperimentCase:
    return ExperimentCase(name=name, config=default_configuration())
