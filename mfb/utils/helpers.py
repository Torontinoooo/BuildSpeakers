from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np


def log_frequency_grid(f_min_hz: float, f_max_hz: float, n_points: int) -> np.ndarray:
    return np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), int(n_points))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pretty_box_comment(actual_box_l: float, vas_l: float) -> str:
    ratio = actual_box_l / vas_l
    if ratio < 0.2:
        return "very small box, so the box air spring dominates"
    if ratio < 0.5:
        return "small sealed box with clearly increased stiffness"
    if ratio < 1.0:
        return "moderate sealed box"
    return "large box, close to free-air compliance"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(".", "p")
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "experiment"


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
