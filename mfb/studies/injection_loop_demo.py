from __future__ import annotations

import numpy as np

from mfb.analysis.stability import approximate_phase_margins, stable
from mfb.studies.compare_boxes import build_acoustic


def run() -> None:
    sys = build_acoustic(3.0).loop
    f = np.logspace(0, 5, 5000)
    margins = approximate_phase_margins(sys.open_loop(), f)
    print("Stable:", stable(sys.closed_loop()))
    print("Phase margins:", margins[:3])


if __name__ == "__main__":
    run()
