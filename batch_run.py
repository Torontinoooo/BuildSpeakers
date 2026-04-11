from __future__ import annotations

from pathlib import Path

from mfb.analysis.analyses import MFBAnalyser
from mfb.config.presets import baseline_case, small_box_to_large_box_case, stable_pi_case


def main() -> None:
    cases = [
        baseline_case(),
        stable_pi_case(),
        small_box_to_large_box_case(actual_box_l=2.5, target_box_l=27.5),
    ]
    for case in cases:
        analyser = MFBAnalyser(case)
        outputs = analyser.run_all()
        print(f"\nCase: {case.slug()}")
        for key, path in outputs.items():
            print(f"  {key:>22s}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
