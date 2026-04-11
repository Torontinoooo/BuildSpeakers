from __future__ import annotations

from pathlib import Path

from .analyses import MFBAnalyser
from .experiment import default_experiment_case


def main() -> None:
    case = default_experiment_case()
    analyser = MFBAnalyser(case)
    outputs = analyser.run_all()

    print((case.paths().summary_txt).read_text(encoding="utf-8"))
    print("\nSaved outputs:")
    for key, path in outputs.items():
        print(f"  {key:>22s}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
