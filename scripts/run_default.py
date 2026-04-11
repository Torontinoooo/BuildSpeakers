from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mfb.analysis.analyses import MFBAnalyser
from mfb.config.experiment import default_experiment_case


if __name__ == "__main__":
    case = default_experiment_case("default_case")
    MFBAnalyser(case).run_all()
