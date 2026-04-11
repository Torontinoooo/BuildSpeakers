from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mfb.config.params import default_configuration
from mfb.simulation.time_domain import TimeDomainExperimentRunner


if __name__ == "__main__":
    trace = TimeDomainExperimentRunner(default_configuration()).run()
    print(f"Generated {len(trace.t)} samples, peak acceleration={trace.a_cone.max():.4g} m/s²")
