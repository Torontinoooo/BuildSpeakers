from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
from eksperiment_sweep import run_pi_sweep

if __name__ == "__main__":
    out = run_pi_sweep([0.8, 1.2, 1.6], [0.04, 0.08, 0.16], results_root=Path("results/pi_sweep"))
    print(out)
