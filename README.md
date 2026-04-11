# MFB OOP refactor

This refactor keeps the earlier continuous-time loudspeaker study working, but makes
it easier to read, extend, and run as named experiments.

## Main ideas

- The plant is assembled in stages: amplifier, loudspeaker, sensor, controller, and repair path.
- Every run is an explicit `ExperimentCase` with a reproducible folder name.
- Every experiment writes its own `config.json`, `references.txt`, `summary.txt`, and `plots/` folder.
- The comments and module docstrings point back to the practical MFB references that motivated the structure.

## Package structure

- `params.py` — immutable configuration objects
- `experiment.py` — case naming and output folders
- `physics.py` — `AMP(s)`, `SPK(s)`, `AccV(s)`, and acoustic model
- `filters.py` — `F(s)`, `C(s)`, `R(s)` pieces
- `loop.py` — readable staged assembly of the loop
- `report.py` — text summary generation
- `analyses.py` — plots and experiment execution
- `run.py` — default entry point

## References reflected in the code

- Schneider et al., AES 138 (2015): plant and control-block structure, practical bandwidth limiting
- Munnig Schmidt, *Motional Feedback Theory in a Nutshell*: loop gain, sensitivity, phase/gain margins
- Munnig Schmidt, *Acceleration Feedback Design*: module partition and experiment-oriented workflow

## Running

From the parent directory:

```bash
python -m mfb_oop_refactored.run
```

Outputs go to:

```text
results/<experiment-slug>/
```
