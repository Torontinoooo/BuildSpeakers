# MFB project structure (paper-aligned)

This repository is now organized to mirror the decomposition used in the papers and design notes:

- **plant**: amplifier + loudspeaker + accelerometer + acoustic radiation.
- **forward path**: configurable controller chain `C(s) = PID(s) + CorrLP(s) + LinkT(s) [+ OutLP(s)]`.
- **plant**: `P(s)` starts at amplifier and includes loudspeaker and accelerometer.
- **feedback path**: `F(s) = CorrHP(s)` for now (designed to accept extra filters later).
- **summing stage**: explicit difference amplifier model used by time-domain flow.
- **experiment workflow**: configuration, case naming, analysis, and scripts are separated.

## Package layout

```text
mfb/
  config/        # params, presets, experiment identity
  plant/         # source, summing amp, amplifier, loudspeaker, sensor, acoustic
  control/       # feedback filter F(s), controller C(s), Linkwitz/output shaping, loop assembler
  models/
    linear_tf/   # low-order transfer-function model (default)
    nonlinear/   # Bl(x), Cms(x), Le(x,i), thermal placeholders
    multifield/  # FE / coupled-field placeholders
  simulation/    # signal-flow and time-domain experiment runner
  analysis/      # plots, metrics, summary text
  circuit/       # hardware-facing assumptions and interfaces
  utils/         # helper and transfer-function math
scripts/         # runnable entry points
```

## Highlighted linear equations

The first model level remains the low-order linear lumped model:

- `u(s) = Re i(s) + s Le i(s) + s Bl x(s)`
- `Bl i(s) = Mms s² x(s) + Rms s x(s) + x(s)/Cms`
- `G_a/v(s) = Bl s² / [Bl² s + (sLe + Re)(s²Mms + sRms + 1/Cms)]`

These equations are documented in `mfb/models/linear_tf/plant_tf.py` and implemented via
`SpeakerParams.voltage_to_acceleration_coefficients(...)`.

## Development order encoded in code

1. Linear transfer-function plant and loop assembly.
2. Explicit sensor path conditioning (`F(s)`).
3. Explicit summing amplifier (`u_sum = k_r u_ref - k_f u_fb + V_bias`).
4. Prefilter and limiter hooks in time-domain experiments.
5. Time-domain signal flow module for tweakable experiments.
6. Nonlinear and multifield directories reserved for later expansion.

Current loop partition:

- forward controller: `C(s) = PID(s) -> CorrLP(s) -> LinkT(s) -> OutLP(s)` (each term optional by config)
- plant: `P(s) = AMP(s) -> SPK(s) -> AccV(s)`
- feedback: `F(s) = CorrHP(s)`

## Run

```bash
python scripts/run_default.py
python scripts/run_time_experiment.py
```
