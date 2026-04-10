# BuildSpeakers

Object-oriented educational toolkit for **accelerometer-based motional feedback (MFB)** loudspeaker simulation.

## What is included

This repository now contains a buildable Python package `mfb/` with:

- Core transfer-function utilities (`series`, `parallel`, `feedback`, frequency response).
- Parameter dataclasses for speaker, box, sensor, controller, and amplifier/injection settings.
- Physical components (speaker plant, accelerometer, PID controller, acoustic radiator).
- Full loop/system assembly for open-loop, closed-loop, and closed-loop acoustic response.
- Analysis helpers for frequency, stability, and time-domain simulation.
- Study scripts to compare free-air vs sealed-box behavior.
- Unit tests.

## Key transfer functions

### Amplifier
The amplifier is modeled as a flat gain block:

\[
G_{amp}(s) = K_{amp},\qquad K_{amp} = 10^{\frac{G_{dB}}{20}}
\]

With the default `G_dB = 26.3 dB`, this gives approximately `20.6 V/V`.

### Loudspeaker plant (voltage to acceleration)

\[
G_{spk}(s)=\frac{B\!l\,s^2}{B\!l^2 s + (sL_e+R_e)\left(s^2M_{ms}+sR_{ms}+\frac{1}{C_{tot}}\right)}
\]

### Sensor
A second-order MEMS term plus output pole:

\[
G_{sens}(s)=S_0\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}\frac{\omega_a}{s+\omega_a}
\]

### Loop equations

\[
L(s)=C(s)G(s)M(s),\quad
T(s)=\frac{C(s)G(s)}{1+C(s)G(s)M(s)},\quad
S(s)=\frac{1}{1+L(s)}
\]

## Quick start

```bash
python -m unittest discover -s tests
python -m mfb.studies.compare_boxes
python -m mfb.studies.injection_loop_demo
```

## Package layout

```
mfb/
├── core/
├── params/
├── components/
├── nonlinear/
├── systems/
├── analysis/
└── studies/
```
