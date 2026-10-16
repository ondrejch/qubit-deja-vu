# qubit-deja-vu: time-correlated readout errors on IBM Heron processors

Reset-free chains of up to 66 mid-circuit readouts per qubit, run on IBM Quantum's ibm_kingston and ibm_fez on 15 and 16 October 2026 (UTC). The chains screen for measurement errors that are correlated from one readout to the next, which matter for quantum error correction. The same data bound two models in which a hidden variable persists between measurements.

This is independent work by Ondrej Chvala (UT Austin) using a personal IBM Quantum account. No lab or institution has endorsed it.

## Results so far

The full write-up is `notes/CLOUD_RUN_RESULTS.md`, and each run's machine-generated report is `data/cloudrun/<run>/report.json`.

| Finding | Value | Where |
|---|---|---|
| Shot-wide readout correlation explained by leakage | removing leak-tagged blocks takes it from 0.001 to 0.0065 per qubit down to 0.0003 ± 0.0002 | kingston follow-up; fez repeat |
| Leakage episodes follow elapsed time, not readouts | leaked fraction changes by a factor of 0.6 to 1.3 when the readout rate drops 3.8-fold; lifetimes about 10 to 120 µs (lower limits) | kingston follow-up |
| Readout memory after leakage is removed | ±2 to 9 x 10^-3, beyond five standard errors on 6 of 12 qubits; reproduced and level from 1.4 to 5.4 µs on three fez qubits | kingston follow-up; fez runs |
| Qubit-anchored persistence (Bell-Mermin completion) | `w1 w2 < 5.3e-4` at a 1.44 µs gap (five sigma) | kingston main run |
| Detector-anchored persistence | no cusp in the angle scan; `w_D < 5.6e-3` | kingston main run |

The two hidden-variable models are completions of the Donadi-Hossenfelder outcome rule constructed for this project (`theory/theorynote.pdf`), not proposals by Donadi or Hossenfelder. The limits bound those models on these processors and readouts at the stated gaps. They are not a test of superdeterminism in general. The question of hidden order in repeated measurements of one atom was tested in quantum-jump data in 1985 to 1990 and in 2004; the novelty here is methodological.

## Method in brief

Every record applies two sx pulses whose relative frame phase sets the state, then a mid-circuit readout. Several properties follow:

1. Every setting, including random relabeling of the readout, is a virtual-Z frame phase on identical pulses. Quantum mechanics with any memory then produces only low harmonics of the phase, which separates threshold-type hidden variables from conventional memory.
2. Balanced chains give fixed-angle statistics (neighbor covariance, and a frame-tracked lag-2 covariance for the qubit-anchored model). Probe chains scan the angle for the shape test.
3. In the follow-up runs, two flip pulses after every four readouts tag leaked blocks, which read "1, 1, 1" instead of alternating.

Each design and its analysis code were committed before the corresponding data existed; `PREREGISTRATION.md` lists the commits and IBM job times.

## Layout

```text
notes/CLOUD_RUN_DESIGN.md           Design of the pilot and main run
notes/CLOUD_RUN_FOLLOWUP_DESIGN.md  Leak-tagged follow-up and the ibm_fez add-ons
notes/CLOUD_RUN_RESULTS.md          Results of every run
notes/cloud_run_results.html        One-page summary with figures (open in a browser)
theory/theorynote.pdf               Theory note: the two completions, predictions, design consequences
analysis/                           Simulation, circuit builder, submission and analysis code, with self-tests
data/cloudrun/<date>_<run>/         Run records: manifest.json, one NPZ per condition, amplitude.npz, job.json, report.json, submit.log
scripts/make_cloud_results_page.py  Builds notes/cloud_run_results.html from the reports
```

In each condition file, `r` holds outcomes as int8, +1 = |0>, with shape (parameter sets, shots, qubits, records). `alpha`, `beta` and `phi` hold the phases per set, qubit and record.

## Reproducing

```sh
python -m pip install -r requirements.txt                 # numpy, scipy
python analysis/cloud_run_design.py                       # design checks (46)
python analysis/cloud_run_followup.py                     # follow-up design checks (26)
python analysis/cloud_run_analyze.py selftest             # analysis self-tests on synthetic runs (40)
python analysis/cloud_run_analyze.py analyze data/cloudrun/2026-10-15_main --output /tmp/report.json
```

The last command reproduces `data/cloudrun/2026-10-15_main/report.json`. Building circuits and running the Qiskit Aer tests needs `qiskit`, `qiskit-ibm-runtime` and `qiskit-aer`:

```sh
python analysis/cloud_run_circuits.py --aer               # offline checks on IBM fake backends, plus an Aer end-to-end test
python analysis/cloud_run_analyze.py selftest --aer
```

`analysis/cloud_run_submit.py` sends jobs to IBM Quantum only with `--submit`, and refuses any job whose estimate exceeds 75% of the instance's remaining usage.

## License

The code in `analysis/` and `scripts/` is MIT-licensed (`LICENSE`). The data, notes and theory note are CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). Please cite this repository when reusing the data.
