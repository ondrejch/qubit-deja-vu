# qubit-deja-vu: time-correlated readout errors on IBM Heron processors

![Art Deco illustration of a single qubit flipping between two orientations](notes/qubit_deja_vu_art_v3.png)

Reset-free chains of up to 72 mid-circuit readouts per qubit, run on IBM Quantum's ibm_kingston and ibm_fez on 15 and 16 October and 19 November 2026 (UTC). The chains screen for measurement errors that are correlated from one readout to the next, which matter for quantum error correction. The same data bound two models in which a hidden variable persists between measurements.

Archived on Zenodo: [doi:10.5281/zenodo.24379286](https://doi.org/10.5281/zenodo.24379286) (all versions; v1.0.0 is [10.5281/zenodo.24379287](https://doi.org/10.5281/zenodo.24379287)).

This is independent work by Ondrej Chvala (Walker Department of Mechanical Engineering, The University of Texas at Austin; ORCID [0000-0003-4614-6649](https://orcid.org/0000-0003-4614-6649)) using a personal IBM Quantum account. No lab or institution has endorsed it.

## Results so far

The full write-up is `notes/CLOUD_RUN_RESULTS.md`, and each run's machine-generated report is `data/cloudrun/<run>/report.json`.

| Finding | Value | Where |
|---|---|---|
| Shot-wide readout correlation explained by leakage | removing leak-tagged blocks takes it from 0.001 to 0.0065 per qubit down to 0.0003 ± 0.0002 | kingston follow-up; fez repeat |
| Leakage does not enter during readouts alone | leaked fraction changes by a factor of 0.6 to 1.3 when the readout rate drops 3.8-fold, between the 0.35 to 0.59 expected if readouts alone caused it and the 1.3 to 2.2 if it built up only with time (buildup model, exploratory); lifetimes about 10 to 120 µs (lower limits) | kingston follow-up; fez repeat |
| Equator covariance after leakage is removed (readouts of equal superpositions) | ±3 to 9 x 10^-3, beyond five standard errors on 6 of 12 qubits, sign varies by qubit; reproduced and level from 1.4 to 5.4 µs on three fez qubits, and repeated five weeks later | kingston follow-up; fez runs; fez E0 and E0R |
| Heralded calibration of the first readout | heralding lowers the pooled first-readout e1 by 0.16 ± 0.03 points, mostly on one qubit; consistent with an initialization population on the pooled test; first-readout calibrations drift by up to 0.9 points within hours | fez AH |
| Readout memory on eigenstate inputs (as a syndrome qubit sees them), in the same shots | below 3 x 10^-4 on all 8 fez qubits in both runs (3-sigma upper limits below 5.5 x 10^-4): the equator covariance does not reach eigenstate readouts; in-chain readouts of a fresh \|1> match the first-readout calibration to within 0.23 points; on 4 of 8 qubits nearly every misread \|1> has decayed (memoryless-instrument fit of 20 November, chosen after the data, committed before it was applied) | fez E0 and E0R |
| Qubit-anchored persistence (Bell-Mermin completion) | `w1 w2 < 5.3e-4` at a 1.44 µs gap (five sigma) | kingston main run |
| Detector-anchored persistence | no cusp in the angle scan; `w_D < 5.6e-3` | kingston main run |

**Withdrawn readings (20 November 2026).** Earlier versions of this table reported eigenstate memory of 0.1 to 0.4 points on 7 of 8 qubits and a first-readout offset of 0.1 to 0.6 points on all 8, each replicated in the pre-registered repeat E0R. The null behind them treated decay during the readout as a misread that leaves the qubit in |1>, and on simulated chains without memory it reports the same signal. Both readings are withdrawn; `notes/CLOUD_RUN_RESULTS.md` Section 13 gives the reasons and the reanalysis.

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
notes/qubit_deja_vu_art_v3.png      Futurist / Art Deco cover illustration: the flipping qubit
notes/cloud_run_results.html        One-page summary with figures (open in a browser)
theory/theorynote.pdf               Theory note: the two completions, predictions, design consequences
analysis/                           Simulation, circuit builder, submission and analysis code, with self-tests
data/cloudrun/<date>_<run>/         Run records: manifest.json, one NPZ per condition, amplitude.npz, job.json, report.json, submit.log
                                    (plus exploratory outputs: leakage_exploratory.json, diagnostics_exploratory.json,
                                    drift_exploratory.json; for the November repeat inchain_reading.json, replication.json;
                                    for the heralded calibration herald_reading.json)
scripts/make_overview_figure.py     Builds a three-panel summary figure from the run reports (output kept out of the repository)
scripts/generate_repo_art.py        Generates the Futurist / Art Deco illustrations via Gemini
scripts/make_cloud_results_page.py  Builds notes/cloud_run_results.html from the reports
scripts/make_paper_figures.py       Builds the manuscript's data figures (figures/*.pdf) from the saved run outputs, checking each plotted value
figures/                            The three data figures of the journal manuscript
```

In each condition file, `r` holds outcomes as int8, +1 = |0>, with shape (parameter sets, shots, qubits, records). `alpha`, `beta` and `phi` hold the phases per set, qubit and record.

## Reproducing

```sh
python -m pip install -r requirements.txt                 # numpy, scipy
python analysis/cloud_run_design.py                       # design checks (46)
python analysis/cloud_run_followup.py                     # follow-up design checks (26)
python analysis/cloud_run_eigenstate.py                   # eigenstate (November) design checks (99)
python analysis/cloud_run_analyze.py selftest             # analysis self-tests on synthetic runs (86)
python analysis/cloud_run_analyze.py analyze data/cloudrun/2026-10-15_main --output /tmp/report.json
```

The last command reproduces `data/cloudrun/2026-10-15_main/report.json`; the same command on any other run directory reproduces its report.

The November eigenstate runs add two steps per block, a repeat and a flip of the post-readout eigenstate. `analysis/cloud_run_eig_instrument.py` fits each qubit's outcome patterns with a memoryless readout instrument that allows decay during the readout, with memory as a free parameter; `selftest` runs its synthetic tests, `legacy` shows the bias of the withdrawn in-chain estimator (`analysis/cloud_run_eig_drift.py`, kept for the record with `analysis/cloud_run_eig_replication.py`), and `derived` writes the upper limits, residuals and implied T1:

```sh
python analysis/cloud_run_eig_instrument.py selftest
python analysis/cloud_run_eig_instrument.py analyze data/cloudrun/2026-11-19_eigrepeat --output /tmp/instrument.json
python analysis/cloud_run_eig_instrument.py derived data/cloudrun/2026-11-19_eigrepeat --output /tmp/derived.json
```

Two scripts are exploratory: their statistics were chosen after seeing the data, and their outputs are labeled that way. `analysis/cloud_run_leakage.py` fits a three-level leakage model (`--recovery` repeats its check on synthetic data, output `analysis/cloud_run_leakage_recovery.json`). `analysis/cloud_run_diagnostics.py` computes the descriptive numbers in the results note that are not in `report.json`: long runs of one outcome, flip-step failures, standard errors clustered by shot and by parameter set, and the leak-buildup model.

```sh
python analysis/cloud_run_leakage.py data/cloudrun/2026-10-15_main --output /tmp/leakage.json
python analysis/cloud_run_diagnostics.py data/cloudrun/2026-10-15_main --output /tmp/diagnostics.json
``` Building circuits and running the Qiskit Aer tests needs `qiskit`, `qiskit-ibm-runtime` and `qiskit-aer`:

```sh
python analysis/cloud_run_circuits.py --aer               # offline checks on IBM fake backends, plus an Aer end-to-end test
python analysis/cloud_run_analyze.py selftest --aer
```

`analysis/cloud_run_submit.py` sends jobs to IBM Quantum only with `--submit`, and refuses any job whose estimate exceeds 75% of the instance's remaining usage.

## License

The code in `analysis/` and `scripts/` is MIT-licensed (`LICENSE`). The data, notes and theory note are CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). Please cite the Zenodo archive (doi:10.5281/zenodo.24379286) when reusing the data; `CITATION.cff` has the details.
