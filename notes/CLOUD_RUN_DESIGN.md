# Cloud run design: hidden-variable persistence on IBM Heron

Draft, 13 October 2026. The run uses the author's own IBM Quantum open-plan access; no lab is involved and no endorsement is implied. The design below was fixed before any data existed (commit 85a36c6). The pilot (job db8hu45ckpjs73b11df0) and the main job (job db8iioheukqs73aqmbc0) ran on ibm_kingston on 15 October 2026. Their records and reports are in `data/cloudrun/`, and `notes/CLOUD_RUN_RESULTS.md` summarizes them, including two deviations from this plan. The simulated tables in this note are design checks, not data.

Companion files:
- `analysis/cloud_run_design.py`: NumPy simulation of the planned chains, their statistics, sensitivity and budget; writes `analysis/cloud_run_design.json`.
- `analysis/cloud_run_circuits.py`: Qiskit circuit builder, with offline checks against IBM fake backends and an end-to-end test on Qiskit Aer.
- `analysis/cloud_run_analyze.py`: saves a job's records and computes the statistics fixed in Section 7, with NumPy and Aer self-tests.
- `analysis/cloud_run_submit.py`: builds a job against the live backend and prints its estimate; with `--submit` it sends the job and saves the records and report.

## 1. Purpose and scope

The run tests the two sequential completions of the Donadi-Hossenfelder outcome rule in the draft theory note (`theory/theorynote.pdf`): the detector-anchored completion (D) and the qubit-anchored completion (Q, the Bell-Mermin model), each with renewal persistence `w = s_m exp(-gap/tau)` between consecutive readouts. These completions and the persistence law are constructions made in this repository, not proposals of Donadi or Hossenfelder. The run tests neither Hossenfelder's 2025 collapse model nor superdeterminism in general; the Bell-scenario version is open (theory note Section 8).

The run can deliver three results:

1. Upper limits on D-type persistence for a superconducting readout chain at readout-to-readout gaps of 1.82, 5.82 and 41.8 µs, with a statistical reach of 6.4e-4 to 2.1e-3 at five standard deviations.
2. Upper limits on Q-type persistence across two readouts, `w1 w2`, at the same gaps and a similar reach.
3. A shape test that identifies either completion at `w >= 0.0047` (D) or `w >= 0.0025` (Q) at five standard deviations, after removing every harmonic that quantum mechanics with memory can produce for a qubit with one leakage level.

At a fixed angle, D-type persistence and ordinary readout memory move the same statistic, so a fixed-angle D result is informative only as an upper limit. The shape test separates them.

## 2. A correction to the theory note found during the design

Completion Q rotates its hidden vector `lambda` with every pulse applied to the qubit (theory note Section 3). The note's repeated-preparation and cusp predictions for Q nevertheless treat re-preparation as setting the qubit state while leaving `lambda` unrotated; in `analysis/sequential_dh_model.py` the `prep` operation changes the state only. Under the note's own rule, a re-preparation built from pulses rotates `lambda`, and the Q predictions change:

1. Let `V` be the net rotation between two readouts. The second readout reads `lambda` along `n' = V^T z`. A balanced second readout needs `n'` perpendicular to `z`, and then its outcome is uncorrelated with the first for isotropic `lambda`. Balanced repeated preparation therefore gives zero for Q whenever re-preparation is made of pulses.
2. IBM's reset is a readout followed by a conditional X, so it also leaves the correlation at zero. The frozen simulation gives -0.0008 without and -0.0027 with an unwinding pulse before the reset, each ± 0.0016.
3. The note's `E(x1 x2) = w` for Q holds only for a re-preparation that leaves `lambda` unrotated: an unwinding pulse followed by a reset that applies no pulse (optical pumping, if the model lets pumping leave `lambda` alone). The simulation of this case gives 1 in the frozen limit. That is a further model choice.
4. Without a reset, re-preparation by pulses is the same as reading along a rotated axis. Q then shows two signatures. One is the orthogonal-chain correlation `w1 w2` between readouts that share an axis, which is the note's triad endpoint (Section 5.2). The other is the first-order pair deviation `q(theta) = E_Q(theta) - cos(theta)`, which has kinks at `theta = pi/4` and `3pi/4` where the note's Q cusp would have been.
5. Completion D is unaffected, because its hidden variable belongs to the detector.

The passages corrected in theory note v0.2 (13 October 2026) are:
- in the theory note: Summary items 1 to 3; Section 4, item 3; the Q entry of Eq. (repeat) in Section 5.1; the Q statements in Section 5.4; the Q cusp in Section 5.5; Section 5.6, item 1; and Section 7, items 1, 2 and 5;

The design below uses only the corrected predictions, and the simulation in `analysis/cloud_run_design.py` checks each of them.

## 3. Platform facts (checked 13 October 2026)

Labels: "docs" means confirmed in IBM documentation; "snapshot" means IBM's fake-backend calibration snapshots (April 2026) shipped with qiskit-ibm-runtime 0.50.0.

| Item | Value | Source |
|---|---|---|
| Open-plan allowance | 10 min of QPU time per rolling 28 days (the pricing page says per month). A one-time opt-in grant of 180 min over 12 months opens after 20 logged minutes in a 12-month period | docs |
| Open-plan backends | ibm_fez, ibm_marrakesh, ibm_kingston (Heron r2); ibm_marrakesh showed as paused on 13 October | docs |
| Primitive | SamplerV2 is deprecated in qiskit-ibm-runtime 0.50.0 (24 September 2026). The replacement is `qiskit_ibm_runtime.executor_sampler.Sampler`, with the same PUB format and one shot count per job | docs |
| Job size | 10 million executions per job; no separate cap on shots or parameter sets | docs |
| Repetition delay | Default 250 µs. The February 2026 changelog gives 20 to 2000 µs on Heron; snapshots list 0 to 2000 µs | docs, snapshot |
| Readout | `measure` takes 1.56 to 2.58 µs. `measure_2` (`MidCircuitMeasure`) takes 1.29 to 1.76 µs where supported; snapshots list it for ibm_kingston but not for ibm_fez or ibm_marrakesh | docs, snapshot |
| Pulses and errors | `sx` lasts 24 to 36 ns with error near 2e-4. Readout assignment error is 0.9% to 1.5% (live medians, kingston, fez, marrakesh) | snapshot, docs |
| Compilation | `rz` is virtual (a frame update of zero duration); circuits must be ISA; only `rz` may carry parameters in dynamic circuits | docs |
| Sampler defaults | Dynamical decoupling, gate twirling and measurement twirling are off | docs |
| Billing | Usage is about 2 s per sub-job plus (repetition delay + circuit duration) times executions. A `dry_run` estimate exists, but access to it may be restricted | docs |
| Shot order | The BitArray's leading axis runs over ordered shots. Execution is column-wise: each shot index runs through all circuits before the next. There are no per-shot timestamps, and ordering across sub-jobs is not stated | docs |
| Pulse control | Removed on 3 February 2025; readout pulses cannot be changed | docs |
| Not documented | Maximum circuit duration; maximum number of mid-circuit readouts per shot (errors 8020 and 8045 exist without stated limits); outcome-dependent readout errors or residual photons; leakage per readout | |

## 4. Record design

### 4.1 The step

Every record on every qubit is

```text
[delay]  rz(beta_(k-1) - alpha_k)  sx  rz(alpha_k - beta_k)  sx  barrier  measure_2
```

with `beta_(-1) = 0`. It applies `step(alpha, beta) = P(beta) P(alpha)`, where `P(g)` is a pi/2 rotation about the equatorial axis at azimuth `g`, and then reads out. Several properties follow:

1. From a post-readout eigenstate `b = +1` (|0>) or `b = -1`, the step leaves `<Z> = -b cos(phi)` with `phi = beta - alpha`. `phi = pi/2` gives a balanced state; `phi = 3pi/2` gives the same state with the readout relabeled (logical outcome = minus the physical outcome); other values are probes.
2. Every setting is a frame phase. All records use the same two physical pulses, so `phi` enters only as a frame phase, and theory note Proposition 2 applies: quantum mechanics with any memory gives probabilities of harmonic degree at most 2 in `phi` for a qubit with one leakage level and a readout diagonal in the energy levels, and at most 4 otherwise.
3. The chain has no reset. A reset is a readout plus a conditional pulse. For Q it rotates `lambda` conditionally and removes the signal (Section 2), and it adds a readout whose outcome is not recorded.
4. The trailing frame phase `beta_k` is carried into the next record's first `rz`. No `rz` therefore sits directly before a readout, where a compiler could drop it and silently shift every later phase.
5. The relabel is a frame phase, not an X pulse. An X pulse with fractional amplitude error `eps` adds a first-order term of size `pi eps` unless its own phase is also randomized. The frame relabel leaves only the frame-invariant term `cos^2((pi/2)(1 + eps))`, about `(pi eps/2)^2`, which the amplitude check measures.

### 4.2 Chains and checks

1. **Balanced chain (conditions A0, A1, A2).** Steps alternate `S = step(a, a + pi/2)` and `S^-1 = step(a + 3pi/2, a + pi)`. Each parameter set has one random frame offset `a`, and every step has an independent random relabel. In the frame that co-rotates with the qubit, the readout axes of records `k - 1` and `k + 1` coincide up to a sign `t = +1` or `-1`, which the analysis computes from the phases. The simulation confirms that `t` takes only these two values.
2. **Probe chain (B0).** Balanced and probe steps alternate (B, P, B, P, ...). Every step has a random frame offset, and probe steps draw `phi` uniformly on [0, 2pi).
3. **Amplitude check (C).** One readout after m = 1, 9, 25, 49 or 97 `sx` pulses. Because m = 1 (mod 4), the outcome is linear in the amplitude error, with gain m.
4. **Phase check (C2).** One probe step from the initial |0> and one readout. Both completions keep this first marginal at its Born value `-cos(phi)`, so any harmonics beyond the first measure imperfections in phase generation (theory note Section 5.5, item 4).

### 4.3 Qubits and timing

Six non-adjacent qubits run independent chains in parallel. The builder picks those with the lowest readout error that support `measure_2`; on the ibm_kingston snapshot it picks qubits 50, 148, 21, 68, 117 and 142. The selection must be redone from live calibration data on the day of the run. With `measure_2` at 1.76 µs and `sx` at 32 ns, consecutive readouts are 1.82 µs plus the added delay apart.

## 5. What each statistic responds to

The planned statistics are covariances, so readout bias cancels:

1. `d_lag1`: physical-label covariance of neighboring records in the balanced chain.
2. `d_lag2`: the same at lag 2.
3. `q_lag2`: lag-2 covariance after multiplying by `t`.
4. Shape fit: the probe chain's (B, P) and (P, B) pairs, fitted jointly by weighted least squares with three components:
   - the D cusp `1 - |cos(phi)|`, shared by both pair types;
   - the Q template `q(theta)` with `cos(theta) = -cos(phi)`, in (B, P) pairs only;
   - every harmonic of `phi` up to order 2 for each pair type (order 4 as a robustness check), as the quantum-allowed background.

Table 1 gives simulated values for 300,000 shots of 8 records each; the standard error is 0.0007.

| Simulated case | `d_lag1` | `d_lag2` | `q_lag2` |
|---|---|---|---|
| Quantum mechanics, ideal | 0.0003 | 0.0004 | -0.0002 |
| D, w = 0.5 | 0.5004 | 0.2500 | 0.0007 |
| Q, w = 0.5 | -0.0008 | -0.0017 | 0.2513 |
| Quantum mechanics, pulse amplitude error 6% | 0.0076 | 0.0009 | -0.0002 |
| Quantum mechanics, T1 decay 10% per gap | 0.0011 | 0.0010 | -0.0006 |
| Quantum mechanics, assignment error 1% and 3% | -0.0018 | 0.0000 | -0.0006 |
| Quantum mechanics, readout memory 2% | 0.0101 | 0.0007 | -0.0001 |
| Quantum mechanics, all four effects | 0.0178 | 0.0003 | 0.0004 |

Two entries sit at 2.4 and 2.6 standard errors: `d_lag2` for Q and `d_lag1` with assignment error. Both fell to 0.4 and 0.6 standard errors in a run with five times as many shots. Several features are noteworthy:

1. `q_lag2` responds only to Q, at `w1 w2`. No simulated conventional effect moves it. Quantum mechanics with a memoryless readout makes the chain Markov, and `t` changes sign at random from record to record.
2. `d_lag1` responds to D at `w`, and equally to readout memory and to pulse amplitude error. The amplitude term `cos^2((pi/2)(1 + eps))` is 0.0088 at `eps` = 6% and 6e-5 at `eps` = 0.5%. At a fixed angle these effects cannot be separated, and check C measures the amplitude term.
3. The lag structure discriminates. Renewal persistence gives `d_lag2 = d_lag1^2`; readout memory that decays within one gap gives `d_lag2` near zero; slow drift gives `d_lag2` near `d_lag1`.

Table 2 gives shape fits on simulated probe chains of 400,000 shots of 8 records each:

| Simulated case | fitted `w_D` | fitted `w_Q` |
|---|---|---|
| Quantum mechanics, ideal | 0.0021 ± 0.0071 | 0.0000 ± 0.0033 |
| Quantum mechanics, all four effects | -0.0004 ± 0.0080 | 0.0012 ± 0.0044 |
| D, w = 0.3 | 0.2988 ± 0.0070 | -0.0027 ± 0.0033 |
| Q, w = 0.3 | -0.0048 ± 0.0072 | 0.3036 ± 0.0032 |

Each template is recovered without leaking into the other. The conventional effects stay within the quantum-allowed harmonics, so they leave both estimates consistent with zero.

## 6. Sensitivity and budget

For N probe records with readout contrast 0.95, the shape fits give `sigma_w sqrt(N)` = 9.2 for the D cusp and 4.9 for the Q template with order-2 background. With order-4 background the values are 20.1 and 39.9. The order-4 background absorbs most of the Q template, whose largest component beyond the first harmonic is the third. The probe chain improves on the theory note's estimate for the D cusp (14.7, that is 0.073 at 10^6 records, which the script reproduces), for two reasons: (B, P) pairs have variance `1 - cos^2(phi)`, which is small near the poles, and (P, B) pairs see the cusp a second time.

Table 3 lists the plan. Timing uses ibm_kingston snapshot values: repetition delay 250 µs, `measure_2` 1.76 µs, `sx` 32 ns. Main-job PUBs share one shot count, as the replacement Sampler requires.

| Job | Condition | Records per shot | Added delay | Gap | Sets x shots | QPU time | Five-sigma reach |
|---|---|---|---|---|---|---|---|
| 1 | P0 probe chain (full size) | 64 | 0 | 1.82 µs | 64 x 300 | 7.0 s | checks size limits and data format |
| 1 | C amplitude, C2 phase | 1 | 0 | | 5 circuits and 32 sets, x 300 | 2.9 s | |
| 2 | A0 balanced | 64 | 0 | 1.82 µs | 409 x 400 | 60 s | `w_D` 6.4e-4; `w1 w2` (Q) 6.4e-4 |
| 2 | A1 balanced | 64 | 4 µs | 5.82 µs | 241 x 400 | 60 s | 8.3e-4; 8.4e-4 |
| 2 | A2 balanced | 16 | 40 µs | 41.8 µs | 163 x 400 | 60 s | 2.1e-3; 2.1e-3 |
| 2 | B0 probe chain | 64 | 0 | 1.82 µs | 1227 x 400 | 180 s | identifies `w_D >= 0.0047` (0.010 at order 4) and `w_Q >= 0.0025` (0.021) |
| 2 | C amplitude, C2 phase | 1 | 0 | | 5 circuits and 32 sets, x 400 | 3.7 s | |
| | Total with 2 s per job | | | | | 378 s | of 600 s per 28 days |

The fixed-angle reaches are statistical: an upper limit is the measured value plus five standard errors. Executions total about 0.83 million per job, under the 10 million cap. The readout data come to about 3e8 bits, or 37 MB. B0 binds 1227 sets of 768 angles, about 0.94 million numbers; if the service rejects that payload, split B0 into two or three PUBs. The 222 s left in the allowance cover one repeat of a condition or a pilot on a second backend.

## 7. Planned analysis (fixed before the run)

Commit this document and the three scripts before the first submission. The commit fixes the analysis in advance.

The primary statistics are computed per qubit and pooled over qubits:

1. `D_fix(gap) = d_lag1 - A`, where A is the amplitude term from check C. Its five-sigma upper limit bounds `s_m exp(-gap/tau)` for D at that gap, provided conventional contributions are not negative. A negative contribution could hide a positive `w`; theory note Section 6 states the same caveat for random-number generators.
2. `Q_fix(gap) = q_lag2`. Its five-sigma upper limit bounds `w1 w2 = s_m^2 exp(-2 gap/tau)` for Q.
3. Shape (B0): fitted `w_D` and `w_Q` with order-2 background, and an order-4 fit as a robustness check.

The diagnostics are:

1. `d_lag2` and the full lag spectrum (Section 5, item 3).
2. Covariances between qubits read in parallel. These test a version of D in which the hidden variable sits in a shared amplifier, and they measure readout crosstalk.
3. The covariance between the last record of one execution and the first record of the next, a gap of about 250 µs plus the initializing reset. This is exploratory, because ordering across sub-jobs is not documented.
4. C2 outcomes against `phi`: harmonic content of phase generation.
5. Readout characterization from probe records near `phi = pi`, where the step repeats the previous state (repeatability and assignment), and near `phi = 0`, where it flips it.

Decision rules:

1. All primary statistics consistent with zero: report the limits per gap and per qubit. The run is then complete.
2. `Q_fix` or `w_Q` above five standard deviations: check that the excess appears on every qubit and in both statistics. Repeat on a second backend (ibm_fez, with the standard readout) before describing it to anyone. A result that persists would be a candidate anomaly needing a second platform.
3. `D_fix` above five standard deviations at the 1.82 µs gap: this is expected from readout memory, so compare gaps and lags (Section 5, item 3). Only a cusp in the shape fit that survives both the order-2 and order-4 backgrounds would point beyond conventional readout memory.

## 8. Risks and mitigations

1. **Size limits.** 384 mid-circuit readouts per shot may exceed an undocumented buffer (errors 8020 and 8045). The pilot runs the full-size circuit for 7 s of QPU time. If it fails, fall back to 32 records per shot.
2. **`measure_2` availability.** If `measure_2` is absent on the day, use `measure` (1.56 µs on the ibm_fez snapshot). The builder switches automatically and checks the fallback offline. All records in a run use the same readout instruction.
3. **Primitive migration.** Use `executor_sampler.Sampler`, with one shot count per job.
4. **Server-side compilation.** It cannot be inspected. The builder checks the transpiled circuits offline: gate counts, every `rz` kept, no added operations, parameters preserved and bindable. The carried frame phase removes the most likely silent change. The Aer test confirms bit order, parameter order and signs: probe pairs follow `-cos(phi)` with slope 0.998 ± 0.003. Qiskit binds parameter arrays in sorted-name order, and the builder permutes its columns to match.
5. **Readout memory at short gaps** (residual resonator photons). Expected to dominate `d_lag1` at 1.82 µs; the delay scan and the shape fit address it.
6. **Leakage.** A leaked qubit reads as a fixed outcome for several records, which moves `d_lag1` and `d_lag2` at a fixed angle. The shape fit's background covers it for a qubit with one leakage level.
7. **Crosstalk between parallel qubits.** The chosen qubits are non-adjacent, and cross-qubit covariances are reported. Spectator decoherence during mid-circuit readout has been reported on ibm_marrakesh (arXiv:2504.07258).
8. **Drift.** Column-wise execution interleaves the conditions of the main job shot by shot, and every covariance is taken within shots.
9. **Model scope.** Limits apply to these completions and this persistence law. Whether the parent model's authors would accept the completions is open (theory note Section 8).

## 9. Expected outcomes

These are the author's judgment, not computed probabilities:

1. Usable data from both jobs: about 90%. The main risk is the size limit, which the pilot settles.
2. `q_lag2` and `w_Q` consistent with zero at the planned reach: about 80%. No conventional mechanism identified here moves them.
3. `d_lag1` consistent with zero at 1.82 µs: about 30%, because readout memory at that gap is plausibly at the 1e-3 level. At 41.8 µs, about 70%.
4. A cusp in `w_D`: not expected. Its main role is to settle where any `d_lag1` excess comes from.

## 10. Building and running

The design checks need only NumPy. The builder needs Qiskit, which is kept outside the repository:

```sh
python3 analysis/cloud_run_design.py --output analysis/cloud_run_design.json
python3 -m venv ~/.venvs/qiskit
~/.venvs/qiskit/bin/pip install qiskit qiskit-ibm-runtime qiskit-aer
~/.venvs/qiskit/bin/python analysis/cloud_run_circuits.py --aer      # main job, offline checks and Aer test
~/.venvs/qiskit/bin/python analysis/cloud_run_circuits.py --pilot --shots 300
```

Submission needs the author's go-ahead for each job. `analysis/cloud_run_submit.py` does it and saves the records in `data/cloudrun/<date>_<job>/`; without `--submit` it only builds and estimates:

```sh
~/.venvs/qiskit/bin/python analysis/cloud_run_submit.py --job pilot                       # estimate only
~/.venvs/qiskit/bin/python analysis/cloud_run_submit.py --job main --submit --out data/cloudrun/2026-10-15_main
```

The same steps in Python:

```python
import sys
sys.path.insert(0, "analysis")
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler
from cloud_run_circuits import build_all
from cloud_run_analyze import save_job

backend = QiskitRuntimeService().backend("ibm_kingston")
built = build_all(backend, n_qubits=6, shots=300, pilot=True)   # qubits from live calibration
job = Sampler(mode=backend).run(built["pubs"], shots=300)
save_job("data/cloudrun/2026-10-15_pilot", built, job.result(), job)
```

The main job is the same with `shots=400` and without `pilot=True`. Every saved run is analyzed the same way, with NumPy only:

```sh
python3 analysis/cloud_run_analyze.py analyze data/cloudrun/2026-10-15_pilot --output data/cloudrun/2026-10-15_pilot/report.json
```

`save_job()` writes a manifest, one file per chain condition (outcomes as int8 arrays of shape sets x shots x qubits x records, with their phases) and the amplitude check. `analyze()` computes the Section 7 statistics per qubit and pooled. These are `D_fix` with the amplitude term from check C subtracted, `Q_fix`, both shape fits at orders 2 and 4, the phase-check harmonics, readout characterization, cross-qubit and cross-execution covariances, and a drift-centered `d_lag1`. It then applies the decision rules. It groups records by parameter set, so a full main job fits in memory.

The analysis has two self-tests:

```sh
python3 analysis/cloud_run_analyze.py selftest                     # NumPy: 34 checks
~/.venvs/qiskit/bin/python analysis/cloud_run_analyze.py selftest --aer   # adds Aer: 44 checks
```

1. **NumPy test.** The synthetic runs inject both completions, pulse amplitude error, readout error and readout memory. The test checks that `analyze()` recovers each effect and fires the right decision rule, and that the grouped estimators equal the record-level estimators of `analysis/cloud_run_design.py`.
2. **Aer test.** The builder's circuits run on Qiskit Aer with a 6% coherent `sx` over-rotation and 1%/3% readout error, and their Qiskit results go through `save_job()`. The amplitude check recovers `eps = 0.06005 ± 0.00015` and contrast 0.961 (true 0.96). The raw `d_lag1` of 0.0092 ± 0.0030 falls to `D_fix` = 0.0001 ± 0.0021 once the measured amplitude term is subtracted.

## Sources

- Plans and allowance: https://quantum.cloud.ibm.com/docs/en/guides/plans-overview, https://www.ibm.com/quantum/pricing, https://www.ibm.com/quantum/blog/open-plan-updates
- Backends and changes: https://quantum.cloud.ibm.com/docs/en/guides/changelog-quantum-compute-service
- Primitive deprecation: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes, https://quantum.cloud.ibm.com/docs/en/guides/migrate-to-client-side-primitives
- Limits and errors: https://quantum.cloud.ibm.com/docs/en/guides/job-limits, https://quantum.cloud.ibm.com/docs/en/errors
- Repetition delay: https://quantum.cloud.ibm.com/docs/en/guides/repetition-rate-execution
- Mid-circuit readout: https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits
- Billing: https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time
- Shot order: https://quantum.cloud.ibm.com/docs/en/guides/sampler-input-output
- Twirling and decoupling defaults: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-twirling-options
- ISA circuits: https://quantum.cloud.ibm.com/docs/en/guides/transpile
- Calibration snapshots: https://github.com/Qiskit/qiskit-ibm-runtime/tree/main/qiskit_ibm_runtime/fake_provider/backends
- Spectator decoherence during mid-circuit readout: https://arxiv.org/abs/2504.07258
