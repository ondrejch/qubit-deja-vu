# Cloud run results: ibm_kingston, 15 October 2026

These are the first experimental data in the package. The records and machine-generated reports are in `data/cloudrun/2026-10-15_pilot/`, `data/cloudrun/2026-10-15_main/` and `data/cloudrun/2026-10-15_followup/`; this note summarizes `report.json` in each. Sections 1 to 5 cover the pilot and main run, Section 6 an exploratory leakage analysis, and Section 7 the leak-tagged follow-up. The design and analysis were committed before the runs (commit 85a36c6; `notes/CLOUD_RUN_DESIGN.md`). The results bound the theory note's two completions of the Donadi-Hossenfelder rule for this processor and readout at the stated gaps. They are not a test of superdeterminism in general, of Hossenfelder's 2025 model, or of any proposal by Donadi or Hossenfelder.

## 1. What ran

| Job | IBM job IDs | Contents | QPU time charged |
|---|---|---|---|
| Pilot | db8hu45ckpjs73b11df0 | P0 probe chain (64 sets x 300 shots), C2 phase check, amplitude check | 12 s |
| Main, first attempt | db8iioheukqs73aqmbc0 | all main conditions in one job; failed with IBM error 1520 before running | 2 s |
| Main | db8ivc5ckpjs73b13pn0, db8ivcdckpjs73b13png, db8ivcheukqs73aqn8p0, db8ivcpeukqs73aqn8q0, db8ive5ckpjs73b13psg | C2 and amplitude check; A0; A1; A2; B0 (two halves of 657 sets) | 6 + 65 + 64 + 63 + 190 = 388 s |

Six qubits ran in parallel: 21, 154, 11, 149, 109 and 89. They were chosen by the builder's pre-set rule from the calibration of 15 October 2026, 14:17 EDT, which listed readout errors of 0.26% to 0.32% and T1 of 137 to 366 µs. `measure_2` took 1.38 µs, so consecutive readouts were 1.44, 5.44 and 41.4 µs apart in A0, A1 and A2 (the design assumed 1.76 µs readouts and 1.82 µs gaps). Each main condition received 400 shots per parameter set: A0 438 sets, A1 251, A2 164, B0 1314, C2 32.

There are two deviations from the design. First, one job holding PUBs of very different sizes fails with IBM error 1520, so the main run became a batch of five jobs. As a result, conditions ran back to back rather than interleaved shot by shot. Second, B0 was split into two PUBs and rejoined when saved. Neither changes the planned statistics.

## 2. Calibration checks

1. **Pulse amplitude.** The fitted `sx` amplitude errors are -0.13% to +0.47%. The amplitude term subtracted from `d_lag1` is at most 5.9e-5, against standard errors of 3e-4 per qubit in A0.
2. **Phase generation.** C2 shows no harmonic above order 2 (largest `sqrt(chi^2)` 1.89, threshold 5).
3. **Readout repeatability.** For probe steps that repeat the previous state, the mean product of the two readouts is 0.977 to 0.982 on four qubits, 0.947 on qubit 149 and 0.936 on qubit 154. Calibration readout errors do not predict this spread.

## 3. Primary statistics

Table 1 gives pooled values with standard errors and five-sigma upper limits (value + 5 SE). `Q_fix` bounds `w1 w2 = s_m^2 exp(-2 gap/tau)` for the qubit-anchored completion; `D_fix` is the neighbor covariance minus the amplitude term.

| Condition | Gap | `Q_fix` | Upper limit | `D_fix` | Upper limit |
|---|---|---|---|---|---|
| A0 | 1.44 µs | -0.00009 ± 0.00012 | 5.3e-4 | 0.0044 ± 0.0001 | 5.1e-3 |
| A1 | 5.44 µs | -0.00002 ± 0.00016 | 8.0e-4 | 0.0042 ± 0.0002 | 5.0e-3 |
| A2 | 41.4 µs | 0.00032 ± 0.00043 | 2.4e-3 | 0.0029 ± 0.0004 | 4.9e-3 |

The B0 shape fits, pooled, give the following values with five-sigma upper limits in parentheses:

| Background | `w_D` | `w_Q` |
|---|---|---|
| Harmonics up to order 2 | -0.0032 ± 0.0009 (1.3e-3) | -0.0002 ± 0.0005 (2.2e-3) |
| Harmonics up to order 4 | -0.0043 ± 0.0020 (5.6e-3) | 0.0043 ± 0.0038 (2.4e-2) |

Several features are noteworthy:

1. **Qubit-anchored completion: null.** Both `Q_fix` and `w_Q` are consistent with zero at every gap and on every qubit; no `Q_fix` value per qubit exceeds 2 standard errors. The limits are `w1 w2 < 5.3e-4` at a 1.44 µs gap and `< 2.4e-3` at 41.4 µs. Equivalently, `s_m < 0.023` per readout when `tau` is much longer than the gap.
2. **Detector-anchored completion: no cusp.** The fixed-angle statistic `D_fix` exceeds five standard errors at all three gaps, which decision rule 3 assigns to conventional memory unless the shape fit shows a cusp. It shows none. A detector-anchored persistence gives `w_D >= 0`; the order-4 fit bounds it at `w_D < 5.6e-3` for a 1.44 µs gap.
3. **The order-2 background is incomplete.** The order-2 fit's `w_D` is negative by 3.5 standard errors, which no completion can produce. Some conventional effect therefore carries even harmonics above order 2. That is allowed when the readout is not diagonal in the energy levels (Proposition 2's bound of 4 for a qubit with one leakage level), so the order-4 limit is the one to quote.
4. **No shared-amplifier signal.** Covariances between qubits read in parallel are consistent with zero (largest |z| 2.0 over 15 pairs, two lags and three conditions). The exploratory covariance between consecutive executions is also consistent with zero.

## 4. Where the `D_fix` excess comes from (exploratory, not pre-registered)

1. **The excess spans the whole shot.** On most qubits the covariance at lags 2 to 8 is about as large as at lag 1. For qubit 149 in A0, for example, it is 0.0078 at lag 1 and 0.0076 to 0.0085 at lags 2 to 8. Renewal persistence gives `d_lag2 = d_lag1^2`, about 2e-5 here, so it cannot produce this. Centering by the mean at each shot index leaves the excess unchanged, so it varies from shot to shot rather than drifting across the job.
2. **At the 41.4 µs gap the lag spectrum decays.** It falls over two to three records, a lifetime of order 100 µs.
3. **Long runs of "1" are asymmetric.** In A0, shots containing at least 10 consecutive "1" readings exceed a within-shot shuffled reference on every qubit: 3.0% to 4.1% of shots against 2.8% to 3.6%. Shots with 10 consecutive "0" readings match their reference within 0.3 percentage points. A detector-anchored hidden variable would produce runs of both values equally. The asymmetry instead fits episodes in which the qubit reads "1" persistently for about 100 µs, such as leakage out of the qubit levels.
4. **Two qubits deviate at lag 1.** Qubit 11 shows a negative lag-1 covariance (-0.0044 in A0, -0.0020 at 41.4 µs) on top of a positive background. Qubit 109 shows a lag-1 excess of about 0.007 over its background. Neither is explained here.

## 5. Status

The main run is complete as designed. The leak-tagged follow-up of Section 7 then used 178 s, leaving 20 s of the 600 s rolling allowance until the 15 October charges age out on 12 November 2026.

## 6. Leakage analysis (exploratory, after the run)

This section was added on 15 October 2026 after the results above. Its tests and model were chosen after seeing the data. The code is `analysis/cloud_run_leakage.py`, its output is `data/cloudrun/2026-10-15_main/leakage_exploratory.json`, and the flip-step counts come from B0 probe steps with `|phi| < 0.15`. Several features are noteworthy:

1. **Flip steps fail in the wrong direction.** A flip step should turn the previous outcome x into -x. On every qubit, failures that read "1, 1" outnumber failures that read "0, 0", by factors of 1.2 to 3.6 (for example 2.53% against 1.05% on qubit 149). Readout errors dominated by decay of |1> would favor "0, 0".
2. **The failures persist.** After the previous five readouts all read "1", a "1, 1" flip failure occurs in 5% to 24% of cases, against 1.6% to 5.6% after a single "1". A state that reads "1" and ignores the pulses persists over several readouts. Qubits 154 and 149 have the highest rates, and also the largest shot-wide covariance and the worst repeatability.
3. **A three-level model fits the 1.44 µs gap.** The model adds a leaked level to quantum mechanics, with no hidden variable: the level reads "1", is entered during readouts and decays back to |1>. Fitted to the B0 records by maximum likelihood, it gives leak-in probabilities of 8e-5 to 1.8e-3 per readout and leaked lifetimes of 53 to 107 µs. On synthetic data it recovers all six parameters. At the 1.44 µs gap it reproduces the A0 covariance at lags 2 to 8 on every qubit, with mean differences between -0.0004 and 0.0008.
4. **The model is incomplete at longer gaps.** At 5.44 and 41.4 µs it under-predicts the lag-2 to lag-8 covariance by 0.0006 to 0.0027. The leakage therefore depends less on the number of readouts than a model of leakage caused only by readouts allows.
5. **Two qubits show lag-1 effects of opposite sign.** Beyond the model, qubit 11 shows a lag-1 covariance of -0.0080 ± 0.0004 at 1.44 µs (-0.0020 ± 0.0011 at 41.4 µs), and qubit 109 shows +0.0070 ± 0.0004 (+0.0062 ± 0.0011). Neither shows a cusp in the shape fit, so both are conventional readout memory at a fixed angle. Pooled over qubits, the lag-1 residual at 1.44 µs is -0.0002 ± 0.0001, but only because these two cancel. It is therefore not a valid limit on persistence.

A follow-up that tags leakage directly is designed in `notes/CLOUD_RUN_FOLLOWUP_DESIGN.md`.

## 7. Follow-up: leak-tagged chains (pre-registered)

The design (`notes/CLOUD_RUN_FOLLOWUP_DESIGN.md`) and its analysis were committed before the data (commit 992b41a). Four jobs ran on ibm_kingston on 15 October 2026, using the same six qubits:

| Job | IBM job ID | QPU time charged |
|---|---|---|
| C2 and amplitude check | db8kv89eukqs73aqqp9g | 6 s |
| T0 (record period 1.444 µs) | db8kv8gb3g5c73egg7sg | 65 s |
| T1 (5.444 µs) | db8kv8ob3g5c73egg7u0 | 64 s |
| T2 (41.444 µs) | db8kv91eukqs73aqqpbg | 43 s |

Amplitude terms were at most 4.7e-5, and the phase check was clean (largest `sqrt(chi^2)` 2.31). The pre-registered statistics give the following:

1. **S1: tagging removes the shot-wide background.** The pooled clean cross-block covariance is 0.0003 ± 0.0002 at both 1.44 and 5.44 µs. Without selection, the same statistic is 0.0011 to 0.0065 per qubit at 1.44 µs. At 41.4 µs, consecutive blocks are 249 µs apart and carry no background even without selection, so that gap does not test S1. By decision rule 1, the background is attributed to leakage as tagged.
2. **S2: clean neighbor covariance per qubit.**

   | Physical qubit | 1.44 µs | 5.44 µs | 41.4 µs |
   |---|---|---|---|
   | 21 | +0.0007 ± 0.0004 (limit 2.8e-3) | +0.0014 ± 0.0006 (4.2e-3) | -0.0010 ± 0.0019 (8.3e-3) |
   | 154 | -0.0027 ± 0.0004 (flagged) | -0.0012 ± 0.0006 (1.8e-3) | -0.0025 ± 0.0019 (7.0e-3) |
   | 11 | -0.0086 ± 0.0004 (flagged) | -0.0075 ± 0.0006 (flagged) | -0.0030 ± 0.0019 (6.3e-3) |
   | 149 | -0.0017 ± 0.0004 (5.0e-4) | -0.0013 ± 0.0006 (1.6e-3) | +0.0018 ± 0.0020 (1.2e-2) |
   | 109 | +0.0060 ± 0.0004 (flagged) | +0.0065 ± 0.0006 (flagged) | +0.0125 ± 0.0020 (flagged) |
   | 89 | -0.0001 ± 0.0004 (2.1e-3) | +0.0007 ± 0.0006 (3.6e-3) | +0.0036 ± 0.0019 (1.3e-2) |

   Values in parentheses are five-sigma upper limits for qubits within five standard errors of zero. As expected, qubits 11 and 109 carry readout memory of opposite signs. Qubit 154 also turns negative once its leakage is removed, and qubit 149 is negative at 3.9 standard errors.
3. **S3: no structure within blocks.** Clean lag-2 and lag-3 covariances are within 0.0015 of zero at 1.44 µs on every qubit.
4. **S4: leakage follows elapsed time, not readouts.** When the readout rate falls by a factor of 3.8 from T0 to T1, the leaked-check fraction changes only by factors of 0.59 to 1.28. It is 0.13% to 1.22% of checks. Persistence after two leaked checks gives leak lifetimes of 9 to 75 µs at T0 and 23 to 124 µs at T1. These are lower limits, because false flags and imperfect detection of the leaked level shorten the apparent persistence. At 41.4 µs, T1 decay during the 40 µs delays corrupts 11% to 19% of the checks, so T2's leak flags are not interpretable. The design did not anticipate this.

What follows for the completions:

1. **Detector-anchored completion.** On qubits 21 and 89, the clean neighbor covariance at 1.44 µs is consistent with zero, and five-sigma limits of `w_D < 2.8e-3` and `< 2.1e-3` follow for those readouts. At 5.44 µs, qubits 21, 154, 149 and 89 give limits of 1.6e-3 to 4.2e-3. These fixed-angle limits assume that conventional lag-1 contributions on a qubit are not negative. Qubits 11, 154 and 149 show that negative contributions occur on this device, so the shape-fit limit of Section 3 (`w_D < 5.6e-3`, pooled) remains the one that needs no such assumption.
2. **Qubit-anchored completion.** The follow-up was not designed to test it, and the main-run limits stand.
3. **Against the design's expectations.** S1 vanished at 1.44 µs (judged about 60% likely) and at 5.44 µs (about 40%). S2 was nonzero on qubits 11 and 109 (about 85%). Leakage turned out to follow elapsed time, against an even chance of following readouts.


## 8. Add-on: one leak-tagged condition on ibm_fez (pre-registered)

The design is Section 9 of `notes/CLOUD_RUN_FOLLOWUP_DESIGN.md`, committed before the data (commit d501525). Job db8pactckpjs73b1f6m0 ran on ibm_fez on 15 October 2026 and was charged 15 s.

- **Qubits:** 124, 26, 86, 144, 2 and 6, chosen by the builder's rule from the 22:46 EDT calibration.
- **Condition:** F0, the T0 chain with a 1.39 µs record period, 88 sets x 400 shots, plus the amplitude check.
- **Amplitude terms:** at most 7.6e-5.

Several features are noteworthy:

1. **S1: tagging removes the shot-wide background here too.** Without selection, the cross-block covariance is 0.0016 to 0.0031 on qubits 124, 26, 144 and 6, and near zero on 86 and 2. After removing leak-tagged blocks, the pooled value is -0.0003 ± 0.0004.
2. **S4: leakage is milder than on ibm_kingston.** Leaked checks are 0.12% to 0.65% of checks, against 0.15% to 1.22% on kingston at the same spacing. Lifetimes from persistence are 15 to 41 µs, against 9 to 75 µs on kingston (lower limits in both cases).
3. **S2: the clean neighbor covariance is positive on three qubits.**

   | Physical qubit | Clean neighbor covariance | Five-sigma upper limit |
   |---|---|---|
   | 124 | +0.0036 ± 0.0010 | 8.4e-3 |
   | 26 | +0.0001 ± 0.0010 | 4.9e-3 |
   | 86 | -0.0008 ± 0.0010 | 4.0e-3 |
   | 144 | +0.0038 ± 0.0010 | 8.6e-3 |
   | 2 | -0.0017 ± 0.0009 | 3.0e-3 |
   | 6 | +0.0051 ± 0.0010 (flagged) | |

   The pooled value is +0.0017 ± 0.0004. Only qubit 6 is beyond five standard errors; qubits 124 and 144 are at 3.8 and 4.0. At a fixed angle these positive values are readout memory or detector-anchored persistence, and only a shape scan on these qubits would separate them. On ibm_kingston, positive memory appeared on one qubit (109), alongside negative memory on three others.
4. **Allowance.** The rolling allowance now stands at 595 s consumed and 5 s remaining.

## 9. Paid ibm_fez repeat (pre-registered)

### 9.1 First attempt: cancelled, no data

The repeat of Section 10 of the follow-up design ran as job db8plalckpjs73b1fung on the author's pay-as-you-go instance. IBM cancelled it at the end of execution after charging 58 s (about $93, charge status still "pending" on 16 October), and no results can be retrieved. The likely cause is the instance's 63 s usage limit: IBM cancels running jobs when a pay-as-you-go instance reaches its cost limit. The records of the attempt are in `data/cloudrun/2026-10-15_fezrepeat/`.

### 9.2 Rerun

The author raised the instance's cost limit to $250. The same job then ran as db8pnu8b3g5c73egp3l0 on ibm_fez, on the F0 qubits, and was charged 58 s (about $93). The records are in `data/cloudrun/2026-10-15_fezrepeat_rerun/`.

| Physical qubit | F0, 1.39 µs | G0, 1.39 µs | G1, 5.39 µs |
|---|---|---|---|
| 124 | +0.0036 ± 0.0010 | +0.0041 ± 0.0006 (flagged) | +0.0056 ± 0.0009 (flagged) |
| 26 | +0.0001 ± 0.0010 | -0.0007 ± 0.0006 | +0.0015 ± 0.0009 |
| 86 | -0.0008 ± 0.0010 | -0.0009 ± 0.0006 | +0.0005 ± 0.0009 |
| 144 | +0.0038 ± 0.0010 | +0.0055 ± 0.0006 (flagged) | +0.0042 ± 0.0009 |
| 2 | -0.0017 ± 0.0009 | -0.0029 ± 0.0006 | -0.0023 ± 0.0009 |
| 6 | +0.0051 ± 0.0010 (flagged) | +0.0052 ± 0.0006 (flagged) | +0.0058 ± 0.0009 (flagged) |

All values are clean neighbor covariances. The pre-registered readings give the following:

1. **Reproducible.** On every qubit, G0 agrees with F0 within 1.4 combined standard errors. The positive covariance on qubits 124, 144 and 6 is confirmed at 6.5, 8.5 and 8.2 standard errors. Qubit 2 is reproducibly negative, at -4.6 standard errors in G0, so its readout memory is conventional.
2. **Level with spacing.** From G0 to G1, no qubit changes by more than 2.0 combined standard errors in either direction (qubit 124: +1.4; 144: -1.2; 6: +0.6). The rule fixed in advance therefore excludes the detector-anchored completion on none of them. A level value fits both conventional readout memory and a persistence time much longer than 5 µs. Only a shape scan on these qubits can separate the two.
3. **S1 and S4 hold.** The clean cross-block covariance is consistent with zero at both spacings (-0.0003 ± 0.0002 and +0.0003 ± 0.0003). Leaked checks are 0.18% to 0.74% at both spacings, so on ibm_fez too leakage follows elapsed time rather than readouts.

