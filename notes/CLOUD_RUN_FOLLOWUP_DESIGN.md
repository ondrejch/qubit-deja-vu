# Follow-up design: leak-tagged chains on ibm_kingston

Draft, 15 October 2026. This design and its analysis code were committed before any follow-up data existed (commit 992b41a). The follow-up ran on ibm_kingston on 15 October 2026. Its records are in `data/cloudrun/2026-10-15_followup/`, and its results are in Section 7 of `notes/CLOUD_RUN_RESULTS.md`. Companion code:
- `analysis/cloud_run_followup.py`: chain settings, a three-level simulation, the tagged statistics and design checks; writes `analysis/cloud_run_followup.json`.
- `analysis/cloud_run_circuits.py`, `analysis/cloud_run_analyze.py` and `analysis/cloud_run_submit.py`: the `tagged` chain kind, its analysis and `--job followup`.

## 1. Why a follow-up

The 15 October 2026 run (`notes/CLOUD_RUN_RESULTS.md`) bounded the qubit-anchored completion and found no cusp for the detector-anchored one. The fixed-angle neighbor covariance, however, exceeded five standard errors at every gap. The exploratory analysis in Section 6 of the results note points to two conventional causes:

1. **Leakage.** At flip steps, "1, 1" failures outnumber "0, 0" failures on every qubit, and after five consecutive "1" readings they rise three- to sevenfold. A three-level model fitted to the probe scan reproduces the shot-wide covariance at the 1.44 µs gap. It under-predicts it at 5.44 and 41.4 µs.
2. **Readout memory on two qubits.** Qubits 11 and 109 show lag-1 covariances of about -0.008 and +0.007 beyond the leakage model, persisting to 41.4 µs, with no cusp in the shape fit.

## 2. Questions

1. **Is the shot-wide background leakage?** If it is, the covariance between clean blocks vanishes at every gap once leaked blocks are removed.
2. **Is leakage caused by readouts or by elapsed time?** The answer comes from the leaked-check fraction and its persistence at record periods of 1.44, 5.44 and 41.4 µs.
3. **What remains at lag 1 after leakage is removed?** Per qubit, this either bounds detector-anchored persistence at a fixed angle or confirms readout memory, which a fixed-angle test cannot distinguish from it.

## 3. Chain

Records come in blocks of six: `B B B B F F`.

- **B steps:** balanced (`beta - alpha = pi/2`, or `3 pi/2` to relabel). The four alternate `S` and `S^-1`, with one random frame offset per block and a random relabel per step.
- **F steps:** flips (`beta = alpha`, a pi rotation, random frame offset).

After the fourth B readout x, the flips give -x and then x. A leaked qubit, which reads "1" and is not moved by the pulses, gives "1, 1, 1" instead. Each block's check takes one of three states:

| Check state | Pattern |
|---|---|
| Normal | x, -x, x |
| Leaked | 1, 1, 1 |
| Other | anything else |

A block's B records are **clean** when its own check and the previous block's check are both normal.

Several properties follow, each verified by simulation (Section 5):

1. Every setting is still a frame phase on the same two `sx` pulses, so theory note Proposition 2 applies unchanged.
2. The checks start from eigenstates, so their outcomes do not depend on a detector-anchored hidden variable. Selecting clean blocks therefore leaves a persistence signal unbiased, apart from the readout contrast factor `(1 - e0 - e1)^2`.
3. A readout error mimics a leaked check with probability about `e0/2`. Such false flags only discard data. The persistence of real leaks is estimated after two flagged checks in a row, where false flags are negligible.
4. Readout memory survives the selection. Tagging separates leakage from persistence, not readout memory from persistence.

The run uses the same six qubits as on 15 October (21, 154, 11, 149, 109 and 89), so that the qubit-level anomalies can be compared. This replaces the builder's selection rule for this job only. On the 15 October 17:52 EDT calibration, their readout errors are 0.26% to 0.32%.

## 4. Conditions, statistics and decision rules

| Condition | Records per shot | Record period | Sets x shots | QPU time |
|---|---|---|---|---|
| T0 | 66 (11 blocks) | 1.444 µs | 434 x 400 | 60 s |
| T1 | 66 | 5.444 µs | 246 x 400 | 60 s |
| T2 | 18 (3 blocks) | 41.444 µs | 100 x 400 | 40 s |
| C2 and amplitude check | 1 | | 32 sets and 5 circuits, x 400 | 3.7 s |

The jobs are submitted as a batch of four: the calibrations together, then each tagged condition alone. The estimate is 163 s plus about 10 s of job overhead, against 198 s remaining in the rolling allowance.

The statistics, computed per qubit and pooled for each condition by `analysis/cloud_run_analyze.py`, are:

1. **S1, clean cross-block covariance.** The covariance between B records at the same position in consecutive clean blocks (lag 6). It is flagged when the pooled value exceeds five standard errors.
2. **S2, `D_fix_clean`.** The within-block lag-1 covariance on clean blocks minus the amplitude term, with its five-sigma upper limit. Any qubit beyond five standard errors in either direction is flagged.
3. **S3, clean within-block lag-2 and lag-3 covariances.**
4. **S4, leakage statistics:** the fraction of leaked checks, of first checks leaked, and of clean blocks, and the probability that a check is leaked after two leaked checks. That probability gives the leak lifetime as `-(6 x record period) / ln p`.

The decision rules are:

1. **S1 consistent with zero at all three gaps:** the background is attributed to leakage as tagged. A nonzero S1 means tagging does not account for it, and the cause stays open.
2. **S2 consistent with zero on a qubit:** its five-sigma upper limit bounds detector-anchored persistence for that qubit's readout at that gap. A nonzero S2 of either sign is readout memory or persistence, which a fixed angle cannot separate. Qubits 11 and 109 are expected to be nonzero.
3. **S4 across gaps:** if leakage is caused by readouts, the leaked fraction falls roughly in proportion to the readout rate; if it is caused by elapsed time, the fraction stays roughly constant.

## 5. Simulation checks

`analysis/cloud_run_followup.py` passes 26 checks. These are simulated values for 120,000 shots of 66 records, with leakage near the levels fitted to qubit 149 and doubled for resolution:

| Simulated case | Clean lag 1 | Clean cross-block | Unselected cross-block |
|---|---|---|---|
| Quantum mechanics, leakage | -0.0002 ± 0.0005 | -0.0009 ± 0.0005 | 0.0351 ± 0.0005 |
| Leakage and readout memory +1% | 0.0054 ± 0.0005 | 0.0001 ± 0.0005 | 0.0354 ± 0.0005 |
| Leakage and readout memory -1% | -0.0036 ± 0.0005 | 0.0003 ± 0.0005 | 0.0359 ± 0.0005 |
| Detector-anchored, w = 0.3, leakage | 0.2882 ± 0.0005 (`0.3 (1 - e0 - e1)^2` = 0.2858) | 0.0009 ± 0.0005 | 0.0371 ± 0.0005 |

After two flagged checks, the estimated leak persistence is 0.884 against a model value of 0.886. The analysis self-tests (`python analysis/cloud_run_analyze.py selftest`, 40 checks) carry tagged runs with leakage through saving, loading and analysis.

A separate simulation shows that leakage alone does not bias the order-2 cusp fit: w_D = 0.0004 ± 0.0061, and -0.0025 ± 0.0062 at twice the leak rate. A leak-tagged shape scan would therefore not explain the negative order-2 cusp estimate of 15 October, and it is not part of this design.

## 6. What more QPU time would buy

The follow-up fits in the remaining free allowance. Extra time (pay-as-you-go, $96 per minute per IBM's pricing page) improves statistical reach only as the square root of the time:

1. **Doubling T0 to T2:** about 3 minutes, roughly $290. Per-qubit S2 limits improve by a factor of 1.4, from about 2.2e-3 to about 1.5e-3 at 1.44 µs.
2. **Tightening the detector-anchored shape bound** from 5.6e-3 to about 1e-3 (order-4 background): about 30 times the 15 October scan, roughly 90 minutes and $9,000. This is not recommended.
3. **Replicating on ibm_fez:** about 6 minutes, roughly $580. This is the most useful extra purchase if the follow-up attributes the background to leakage, because it tests whether the qubit-level readout memory is specific to this device.

## 7. Expected outcomes

These are the author's judgment, not computed probabilities:

1. S1 consistent with zero at 1.44 µs: about 60%. The leakage model already matches lags 2 to 8 there. At 5.44 and 41.4 µs: about 40%, because the model under-predicted those gaps.
2. S2 nonzero on qubits 11 and 109: about 85%. S2 consistent with zero on at least three of the other four qubits: about 60%.
3. Leakage tied mainly to readouts rather than elapsed time: about 50%.

## 8. Commands

```sh
python3 analysis/cloud_run_followup.py --output analysis/cloud_run_followup.json         # design checks
~/.venvs/qiskit/bin/python analysis/cloud_run_submit.py --job followup                     # estimate only
~/.venvs/qiskit/bin/python analysis/cloud_run_submit.py --job followup --submit --out data/cloudrun/2026-10-DD_followup
```

## 9. Add-on: one leak-tagged condition on ibm_fez

Ran on 15 October 2026 (job db8pactckpjs73b1f6m0). Its records are in `data/cloudrun/2026-10-15_fezmini/`, and its results are in Section 8 of `notes/CLOUD_RUN_RESULTS.md`.

Added 15 October 2026, before its data, with the author's go-ahead for this job. It uses the last 20 s of the 28-day allowance and puts the first ibm_fez data on record. The questions are whether ibm_fez shows the shot-wide leakage background and qubit-level readout memory seen on ibm_kingston, and at what size.

1. **Condition.** F0 is the T0 chain of Section 3 (66 records, no added delay; record period 1.39 µs with fez's 1.34 µs `measure_2` and 24 ns `sx`), with 88 sets x 400 shots, about 12 s. The amplitude check is included; the phase check is left out to save time. Everything runs as one job, estimated at 12.5 s plus job overhead.
2. **Qubits.** The builder picks six by its usual rule (lowest readout error, non-adjacent, `measure_2` available) from the 15 October 22:46 EDT calibration: 124, 26, 86, 144, 2 and 6. These differ from the kingston qubits, so the comparison is between devices, not qubits.
3. **Statistics.** S1, S2 and S4 of Section 4, computed by `analysis/cloud_run_analyze.py` as for T0. The five-sigma reach is about 4.7e-3 per qubit and 1.9e-3 pooled.
4. **Reading.** This is a small replication of one condition, with no shape scan and no qubit-anchored chain. It shows whether the kingston leakage and readout-memory pattern is device-specific. It does not change the kingston limits.

## 10. Add-on 2: the ibm_fez qubits again, at two spacings (paid)

Added 15 October 2026, before its data, after the author approved about one paid minute ($96 per minute pay-as-you-go). The F0 add-on (Section 9) left a positive clean neighbor covariance on fez qubits 6 (+0.0051 ± 0.0010), 144 (+0.0038) and 124 (+0.0036). At a fixed angle that is readout memory or detector-anchored persistence. This add-on asks two questions that one minute can answer:

1. **Is the positive covariance reproducible on the same qubits?**
2. **How does it change with spacing?**

Design:

1. **Conditions.** These run as one job on the F0 qubits (124, 26, 86, 144, 2 and 6) on the author's pay-as-you-go instance, reached with the default saved key (`--instance-of payg`):

   | Condition | Chain | Record period | Sets x shots | QPU time |
   |---|---|---|---|---|
   | G0 | T0 chain | 1.39 µs | 198 x 400 | 27 s |
   | G1 | T1 chain | 5.39 µs | 111 x 400 | 27 s |

   There is no amplitude check. F0 measured amplitude terms of at most 7.6e-5, below a tenth of the standard errors here. The estimate is 54.0 s plus 3 s of job overhead, and `--max-qpu-s 60` refuses submission above 60 s. The instance itself has a 63 s usage limit. The two conditions were trimmed from 29 s to 27 s each before any data, to leave margin under that limit.
2. **Statistics.** S1, S2 and S4 of Section 4 for each condition. Per qubit, S2 has a standard error of about 6e-4 in G0 and 8e-4 in G1.
3. **Readings, fixed in advance.**
   - *Reproducibility:* a qubit's G0 value agrees with F0 if they differ by less than three combined standard errors.
   - *Spacing:* the renewal law `w = s_m exp(-gap/tau)` can only fall or stay level as the gap grows. If S2 on a qubit rises from G0 to G1 by more than three combined standard errors, the detector-anchored completion cannot explain that qubit's covariance, so it is conventional readout memory. A fall or a level value is consistent with both explanations, and only a shape scan separates them.
4. **Not covered.** The add-on includes no shape scan and no qubit-anchored chain, and it does not change the kingston limits.

Outcome (15 October 2026): job db8plalckpjs73b1fung ran from 03:31:56 to 03:33:05 UTC. IBM then cancelled it after charging 58 s, and no results could be retrieved. IBM's documentation says running jobs are cancelled when a pay-as-you-go instance reaches its cost limit, so the 63 s limit most likely stopped it: the 57 s estimate with overhead left too little margin. The submission script now refuses any job whose estimate exceeds 75% of the instance's remaining usage. This design is unchanged and can be rerun as written.

Rerun (15 October 2026, after the cost limit was raised to $250): job db8pnu8b3g5c73egp3l0 completed, charged 58 s. Its records are in `data/cloudrun/2026-10-15_fezrepeat_rerun/`, and its results are in Section 9.2 of `notes/CLOUD_RUN_RESULTS.md`.

