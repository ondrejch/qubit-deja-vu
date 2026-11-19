#!/usr/bin/env python3
"""Eigenstate repeat/flip extension to the leak-tagged chain.

Draft, 1 November 2026. Extends notes/CLOUD_RUN_FOLLOWUP_DESIGN.md without
changing its tagged statistics. Records come in blocks of eight:

  B B B B F F R P

B and F are the follow-up steps (balanced pi/2 and flip pi). R is a repeat
step (phi = pi, an identity on an eigenstate) and P is a flip step (phi = 0)
outside the leak check. After a normal check ending at x, R should give x and P
should give -x. A leaked level that ignores the pulses reports 1, 1 for R, P.

Why: the October chains measure readout memory for equator-state inputs
(B steps). A syndrome ancilla is read near an eigenstate. If the memory sits in
the readout assignment, a comparable bias applies to syndrome readouts; if it
acts through the pulses that follow a readout, its effect on eigenstate readouts
differs (the IBM analysis note, Section 2.1). R and P measure the
memory there directly.

Tagging is unchanged: a block is clean when its own F check and the previous
block check are both normal (x, -x, x). Checks start from eigenstates, so their
outcomes do not depend on the detector-anchored hidden variable; selecting clean
blocks leaves D persistence unbiased apart from readout contrast, as in the
follow-up design. R and P are never used in the selection.

Every statistic is reported on three masks, because the clean selection is
itself a conditioning on F2 and therefore moves the eigenstate fidelities it is
meant to measure:

  all         every block
  clean       own and previous check normal (the follow-up rule)
  clean_next  clean and the next block's check normal too

A leak that begins during R or P leaves its block marked clean and lowers
flip_agree in the same direction as assignment memory, so the clean_next mask is
reported beside clean and the difference quantifies that residual (the other
options, a flagging step after P or a leak-model bias correction, would change
the chain or need a fitted model).

Fidelities are also split by input state: repeat_agree_0 and _1 condition on
F2 reading 0 or 1, flip_agree_0 and _1 on R reading 0 or 1. T1 relaxation lowers
both _1 fidelities by about the same amount, positive memory (a "1" makes the
next readout of |0> report "1") lowers flip_agree_1 only, and negative memory (a
"1" makes the next readout of |1> report "0") lowers repeat_agree_1 only. The
primary statistic is therefore the contrast

  contrast = flip_agree_1 - repeat_agree_1,

which T1 leaves almost unchanged and which memory moves with its sign. Static
readout asymmetry moves it as much as 1 percent memory does, so its null is
built per qubit (qubit_null) from that qubit's own assignment errors, measured
in the same run by the AS condition (assign_errors), its calibrated T1 and its
own leaked-check fraction. eigen_reading applies the rule fixed in
notes/CLOUD_RUN_NEXT_DESIGN.md Section 2G: a deviation counts only beyond
K_SIGMA combined standard errors, on both the clean and clean_next masks.

flip_control = P(F1 = -x) is a flip fidelity present in every block, quoted on
unselected blocks. D_flip is sign-blind (its ideal value sits at the -1 floor);
the sign of a memory shows in the contrast, in repeat_agree and in the B lag-1
covariance.

NumPy only. Nothing here contacts IBM and nothing here is experimental
data. Submit only with the author's explicit go-ahead for that job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_run_design as design  # noqa: E402
import cloud_run_followup as followup  # noqa: E402
import cloud_run_submit as submit  # noqa: E402  (its module level imports no Qiskit)

BLOCK = 8
B_POS = (0, 1, 2, 3)
F1, F2 = 4, 5
R_POS, P_POS = 6, 7
CHECK_POS = (3, F1, F2)
MASKS = ("all", "clean", "clean_next")
TWO_PI = 2.0 * math.pi

# E0 runs on ibm_fez with no added delay: two 24 ns sx pulses and a 1.34 µs
# measure_2 (the live 15 October durations). The relaxation case uses it.
FEZ_READOUT_S, FEZ_SX_S = 1.34e-6, 24e-9
FEZ_PERIOD_S = 2 * FEZ_SX_S + FEZ_READOUT_S
T1_S = 100e-6      # deliberately short against the 137 to 366 µs the 15 October calibration listed
T1_LONG_S = 300e-6  # inside that range; the two cases give the null's relaxation term its T1 scaling

# Qubits. The fresh-qubit question (are the positive clean memories of fez 124,
# 144 and 6 those qubits or device-wide?) needs both fresh qubits and the old
# positives in the same window, since a null on fresh qubits four weeks later
# alone could be drift. The other October fez qubits are excluded so the
# picker's lowest-readout-error rule cannot hand them back as "fresh". QPU time
# depends on shots and circuit length, not on qubit count, so the extra qubits
# cost nothing; readouts per shot grow to EIG_QUBITS x 72 = 576 (October ran
# 384, so this size is untested on hardware).
EIG_CONTROLS = (124, 144, 6)
EIG_EXCLUDE = tuple(q for q in followup.FEZ_LAYOUT if q not in EIG_CONTROLS)
EIG_QUBITS = 8
K_SIGMA = 3.0  # combined standard errors a deviation from the per-qubit null must exceed
# The 15 October jobs were charged 2.5 to 5.1 s above their estimates
# (notes/CLOUD_RUN_RESULTS.md, job tables): (estimate, charged) per job.
OCTOBER_CHARGES = ((12.5, 15.0), (60.0, 65.0), (60.1, 64.0), (59.9, 63.0), (90.0, 95.0), (90.0, 95.0),
                   (59.9, 65.0), (60.0, 64.0), (39.8, 43.0), (54.0, 58.0))


def tagged_eig_settings(n: int, K: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """alpha, beta, phi (n, K) for B B B B F F R P blocks; K a multiple of 8.

    Same B branch and the same draws as followup.tagged_settings, so the two
    chains share a random stream per block. phi = 0 at the flip steps
    (F1, F2, P) and phi = pi at the repeat step R.
    """
    if K % BLOCK:
        raise ValueError("records per shot must be a multiple of 8")
    alpha = np.empty((n, K))
    phi = np.empty((n, K))
    for b in range(K // BLOCK):
        a0 = rng.uniform(0.0, TWO_PI, n)
        for j in range(BLOCK):
            k = BLOCK * b + j
            if j in B_POS:
                if j % 2 == 0:
                    alpha[:, k], phi[:, k] = a0, 0.5 * math.pi
                else:
                    alpha[:, k], phi[:, k] = a0 + 1.5 * math.pi, -0.5 * math.pi
                phi[:, k] += math.pi * (rng.random(n) < 0.5)
            else:  # F1, F2, R and P take a fresh random frame each step
                alpha[:, k] = rng.uniform(0.0, TWO_PI, n)
                phi[:, k] = math.pi if j == R_POS else 0.0
    return alpha, alpha + phi, np.mod(phi, TWO_PI)


# ---------------------------------------------------------------- statistics

def block_masks(r: np.ndarray) -> Dict[str, np.ndarray]:
    """(N, K) outcomes -> the three block masks of the module docstring."""
    st = followup.check_states(r, BLOCK)
    clean = followup.clean_blocks(st)
    nxt = np.concatenate([clean[..., 1:], np.zeros(clean[..., :1].shape, bool)], axis=-1)
    return {"all": np.ones_like(clean), "clean": clean, "clean_next": clean & nxt}


def _masked_cov(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> Tuple[float, float]:
    x = x[m].astype(np.float64)
    y = y[m].astype(np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    prod = (x - x.mean()) * (y - y.mean())
    return float(prod.mean()), float(prod.std() / math.sqrt(prod.size))


def _masked_rate(hit: np.ndarray, m: np.ndarray) -> Tuple[float, float]:
    """Fraction of masked blocks where hit holds, and its binomial standard error."""
    h = hit[m]
    if h.size == 0:
        return float("nan"), float("nan")
    p = float(h.mean())
    return p, float(math.sqrt(p * (1.0 - p) / h.size))


def eigen_statistics(r: np.ndarray) -> Dict[str, object]:
    """Eigenstate-input memory, one qubit: r (N, K), rows are shots.

    Per mask: covariances D_repeat = Cov(F2, R) and D_flip = Cov(R, P), the
    repeat fidelity P(R == F2), the flip fidelity P(P == -R) and the control
    P(F1 == -x), and the B-record lag-1 and lag-2 covariances for the equator
    comparison. Values are (estimate, standard error) pairs.
    """
    N, K = r.shape
    nb = K // BLOCK
    masks = block_masks(r)
    blocks = r.reshape(N, nb, BLOCK)
    col = {name: blocks[:, :, i].reshape(-1)
           for name, i in (("x", 3), ("f1", F1), ("f2", F2), ("r", R_POS), ("p", P_POS))}
    out: Dict[str, object] = {}
    for name in MASKS:
        m = masks[name].reshape(-1)
        st: Dict[str, object] = {}
        st["D_repeat"] = _masked_cov(col["f2"], col["r"], m)
        st["D_flip"] = _masked_cov(col["r"], col["p"], m)
        st["repeat_agree"] = _masked_rate(col["r"] == col["f2"], m)
        st["flip_agree"] = _masked_rate(col["p"] == -col["r"], m)
        st["flip_control"] = _masked_rate(col["f1"] == -col["x"], m)
        for lab, val in (("0", 1), ("1", -1)):
            st[f"repeat_agree_{lab}"] = _masked_rate(col["r"] == col["f2"], m & (col["f2"] == val))
            st[f"flip_agree_{lab}"] = _masked_rate(col["p"] == -col["r"], m & (col["r"] == val))
        st["contrast"] = (st["flip_agree_1"][0] - st["repeat_agree_1"][0],
                          math.hypot(st["flip_agree_1"][1], st["repeat_agree_1"][1]))
        for lag in (1, 2):
            xs = np.concatenate([blocks[:, :, j] for j in range(4 - lag)], axis=1)
            ys = np.concatenate([blocks[:, :, j + lag] for j in range(4 - lag)], axis=1)
            ms = np.concatenate([masks[name]] * (4 - lag), axis=1)
            st[f"B_lag{lag}"] = _masked_cov(xs.reshape(-1), ys.reshape(-1), ms.reshape(-1))
        st["n_blocks"] = int(m.sum())
        out[name] = st
    st_all = followup.check_states(r, BLOCK)
    out["fraction_checks_leaked"] = float((st_all == 1).mean())
    out["fraction_checks_other"] = float((st_all == 2).mean())
    out["fraction_blocks_clean"] = float(masks["clean"].mean())
    out["fraction_blocks_clean_next"] = float(masks["clean_next"].mean())
    shift = {k: float(out["clean"][k][0] - out["all"][k][0])  # type: ignore[index]
             for k in ("repeat_agree", "flip_agree")}
    out["selection_shift_repeat"] = shift["repeat_agree"]
    out["selection_shift_flip"] = shift["flip_agree"]
    return out


def pool_eigen(per_qubit: list) -> Dict[str, Tuple[float, float]]:
    """Pooled (value, standard error) per key: the unweighted mean of the
    per-qubit values and the standard error of that mean, as
    analysis/cloud_run_analyze.py._pool computes it for the October runs. The
    per-qubit values stay primary: a pooled value misrepresents every qubit on
    its own (notes/CLOUD_RUN_RESULTS.md, item 6 of Section 1).
    """
    keys = [k for k, v in per_qubit[0].items() if isinstance(v, tuple)]
    out = {}
    for key in keys:
        vals = np.array([pq[key][0] for pq in per_qubit], float)
        ses = np.array([pq[key][1] for pq in per_qubit], float)
        out[key] = (float(vals.mean()), float(math.sqrt(float(np.sum(ses ** 2))) / len(vals)))
    return out


# ---------------------------------------------------------------- per-qubit null

def assign_settings(n_sets: int, n_qubits: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """alpha, beta, phi (n_sets * n_qubits, 1) for the AS condition: phi = pi
    (prepare |0>) where set + qubit is even, phi = 0 (prepare |1>) where odd."""
    j, q = np.meshgrid(np.arange(n_sets), np.arange(n_qubits), indexing="ij")
    phi = np.where((j + q) % 2 == 0, math.pi, 0.0).reshape(-1, 1)
    alpha = rng.uniform(0.0, TWO_PI, phi.shape)
    return alpha, alpha + phi, phi


def assign_errors(r: np.ndarray, phi: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """One qubit's static assignment errors from the AS condition.

    r (sets, shots) outcomes (+1 = "0"), phi (sets,) the step phase. Returns
    e0 = P("1" | |0>) and e1 = P("0" | |1>) with binomial standard errors.
    """
    prep0 = np.cos(phi) < 0
    out = {}
    for name, sel, bad in (("e0", prep0, -1), ("e1", ~prep0, 1)):
        x = r[sel].reshape(-1)
        p = float((x == bad).mean()) if x.size else float("nan")
        out[name] = (p, float(math.sqrt(p * (1 - p) / x.size)) if x.size else float("nan"))
    return out


NULL_LEAK = dict(g_dn=0.015, q2=0.99)  # leak return and leaked readout, as in the table below
NULL_SHOTS = 400_000                  # simulated shots for a qubit's null
NULL_KEYS = ("repeat_agree", "flip_agree", "repeat_agree_0", "repeat_agree_1",
             "flip_agree_0", "flip_agree_1", "contrast")


def _leak_rate_for(e0: float, e1: float, t1_s: float | None, leaked_fraction: float, seed: int,
                   n: int = 40_000) -> float:
    """Readout leak rate g (g0 = g1 = g) whose simulated leaked-check fraction matches the qubit's."""
    lo, hi = 0.0, 0.01
    for i in range(14):
        g = 0.5 * (lo + hi)
        rng = np.random.default_rng(seed + i)
        _, _, phi = tagged_eig_settings(n, 72, rng)
        r = followup.simulate_leaky("QM", phi, rng, e0=e0, e1=e1, g0=g, g1=g, t1=t1_s,
                                    period_s=FEZ_PERIOD_S if t1_s else None, **NULL_LEAK)
        frac = float((followup.check_states(r, BLOCK) == 1).mean())
        lo, hi = (g, hi) if frac < leaked_fraction else (lo, g)
    return 0.5 * (lo + hi)


def qubit_null(e0: float, e1: float, t1_s: float | None, leaked_fraction: float,
               seed: int = 20261102, n: int = NULL_SHOTS, K: int = 72) -> Dict[str, object]:
    """Conventional null for one qubit: leakage plus T1 relaxation plus its own
    static assignment errors, no readout memory and no persistence.

    e0 = P("1" | |0>) and e1 = P("0" | |1>) come from the run's AS condition
    (assign_errors), t1_s from the run-day calibration in the build report,
    and the leak rate is tuned so the simulated leaked-check fraction matches
    the qubit's measured one. Leak return and the leaked readout are fixed at
    NULL_LEAK. Returns the fidelities of eigen_statistics on every mask with
    simulation standard errors, plus the inputs.
    """
    g = _leak_rate_for(e0, e1, t1_s, leaked_fraction, seed)
    rng = np.random.default_rng(seed)
    _, _, phi = tagged_eig_settings(n, K, rng)
    r = followup.simulate_leaky("QM", phi, rng, e0=e0, e1=e1, g0=g, g1=g, t1=t1_s,
                                period_s=FEZ_PERIOD_S if t1_s else None, **NULL_LEAK)
    st = eigen_statistics(r)
    out: Dict[str, object] = {m: {k: st[m][k] for k in NULL_KEYS} for m in MASKS}  # type: ignore[index]
    out["inputs"] = {"e0": e0, "e1": e1, "t1_s": t1_s, "leak_rate": g,
                     "leaked_fraction_target": leaked_fraction,
                     "leaked_fraction_null": st["fraction_checks_leaked"], **NULL_LEAK}
    out["selection_shift_repeat"] = st["selection_shift_repeat"]
    out["selection_shift_flip"] = st["selection_shift_flip"]
    return out


def eigen_reading(st: Dict[str, object], null: Dict[str, object],
                  sens: Dict[str, Tuple[float, float]] | None = None,
                  e_se: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, object]:
    """The reading fixed in advance (note Section 2G, item 5), for one qubit.

    For each key of NULL_KEYS on the clean and clean_next masks: deviation =
    measured - null, its combined standard error (data, simulation, and the
    assignment calibration's errors propagated through sens, the per-unit-error
    slopes from the design table), and z. The contrast is primary: a memory is
    read only when |z| > K_SIGMA on both masks with the same sign. Negative
    contrast deviation is positive memory, positive is negative memory. The
    absolute fidelities are secondary: a deficit counts beyond K_SIGMA plus the
    difference between the measured and null selection shifts.
    """
    out: Dict[str, object] = {}
    for mask in ("clean", "clean_next"):
        rows = {}
        for key in NULL_KEYS:
            v, s = st[mask][key]  # type: ignore[index]
            nv, ns = null[mask][key]  # type: ignore[index]
            se2 = s ** 2 + ns ** 2
            if sens and key in sens:
                se2 += (sens[key][0] * e_se[0]) ** 2 + (sens[key][1] * e_se[1]) ** 2
            se = math.sqrt(se2)
            rows[key] = {"deviation": v - nv, "se": se, "z": (v - nv) / se if se > 0 else float("nan")}
        out[mask] = rows
    zc, zn = out["clean"]["contrast"]["z"], out["clean_next"]["contrast"]["z"]  # type: ignore[index]
    if abs(zc) > K_SIGMA and abs(zn) > K_SIGMA and zc * zn > 0:
        out["contrast_reading"] = ("positive memory on eigenstate readouts" if zc < 0
                                   else "negative memory on eigenstate readouts")
    else:
        out["contrast_reading"] = "consistent with the per-qubit conventional null"
    margins = {"repeat_agree": abs(float(st["selection_shift_repeat"]) - float(null["selection_shift_repeat"])),
               "flip_agree": abs(float(st["selection_shift_flip"]) - float(null["selection_shift_flip"]))}
    out["fidelity_below_null"] = {
        key: bool(-out["clean"][key]["deviation"] > K_SIGMA * out["clean"][key]["se"] + margins[key])  # type: ignore[index]
        for key in margins}
    out["selection_margin"] = margins
    return out


# ---------------------------------------------------------------- plan and budget

# The next-window chain: label, kind, added delay before each step (s), records
# per shot, QPU seconds. One source for analysis/cloud_run_submit.py,
# analysis/cloud_run_circuits.py and budget() below.
# AS is the assignment calibration: one step from |0> with phi = pi (stays |0>)
# or phi = 0 (flips to |1>), alternating over qubits and sets, so each readout is
# the first of its shot and both preparations carry the same two sx pulses.
EIG_CONDITIONS = [("E0", "tagged_eig", 0.0, 72, 360.0), ("C2", "phase", 0.0, 1, 4.0),
                  ("AS", "assign", 0.0, 1, 20.0)]
# The 66-record fresh-qubit chain (candidate A of the note) that E0 replaces: E0
# measures the same tagged statistic on the same fresh qubits, so the two are
# alternatives rather than additions. Kept here for its reach only.
A_CONDITION = ("A", "tagged", 0.0, 66, 60.0)
# The repeat (E0R, notes/CLOUD_RUN_NEXT_DESIGN.md Section 2H): the same chain on
# the same eight qubits as the 19 November E0 run, sized to the 185 s left in the
# window. The submit script sends only when the estimate with JOB_OVERHEAD_S per
# job stays under USAGE_MARGIN of the remaining usage, so the plan must fit
# REPEAT_WINDOW_S * USAGE_MARGIN = 138.75 s.
EIG_REPEAT_CONDITIONS = [("E0", "tagged_eig", 0.0, 72, 98.0), ("C2", "phase", 0.0, 1, 4.0),
                         ("AS", "assign", 0.0, 1, 20.0)]
EIG_REPEAT_LAYOUT = (124, 144, 6, 50, 23, 59, 114, 132)
REPEAT_WINDOW_S = 185.0
A_QUBITS = 6  # candidate A was planned on six qubits


def _sets_for(seconds: float, K: int, delay: float, rep_delay: float, shots: int) -> Tuple[float, int]:
    """(record period, parameter sets) as analysis/cloud_run_circuits.build_all does."""
    period = 2 * FEZ_SX_S + FEZ_READOUT_S + delay
    t_exec = rep_delay + K * period
    return period, max(1, int(round(seconds / (t_exec * shots))))


def budget(qubits: int = EIG_QUBITS, rep_delay: float = 250e-6, shots: int = 400,
           overhead_per_job: float = submit.JOB_OVERHEAD_S,
           max_job_s: float = submit.MAX_JOB_S, clean_fraction: float | None = None,
           conditions: list | None = None) -> Dict[str, object]:
    """QPU budget for one free window, using analysis/cloud_run_submit.py's own
    partitioner so that the job count and the per-job estimates are the ones that
    job plan would produce. No B0eig row: that chain has no settings, simulation
    or analysis, so it is not budgeted (see the note, Section 3).
    """
    conditions = conditions or EIG_CONDITIONS
    rows, entries = [], []
    for label, kind, delay, K, seconds in conditions:
        period, sets = _sets_for(seconds, K, delay, rep_delay, shots)
        per_set = (rep_delay + K * period) * shots
        spans = submit.set_spans(sets, per_set, max_job_s)
        entries.append((label, sets, per_set))
        row: Dict[str, object] = {
            "label": label, "kind": kind, "records": K, "period_us": round(period * 1e6, 3),
            "sets": sets, "pub_parts": len(spans),
            "part_qpu_s": [round((b - a) * per_set, 1) for a, b in spans],
            "qpu_s": round(sets * per_set, 1)}
        if K > BLOCK:
            pairs = sets * shots * (K // BLOCK) * 3
            row["neighbor_pairs_per_qubit"] = f"{pairs:.3e}"
            row["five_sigma_per_qubit_if_all_clean"] = round(5.0 / math.sqrt(pairs), 6)
            row["five_sigma_pooled_if_all_clean"] = round(5.0 / math.sqrt(pairs * qubits), 6)
            if clean_fraction:
                row["five_sigma_per_qubit_at_simulated_clean"] = round(5.0 / math.sqrt(pairs * clean_fraction), 6)
                row["five_sigma_pooled_at_simulated_clean"] = round(5.0 / math.sqrt(pairs * clean_fraction * qubits), 6)
        rows.append(row)
    amp_each = shots * (rep_delay + 100 * FEZ_SX_S + FEZ_READOUT_S)  # the builder's amplitude estimate
    amp_s = len(design.AMPLITUDE_PULSES) * amp_each
    for m in design.AMPLITUDE_PULSES:
        entries.append((f"amplitude_{m}", 1, amp_each))
    plan = submit.job_plan(entries, max_job_s)
    per_set_by_label = {label: per_set for label, _sets, per_set in entries}
    estimates = [round(sum((span[1] - span[0]) * per_set_by_label[label] for label, span in job), 2)
                 for job in plan]
    total = sum(estimates)
    worst = max(c - e for e, c in OCTOBER_CHARGES)
    period_a, sets_a = _sets_for(A_CONDITION[4], A_CONDITION[3], A_CONDITION[2], rep_delay, shots)
    pairs_a = sets_a * shots * (A_CONDITION[3] // followup.BLOCK) * 3
    return {
        "conditions": rows,
        "amplitude_check_s": round(amp_s, 2),
        "jobs": [{"entries": [label for label, _ in job], "estimated_qpu_s": e} for job, e in zip(plan, estimates)],
        "n_jobs": len(plan),
        "max_job_qpu_s": max(estimates),
        "total_qpu_s": round(total, 1),
        "overhead_s": round(len(plan) * overhead_per_job, 1),
        "total_with_overhead_s": round(total + len(plan) * overhead_per_job, 1),
        "fits_600s_window": bool(total + len(plan) * overhead_per_job < 600.0),
        "spare_in_window_s": round(600.0 - total - len(plan) * overhead_per_job, 1),
        "october_overhead_range_s": [round(min(c - e for e, c in OCTOBER_CHARGES), 1), round(worst, 1)],
        "total_at_worst_october_overhead_s": round(total + len(plan) * worst, 1),
        "spare_at_worst_october_overhead_s": round(600.0 - total - len(plan) * worst, 1),
        "max_job_at_worst_october_overhead_s": round(max(estimates) + worst, 1),
        "qubits": qubits, "readouts_per_shot": qubits * conditions[0][3],
        "supersedes": {
            "label": A_CONDITION[0], "records": A_CONDITION[3], "qpu_s": A_CONDITION[4],
            "sets": sets_a, "period_us": round(period_a * 1e6, 3),
            "five_sigma_per_qubit_if_all_clean": round(5.0 / math.sqrt(pairs_a), 6),
            "five_sigma_pooled_if_all_clean": round(5.0 / math.sqrt(pairs_a * A_QUBITS), 6),
            "note": "Candidate A of the note; E0 replaces it as the fresh-qubit condition, "
                    "so A's reach does not carry over to E0 and A is not in the total above.",
        },
        "note": "Job plan from analysis/cloud_run_submit.job_plan at a "
                f"{max_job_s:.0f} s per-job cap; sets follow the builder's formula, "
                "so the live build's estimate is authoritative. Submit needs author go-ahead.",
    }


class Check:
    def __init__(self) -> None:
        self.count = 0

    def within(self, label: str, value: float, se: float, want: float, k: float = 5.0) -> None:
        self.count += 1
        if not abs(value - want) <= k * se + 1e-12:
            raise AssertionError(f"{label}: got {value:.5f} +- {se:.5f}, want {want:.5f}")

    def true(self, label: str, cond: bool) -> None:
        self.count += 1
        if not cond:
            raise AssertionError(label)


def _pair(st: Dict[str, object], mask: str, key: str) -> Tuple[float, float]:
    inner = st[mask]
    assert isinstance(inner, dict)
    value = inner[key]
    assert isinstance(value, tuple)
    return float(value[0]), float(value[1])


def _num(st: Dict[str, object], key: str) -> float:
    value = st[key]
    assert isinstance(value, float)
    return value


def run(output: str | None) -> int:
    rng = np.random.default_rng(20261101)
    chk = Check()
    n, K = 120_000, 72
    alpha, beta, phi = tagged_eig_settings(n, K, rng)
    # Layout: B pi/2, F and P 0, R pi per block.
    chk.true("R steps carry phi = pi",
             bool(np.allclose(np.cos(phi[:, R_POS::BLOCK]), -1.0)))
    chk.true("F and P steps carry phi = 0",
             bool(np.allclose(np.cos(phi[:, F1::BLOCK]), 1.0)
                  and np.allclose(np.cos(phi[:, P_POS::BLOCK]), 1.0)))
    chk.true("B steps balanced",
             bool(np.allclose(np.cos(phi[:, 0::BLOCK]), 0.0, atol=1e-12)))
    table = {}
    leak = dict(e0=0.012, e1=0.012, g0=0.002, g1=0.002, g_dn=0.015, q2=0.99)
    cases = [
        ("QM, ideal", "QM", 0.0, {}),
        ("QM, leakage", "QM", 0.0, leak),
        ("QM, leakage + readout memory +1%", "QM", 0.0, {**leak, "mem_pos": 0.01}),
        ("D w=0.3, leakage", "D", 0.3, leak),
        ("QM, leakage + readout memory -1%", "QM", 0.0, {**leak, "mem_neg": 0.01}),
        ("QM, leakage + T1 relaxation", "QM", 0.0, {**leak, "t1": T1_S, "period_s": FEZ_PERIOD_S}),
        ("QM, leakage + T1 = 300 us relaxation", "QM", 0.0,
         {**leak, "t1": T1_LONG_S, "period_s": FEZ_PERIOD_S}),
        # Appended after the October-era cases so their random streams are unchanged.
        ("QM, leakage + asymmetric readout (e0 0.8%, e1 1.6%)", "QM", 0.0, {**leak, "e0": 0.008, "e1": 0.016}),
        ("QM, leakage + readout e0 1.6%", "QM", 0.0, {**leak, "e0": 0.016}),
        ("QM, leakage + readout e1 1.6%", "QM", 0.0, {**leak, "e1": 0.016}),
    ]
    kept = {}
    for name, model, w, kw in cases:
        r = followup.simulate_leaky(model, phi, rng, w=w, **kw)
        st = eigen_statistics(r)
        kept[name] = st
        row: Dict[str, object] = {}
        for k, v in st.items():
            if isinstance(v, dict):
                row[k] = {k2: ([round(float(v2[0]), 5), round(float(v2[1]), 5)] if isinstance(v2, tuple) else v2)
                          for k2, v2 in v.items()}
            elif isinstance(v, tuple):
                row[k] = [round(float(v[0]), 5), round(float(v[1]), 5)]
            else:
                row[k] = round(float(v), 5)  # type: ignore[arg-type]
        table[name] = row
        contrast = (1.0 - float(kw.get("e0", 0.0)) - float(kw.get("e1", 0.0))) ** 2
        b_lag1 = _pair(st, "clean", "B_lag1")
        rep_raw, flip_raw = _pair(st, "clean", "repeat_agree"), _pair(st, "clean", "flip_agree")
        if model == "QM" and "g1" not in kw:
            chk.true(f"{name}: repeat agree is 1 without errors", rep_raw[0] == 1.0)
            chk.true(f"{name}: flip agree is 1 without errors", flip_raw[0] == 1.0)
            chk.within(f"{name}: clean B lag-1 near zero", b_lag1[0], b_lag1[1], 0.0)
        if "g1" in kw:
            chk.true(f"{name}: leaked checks detected", _num(st, "fraction_checks_leaked") > 0.005)
            chk.true(f"{name}: repeat agree in (0.95, 1.0)", 0.95 < rep_raw[0] < 1.0)
            chk.true(f"{name}: flip agree in (0.95, 1.0)", 0.95 < flip_raw[0] < 1.0)
            if "mem_pos" in kw:
                chk.true(f"{name}: readout memory survives in B lag-1", b_lag1[0] > 5 * b_lag1[1])
            elif "mem_neg" in kw:
                chk.true(f"{name}: negative readout memory survives in B lag-1", b_lag1[0] < -5 * b_lag1[1])
            elif "t1" not in kw and model == "QM":
                chk.within(f"{name}: clean B lag-1 near zero", b_lag1[0], b_lag1[1], 0.0)
            if model == "D":
                chk.within(f"{name}: clean B lag-1 near w times contrast",
                           b_lag1[0], b_lag1[1], w * contrast)
    # The clean selection moves both fidelities, and moving them is why the null
    # has to be read off the same mask.
    base = kept["QM, leakage"]
    for key, shift_key in (("repeat_agree", "selection_shift_repeat"), ("flip_agree", "selection_shift_flip")):
        all_v = _pair(base, "all", key)[0]
        clean_v = _pair(base, "clean", key)[0]
        chk.true(f"QM, leakage: clean selection raises {key}", clean_v > all_v)
        chk.within(f"QM, leakage: reported {shift_key} equals clean minus all",
                   _num(base, shift_key), 0.0, clean_v - all_v, k=1e-9)
    # A leak starting inside R or P is untagged. Requiring the next block's check
    # too removes those that persist, so clean_next's fidelity is at least the
    # clean one and the gap measures how much of the clean deficit was such leaks.
    chk.true("QM, leakage: clean_next is a subset of clean",
             _num(base, "fraction_blocks_clean_next") <= _num(base, "fraction_blocks_clean"))
    chk.true("QM, leakage: clean_next flip agree is not below clean (it removes untagged leaks)",
             _pair(base, "clean_next", "flip_agree")[0] >= _pair(base, "clean", "flip_agree")[0] - 1e-9)
    # Persistence moves the equator statistic and leaves the eigenstate
    # fidelities alone; readout memory of either sign moves the eigenstate flip
    # covariance, which is sign-blind, while repeat agree carries the sign.
    dcase = kept["D w=0.3, leakage"]
    mpos = kept["QM, leakage + readout memory +1%"]
    mneg = kept["QM, leakage + readout memory -1%"]
    for key in ("repeat_agree", "flip_agree"):
        chk.true(f"D leaves the eigenstate {key} alone",
                 abs(_pair(dcase, "clean", key)[0] - _pair(base, "clean", key)[0]) < 0.005)
    df_base, se_base = _pair(base, "clean", "D_flip")
    df_pos, se_pos = _pair(mpos, "clean", "D_flip")
    df_neg, se_neg = _pair(mneg, "clean", "D_flip")
    chk.true("positive readout memory moves the eigenstate flip covariance",
             (df_pos - df_base) > 5 * math.sqrt(se_base ** 2 + se_pos ** 2))
    chk.true("negative readout memory moves the eigenstate flip covariance the same way",
             (df_neg - df_base) > 5 * math.sqrt(se_base ** 2 + se_neg ** 2))
    chk.true("D_flip is sign-blind: the two signs move it alike",
             abs(df_pos - df_neg) < 5 * math.sqrt(se_pos ** 2 + se_neg ** 2))
    rp_pos, sp_pos = _pair(mpos, "clean", "repeat_agree")
    rp_neg, sneg = _pair(mneg, "clean", "repeat_agree")
    chk.true("repeat agree carries the sign: it separates the two memories",
             rp_pos - rp_neg > 5 * math.sqrt(sp_pos ** 2 + sneg ** 2))
    chk.true("negative memory lowers repeat agree against the null",
             _pair(base, "clean", "repeat_agree")[0] - rp_neg > 5 * math.sqrt(
                 _pair(base, "clean", "repeat_agree")[1] ** 2 + sneg ** 2))
    # F1 is one of the three check records, so this control is 1 by construction
    # on the selected masks and informative only on unselected blocks.
    for mask in ("clean", "clean_next"):
        chk.true(f"QM, leakage: flip_control is 1 on the {mask} mask, as F1 is part of the check",
                 _pair(base, mask, "flip_control")[0] == 1.0)
    residual = _pair(base, "clean_next", "flip_agree")[0] - _pair(base, "clean", "flip_agree")[0]
    mem_effect = _pair(base, "clean", "flip_agree")[0] - _pair(mpos, "clean", "flip_agree")[0]
    mem_effect_dflip = abs(df_pos - df_base)
    chk.true("the untagged-leak residual between the two clean masks is smaller than the "
             "1 percent memory effect it could hide", 0.0 < residual < mem_effect)
    # Relaxation is a conventional effect of the same size as the effect sought,
    # so it belongs in the null; the F1 control on unselected blocks tracks it,
    # which is why the control is quoted next to the fidelities.
    t1case = kept["QM, leakage + T1 relaxation"]
    chk.true("T1 relaxation lowers the clean-mask flip fidelity below the leakage-only value",
             _pair(base, "clean", "flip_agree")[0] - _pair(t1case, "clean", "flip_agree")[0] > 0.0)
    chk.true("T1 relaxation lowers the unselected F1 control below the leakage-only value",
             _pair(base, "all", "flip_control")[0] - _pair(t1case, "all", "flip_control")[0] > 0.0)
    flip_drop = _pair(base, "all", "flip_agree")[0] - _pair(t1case, "all", "flip_agree")[0]
    ctrl_drop = _pair(base, "all", "flip_control")[0] - _pair(t1case, "all", "flip_control")[0]
    chk.true("T1 lowers the P flip fidelity and the F1 control by a comparable amount "
             "on unselected blocks", 0.5 * ctrl_drop < flip_drop < 2.0 * ctrl_drop)
    chk.true("T1 relaxation moves the clean flip fidelity as much as 1 percent readout memory does",
             _pair(base, "clean", "flip_agree")[0] - _pair(t1case, "clean", "flip_agree")[0]
             > 0.5 * mem_effect)
    chk.true("T1 relaxation alone shifts D_flip as much as 1 percent readout memory does, so the "
             "null must include relaxation and D_flip is not clean against it",
             abs(_pair(base, "clean", "D_flip")[0] - _pair(t1case, "clean", "D_flip")[0])
             > 0.5 * mem_effect_dflip)
    t1slow = kept["QM, leakage + T1 = 300 us relaxation"]
    dflip_100 = abs(_pair(base, "clean", "D_flip")[0] - _pair(t1case, "clean", "D_flip")[0])
    dflip_300 = abs(_pair(base, "clean", "D_flip")[0] - _pair(t1slow, "clean", "D_flip")[0])
    chk.true("the relaxation shift of D_flip scales with 1/T1, so a run scales the null's "
             "relaxation term by its own calibrated T1",
             dflip_300 < dflip_100 and dflip_300 > 0.2 * dflip_100)

    # Input-state split and the contrast. T1 lowers both _1 fidelities and
    # leaves the contrast nearly unchanged; memory moves the contrast with its sign.
    for key in ("repeat_agree_1", "flip_agree_1"):
        drop, se = _pair(base, "clean", key)[0] - _pair(t1case, "clean", key)[0], math.hypot(
            _pair(base, "clean", key)[1], _pair(t1case, "clean", key)[1])
        chk.true(f"T1 relaxation lowers {key} by more than 5 standard errors", drop > 5 * se)
    for key in ("repeat_agree_0", "flip_agree_0"):
        drop = _pair(base, "clean", key)[0] - _pair(t1case, "clean", key)[0]
        chk.true(f"T1 relaxation moves {key} by less than 0.002", abs(drop) < 0.002)
    con = {name: _pair(kept[name], "clean", "contrast") for name in kept}
    mem_shift_pos = con["QM, leakage + readout memory +1%"][0] - con["QM, leakage"][0]
    mem_shift_neg = con["QM, leakage + readout memory -1%"][0] - con["QM, leakage"][0]
    t1_shift = con["QM, leakage + T1 relaxation"][0] - con["QM, leakage"][0]
    se_con = math.hypot(con["QM, leakage"][1], con["QM, leakage + readout memory +1%"][1])
    chk.true("positive memory lowers the contrast by more than 5 standard errors", mem_shift_pos < -5 * se_con)
    chk.true("negative memory raises the contrast by more than 5 standard errors", mem_shift_neg > 5 * se_con)
    chk.true("T1 = 100 us moves the contrast by less than a fifth of the 1 percent memory shift",
             abs(t1_shift) < 0.2 * min(abs(mem_shift_pos), abs(mem_shift_neg)))
    asym = "QM, leakage + asymmetric readout (e0 0.8%, e1 1.6%)"
    asym_shift = con[asym][0] - con["QM, leakage"][0]
    chk.true("static readout asymmetry moves the contrast at least half as much as 1 percent memory, "
             "so the null must use the qubit's own assignment errors",
             abs(asym_shift) > 0.5 * abs(mem_shift_pos))
    # Slopes of each null statistic per unit assignment error (clean mask), for
    # propagating the AS calibration's standard errors into the reading.
    sens = {}
    for key in NULL_KEYS:
        v0 = _pair(base, "clean", key)[0]
        sens[key] = ((_pair(kept["QM, leakage + readout e0 1.6%"], "clean", key)[0] - v0) / 0.004,
                     (_pair(kept["QM, leakage + readout e1 1.6%"], "clean", key)[0] - v0) / 0.004)
    # The per-qubit null and the reading, end to end on simulated qubits.
    null_t1 = qubit_null(0.012, 0.012, T1_S, _num(t1case, "fraction_checks_leaked"), seed=7)
    for key in ("repeat_agree", "flip_agree", "contrast"):
        v, se = _pair(t1case, "clean", key)
        nv, ns = null_t1["clean"][key]  # type: ignore[index]
        chk.within(f"qubit_null reproduces the T1 case's clean {key}", v, math.hypot(se, ns), nv)
    chk.true("qubit_null matches the leaked-check fraction it was given within 10 percent",
             abs(null_t1["inputs"]["leaked_fraction_null"] / _num(t1case, "fraction_checks_leaked") - 1) < 0.1)  # type: ignore[index]
    rd_null = eigen_reading(t1case, null_t1, sens, (5e-4, 5e-4))
    chk.true("the reading calls the conventional T1 case consistent with its null",
             rd_null["contrast_reading"] == "consistent with the per-qubit conventional null")
    null_t1_off = qubit_null(0.012, 0.012, 1.5 * T1_S, _num(t1case, "fraction_checks_leaked"), seed=8)
    rd_off = eigen_reading(t1case, null_t1_off, sens, (5e-4, 5e-4))
    chk.true("a 50 percent T1 calibration error leaves the contrast reading consistent",
             rd_off["contrast_reading"] == "consistent with the per-qubit conventional null")
    readings = {}
    for name, want in (("QM, leakage + readout memory +1%", "positive memory on eigenstate readouts"),
                       ("QM, leakage + readout memory -1%", "negative memory on eigenstate readouts")):
        nl = qubit_null(0.012, 0.012, None, _num(kept[name], "fraction_checks_leaked"), seed=9)
        rd = eigen_reading(kept[name], nl, sens, (5e-4, 5e-4))
        readings[name] = {"contrast_z_clean": round(rd["clean"]["contrast"]["z"], 2),  # type: ignore[index]
                          "contrast_z_clean_next": round(rd["clean_next"]["contrast"]["z"], 2),  # type: ignore[index]
                          "reading": rd["contrast_reading"]}
        chk.true(f"the reading detects {name} with its sign, its null tuned to the case's own leaked fraction",
                 rd["contrast_reading"] == want)
    readings["QM, leakage + T1 relaxation"] = {
        "contrast_z_clean": round(rd_null["clean"]["contrast"]["z"], 2),  # type: ignore[index]
        "contrast_z_clean_next": round(rd_null["clean_next"]["contrast"]["z"], 2),  # type: ignore[index]
        "reading": rd_null["contrast_reading"]}
    readings["QM, leakage + T1 relaxation, null at 1.5 x T1"] = {
        "contrast_z_clean": round(rd_off["clean"]["contrast"]["z"], 2),  # type: ignore[index]
        "contrast_z_clean_next": round(rd_off["clean_next"]["contrast"]["z"], 2),  # type: ignore[index]
        "reading": rd_off["contrast_reading"]}

    b = budget(clean_fraction=_num(base, "fraction_blocks_clean"))
    # Reach of the contrast in the planned run: the AS calibration's errors,
    # propagated through the slopes, against the 1 percent memory shift.
    as_row = next(c for c in b["conditions"] if c["label"] == "AS")
    per_prep = as_row["sets"] * 400 // 2
    e_se = math.sqrt(0.012 * 0.988 / per_prep)
    cal_se = math.hypot(sens["contrast"][0] * e_se, sens["contrast"][1] * e_se)
    per_pct = min(abs(mem_shift_pos), abs(mem_shift_neg))
    reach = {"as_shots_per_preparation_per_qubit": per_prep, "assignment_error_se_at_1.2pct": round(e_se, 6),
             "contrast_null_se_from_calibration": round(cal_se, 6),
             "contrast_shift_per_1pct_memory": round(per_pct, 5),
             "memory_at_K_SIGMA_from_calibration_alone_pct": round(K_SIGMA * cal_se / per_pct, 3)}
    chk.true("the calibration alone resolves memory below 0.5 percent at K_SIGMA",
             reach["memory_at_K_SIGMA_from_calibration_alone_pct"] < 0.5)
    chk.true("fits one 600 s free window", bool(b["fits_600s_window"]))

    jobs = b["jobs"]
    chk.true("every job is at most 60 s estimated",
             all(float(j["estimated_qpu_s"]) <= submit.MAX_JOB_S for j in jobs))
    chk.true("the longest job leaves margin under the cap", float(b["max_job_qpu_s"]) < 0.95 * submit.MAX_JOB_S)
    chk.true("E0 is split across several jobs, not packed into one",
             int(b["conditions"][0]["pub_parts"]) >= 5 and int(b["n_jobs"]) >= 6)
    c2 = [j for j in jobs if "C2" in j["entries"]]
    as_jobs = [j for j in jobs if "AS" in j["entries"]]
    chk.true("the phase and amplitude checks share one job, the assignment calibration runs alone "
             "and every other job is a lone E0 part",
             len(c2) == 1 and len(c2[0]["entries"]) == 1 + len(design.AMPLITUDE_PULSES)
             and len(as_jobs) == 1 and as_jobs[0]["entries"] == ["AS"]
             and all(set(j["entries"]) == {"E0"} for j in jobs if j is not c2[0] and j is not as_jobs[0]))
    chk.true("every job stays under the 60 s cap at the worst October overhead",
             float(b["max_job_at_worst_october_overhead_s"]) < submit.MAX_JOB_S)
    chk.true("the window fits at the worst October overhead", float(b["spare_at_worst_october_overhead_s"]) > 0)
    chk.true("the controls are October fez qubits and none of them is excluded",
             set(EIG_CONTROLS) <= set(followup.FEZ_LAYOUT) and not set(EIG_CONTROLS) & set(EIG_EXCLUDE))
    chk.true("every other October fez qubit is excluded",
             set(EIG_CONTROLS) | set(EIG_EXCLUDE) == set(followup.FEZ_LAYOUT))
    # The AS estimator recovers known assignment errors.
    rng_as = np.random.default_rng(5)
    a_, b_, ph = assign_settings(200, 8, rng_as)
    ph = ph.reshape(200, 8)
    truth = (0.01, 0.02)
    sim = np.where(np.cos(ph)[:, :, None] < 0,
                   np.where(rng_as.random((200, 8, 400)) < truth[0], -1, 1),
                   np.where(rng_as.random((200, 8, 400)) < truth[1], 1, -1)).astype(np.int8)
    for q in (0, 1):
        ae = assign_errors(sim[:, q, :], ph[:, q])
        chk.within(f"assign_errors recovers e0 on qubit {q}", *ae["e0"], truth[0])
        chk.within(f"assign_errors recovers e1 on qubit {q}", *ae["e1"], truth[1])
        chk.true(f"assign_errors standard error under 1e-3 on qubit {q}", ae["e1"][1] < 1e-3)
    # The repeat E0R must pass the submit script's usage check in the 185 s left.
    rb = budget(conditions=EIG_REPEAT_CONDITIONS)
    chk.true("repeat: every job under the per-job cap", rb["max_job_qpu_s"] < submit.MAX_JOB_S)
    chk.true("repeat: estimate with overhead under the usage margin of the 185 s left",
             rb["total_with_overhead_s"] < submit.USAGE_MARGIN * REPEAT_WINDOW_S)
    chk.true("repeat: charge at the worst October overhead under the 185 s left",
             rb["total_at_worst_october_overhead_s"] < REPEAT_WINDOW_S)
    chk.true("repeat: same chain as E0", EIG_REPEAT_CONDITIONS[0][:4] == EIG_CONDITIONS[0][:4]
             and EIG_REPEAT_CONDITIONS[1:] == EIG_CONDITIONS[1:])
    chk.true("repeat: same eight qubits, controls first", len(EIG_REPEAT_LAYOUT) == EIG_QUBITS
             and EIG_REPEAT_LAYOUT[:3] == EIG_CONTROLS)
    t1row = table["QM, leakage + T1 relaxation"]
    result = {"status": "PASS", "checks": chk.count, "simulated_cases": table,
              "null_reference": {
                  "case": "QM, leakage + T1 relaxation",
                  "meaning": "the conventional null (leakage plus T1 relaxation, no readout memory), "
                             "on the same masks the fidelities are reported on",
                  "clean": {k: t1row["clean"][k] for k in ("repeat_agree", "flip_agree", "D_flip", "B_lag1")},
                  "clean_next": {k: t1row["clean_next"][k] for k in ("repeat_agree", "flip_agree", "D_flip", "B_lag1")},
                  "all": {k: t1row["all"][k] for k in ("repeat_agree", "flip_agree", "flip_control", "B_lag1")},
                  "fraction_checks_leaked": t1row["fraction_checks_leaked"],
                  "fraction_blocks_clean": t1row["fraction_blocks_clean"],
                  "reading": "Design reference only. A run's null is built per qubit by qubit_null from its "
                             "own AS assignment errors, run-day T1 and leaked-check fraction, and read by "
                             "eigen_reading: the contrast flip_agree_1 - repeat_agree_1 deviating by more than "
                             f"{K_SIGMA} combined standard errors on both clean and clean_next, with one sign, "
                             "is assignment-like memory on eigenstate readouts (negative deviation: positive "
                             "memory). Absolute fidelities are secondary and need K_SIGMA plus the difference "
                             "between measured and null selection shifts.",
              },
              "sensitivity": {k: [round(v[0], 4), round(v[1], 4)] for k, v in sens.items()},
              "sensitivity_meaning": "change of each clean-mask statistic per unit e0 and per unit e1 "
                                     "(finite differences at e = 0.012 + 0.004); the analysis uses them to "
                                     "propagate the AS calibration's standard errors into the reading",
              "readings_on_simulated_cases": readings,
              "contrast_reach": reach,
              "sign_convention": {
                  "contrast": "negative deviation from the null is positive memory (a reported 1 makes "
                              "the next readout of |0> report 1), positive deviation is negative memory",
                  "D_flip": "magnitude only: the ideal value sits at the -1 floor, so either sign of "
                            "readout memory moves it the same way",
                  "repeat_agree": "carries the sign, as does the sign of the B lag-1 covariance",
              },
              "budget": b,
              "repeat_budget": {k: rb[k] for k in ("conditions", "jobs", "n_jobs", "max_job_qpu_s", "total_qpu_s",
                                                   "total_with_overhead_s", "total_at_worst_october_overhead_s",
                                                   "max_job_at_worst_october_overhead_s")}
              | {"window_left_s": REPEAT_WINDOW_S, "usage_margin_s": submit.USAGE_MARGIN * REPEAT_WINDOW_S,
                 "layout": list(EIG_REPEAT_LAYOUT)},
              "wiring": {
                  "circuits": "draw_settings(kind='tagged_eig', ...) via tagged_eig_settings; chain_circuit "
                              "(K, n_qubits, delay_dt, use_measure_2) unchanged",
                  "submit": "cloud_run_submit.py --job eig --backend ibm_fez --max-job-s: conditions = "
                            "EIG_CONDITIONS, split by job_plan so no job exceeds the cap; --submit needs "
                            "author go-ahead",
                  "analyze": "save_job() unchanged (alpha/beta/phi stored); analyze() has a tagged_eig branch "
                             "with eigen_statistics() per qubit and pool_eigen() over qubits (the unweighted "
                             "mean of per-qubit values, as the October reports pool); report.json stays the source",
              },
              "note": "Simulated design checks; nothing here is experimental data."}
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    print(f"PASS: {chk.count} checks")
    print(json.dumps(b, indent=1))
    return 0


def load_sensitivities(path: str | None = None) -> Dict[str, Tuple[float, float]]:
    """The saved per-unit-error slopes of this script's output (analysis/cloud_run_eigenstate.json)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_run_eigenstate.json")
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    return {k: (float(v[0]), float(v[1])) for k, v in saved["sensitivity"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Eigenstate repeat/flip extension checks")
    parser.add_argument("--output", help="JSON output, e.g. analysis/cloud_run_eigenstate.json")
    args = parser.parse_args()
    try:
        return run(args.output)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
