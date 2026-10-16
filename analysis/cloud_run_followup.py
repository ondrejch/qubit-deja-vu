#!/usr/bin/env python3
"""Leak-tagged follow-up to the IBM cloud run (notes/CLOUD_RUN_FOLLOWUP_DESIGN.md).

The 15 October 2026 run left a neighbor covariance that a conventional
leakage model explains only in part (notes/CLOUD_RUN_RESULTS.md). The
follow-up chain tags leakage directly. Records come in blocks of six,

  B B B B F F

where B is a balanced step (beta - alpha = pi/2, or 3 pi/2 to relabel; the
four alternate S, S^-1 with one frame offset per block) and F a flip step
(beta = alpha, a pi rotation). After the fourth B readout x, the two flips
give -x and x. A leaked qubit, which the readout reports as "1" and which no
pulse moves, gives "1, 1, 1" instead. A block's B records are clean when the
check that ends it and the check that ends the block before it both show the
normal pattern.

This module holds the chain settings, a three-level simulation (quantum
mechanics with leakage, readout error, pulse amplitude error and readout
memory of either sign, and the detector-anchored completion with the same
leakage), the tagged statistics used by analysis/cloud_run_analyze.py, and
design checks:

  python analysis/cloud_run_followup.py --output analysis/cloud_run_followup.json

The checks confirm that tagging removes a leakage background, that selecting
clean blocks leaves a detector-anchored persistence w unbiased (the checks
start from eigenstates, so their outcomes do not depend on the detector
variable), and that readout memory survives selection. NumPy only; nothing
here contacts IBM, and nothing here is experimental data.
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

BLOCK = 6
B_POS = (0, 1, 2, 3)
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- settings

def tagged_settings(n: int, K: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """alpha, beta, phi (n, K) for B B B B F F blocks; K a multiple of 6."""
    if K % BLOCK:
        raise ValueError("records per shot must be a multiple of 6")
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
            else:
                alpha[:, k] = rng.uniform(0.0, TWO_PI, n)
                phi[:, k] = 0.0
    return alpha, alpha + phi, np.mod(phi, TWO_PI)


# ---------------------------------------------------------------- simulation

def simulate_leaky(model: str, phi: np.ndarray, rng, w: float = 0.0, eps: float = 0.0,
                   e0: float = 0.0, e1: float = 0.0, g0: float = 0.0, g1: float = 0.0,
                   g_dn: float = 0.0, q2: float = 1.0, mem_pos: float = 0.0,
                   mem_neg: float = 0.0) -> np.ndarray:
    """Reported outcomes (n, K), +1 = |0>, from |0>.

    Levels 0, 1 (computational) and 2 (leaked). Before each step a leaked qubit
    returns to |1> with probability g_dn. The step leaves <Z> = b S_zz(phi)
    with S_zz = cos^2 t - sin^2 t cos(phi), t = (pi/2)(1 + eps), and leaves |2>
    alone. During the readout |0> and |1> leak with probabilities g0 and g1.
    The readout reports "1" from |0> with probability e0 (plus mem_pos after a
    reported "1"), "0" from |1> with probability e1 (plus mem_neg after a
    reported "1"), and "1" from |2> with probability q2. model "D" draws the
    computational outcome from the detector variable u with gap persistence w.
    """
    n, K = phi.shape
    t = 0.5 * math.pi * (1.0 + eps)
    level = np.zeros(n, int)
    u = rng.random(n)
    r_prev = np.ones(n)
    out = np.empty((n, K), np.int8)
    for k in range(K):
        if model == "D" and k > 0:
            redraw = rng.random(n) >= w
            u[redraw] = rng.random(int(redraw.sum()))
        level = np.where((level == 2) & (rng.random(n) < g_dn), 1, level)
        b = np.where(level == 0, 1.0, -1.0)
        sz = b * (math.cos(t) ** 2 - math.sin(t) ** 2 * np.cos(phi[:, k]))
        draw = u if model == "D" else rng.random(n)
        comp = np.where(draw < 0.5 * (1.0 + sz), 0, 1)
        level = np.where(level < 2, comp, level)
        x = rng.random(n)
        level = np.where(((level == 0) & (x < g0)) | ((level == 1) & (x < g1)), 2, level)
        after1 = r_prev < 0
        p1 = np.where(level == 0, e0 + mem_pos * after1,
                      np.where(level == 1, 1.0 - (e1 + mem_neg * after1), q2))
        r = np.where(rng.random(n) < p1, -1, 1).astype(np.int8)
        out[:, k] = r
        r_prev = r
    return out


# ---------------------------------------------------------------- statistics

def check_states(r: np.ndarray) -> np.ndarray:
    """r (..., K) -> check outcome per block (..., K // 6): 0 normal, 1 leaked, 2 other."""
    K = r.shape[-1]
    blocks = r.reshape(r.shape[:-1] + (K // BLOCK, BLOCK)).astype(np.int16)
    x, f1, f2 = blocks[..., 3], blocks[..., 4], blocks[..., 5]
    normal = (f1 == -x) & (f2 == x)
    leaked = (x == -1) & (f1 == -1) & (f2 == -1)
    return np.where(normal, 0, np.where(leaked, 1, 2))


def clean_blocks(states: np.ndarray) -> np.ndarray:
    """A block is clean when its own check and the previous block's check are normal."""
    ok = states == 0
    prev = np.concatenate([np.ones(ok.shape[:-1] + (1,), bool), ok[..., :-1]], axis=-1)
    return ok & prev


def _masked_cov(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> Tuple[float, float]:
    x = x[m].astype(np.float64)
    y = y[m].astype(np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    prod = (x - x.mean()) * (y - y.mean())
    return float(prod.mean()), float(prod.std() / math.sqrt(prod.size))


def tagged_statistics(r: np.ndarray) -> Dict[str, object]:
    """r (N, K) outcomes of one qubit, rows are shots (any order)."""
    N, K = r.shape
    nb = K // BLOCK
    st = check_states(r)
    clean = clean_blocks(st)
    blocks = r.reshape(N, nb, BLOCK)
    out: Dict[str, object] = {}
    for name, mask in (("clean", clean), ("all", np.ones_like(clean))):
        res = {}
        for lag in (1, 2, 3):
            xs = np.concatenate([blocks[:, :, j] for j in range(4 - lag)], axis=1)
            ys = np.concatenate([blocks[:, :, j + lag] for j in range(4 - lag)], axis=1)
            ms = np.concatenate([mask] * (4 - lag), axis=1)
            res[f"lag{lag}"] = _masked_cov(xs, ys, ms)
        if nb > 1:
            both = mask[:, :-1] & mask[:, 1:]
            xs = np.concatenate([blocks[:, :-1, j] for j in B_POS], axis=1)
            ys = np.concatenate([blocks[:, 1:, j] for j in B_POS], axis=1)
            ms = np.concatenate([both] * 4, axis=1)
            res["cross_block"] = _masked_cov(xs, ys, ms)
        out[name] = res
    leaked = st == 1
    out["fraction_checks_leaked"] = float(leaked.mean())
    out["fraction_checks_other"] = float((st == 2).mean())
    out["fraction_blocks_clean"] = float(clean.mean())
    out["fraction_first_check_leaked"] = float(leaked[:, 0].mean())
    if nb > 1:
        prev = leaked[:, :-1]
        stay = float(leaked[:, 1:][prev].mean()) if prev.any() else float("nan")
        start = float(leaked[:, 1:][~prev].mean())
        out["p_leaked_next_check_given_leaked"] = stay
        out["p_leaked_next_check_given_not"] = start
    if nb > 2:
        # Two flagged checks in a row are almost never both readout errors, so
        # this conditional estimates the per-block persistence of a real leak.
        two = leaked[:, :-2] & leaked[:, 1:-1]
        out["p_leaked_next_check_given_two_leaked"] = float(leaked[:, 2:][two].mean()) if two.any() else float("nan")
    return out


# ---------------------------------------------------------------- plan and checks

# Follow-up job: label, kind, added delay before each step (s), records per shot, QPU seconds.
FOLLOWUP_CONDITIONS = [("T0", "tagged", 0.0, 66, 60.0), ("T1", "tagged", 4e-6, 66, 60.0),
                       ("T2", "tagged", 40e-6, 18, 40.0), ("C2", "phase", 0.0, 1, 3.2)]
LAYOUT = [21, 154, 11, 149, 109, 89]  # the qubits of the 15 October 2026 run
# Add-on (design Section 9): one leak-tagged condition on ibm_fez, sized to the
# last 20 s of the 28-day allowance; one job, with the amplitude check.
FEZ_MINI_CONDITIONS = [("F0", "tagged", 0.0, 66, 12.0)]
# Add-on 2 (design Section 10): the F0 qubits again at two spacings, about one
# paid minute in one job, without the amplitude check.
FEZ_LAYOUT = [124, 26, 86, 144, 2, 6]
FEZ_REPEAT_CONDITIONS = [("G0", "tagged", 0.0, 66, 27.0), ("G1", "tagged", 4e-6, 66, 27.0)]


def budget(qubits: int = 6, rep_delay: float = 250e-6, readout: float = 1.38e-6, sx: float = 32e-9,
           overhead_per_job: float = 2.5) -> Dict[str, object]:
    rows, total = [], 0.0
    for label, kind, delay, K, seconds in FOLLOWUP_CONDITIONS:
        period = 2 * sx + readout + delay
        t_exec = rep_delay + K * period
        execs = seconds / t_exec
        row = {"label": label, "records": K, "period_us": round(period * 1e6, 3), "qpu_s": seconds,
               "executions": int(execs)}
        if kind == "tagged":
            pairs = execs * (K // BLOCK) * 3 * qubits
            row["within_block_neighbor_pairs"] = f"{pairs:.2e}"
            row["five_sigma_pooled_if_all_clean"] = round(5.0 / math.sqrt(pairs), 5)
            row["five_sigma_per_qubit_if_all_clean"] = round(5.0 / math.sqrt(pairs / qubits), 5)
        rows.append(row)
        total += seconds
    amplitude_s = 5 * 400 * (rep_delay + 100 * sx + readout)
    jobs = 1 + sum(1 for c in FOLLOWUP_CONDITIONS if c[1] == "tagged")
    total += amplitude_s + jobs * overhead_per_job
    return {"conditions": rows, "amplitude_check_s": round(amplitude_s, 2), "jobs": jobs,
            "total_qpu_s_with_overheads": round(total, 1)}


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


# Leakage roughly as fitted to the 15 October 2026 records (qubit 149), made
# larger so the design checks resolve it with modest samples.
LEAK = dict(e0=0.012, e1=0.012, g0=0.002, g1=0.002, g_dn=0.015, q2=0.99)


def run(output: str | None) -> int:
    rng = np.random.default_rng(20261016)
    chk = Check()
    n, K = 120_000, 66
    _, _, phi = tagged_settings(n, K, rng)
    chk.true("block layout: four balanced steps then two flips",
             np.allclose(np.cos(phi[:, 4::6]), 1.0) and np.allclose(np.cos(phi[:, 0::6]), 0.0, atol=1e-12))
    table = {}
    cases = [
        ("QM, ideal", "QM", 0.0, {}),
        ("QM, leakage", "QM", 0.0, LEAK),
        ("QM, leakage and readout memory +1%", "QM", 0.0, {**LEAK, "mem_pos": 0.01}),
        ("QM, leakage and readout memory -1%", "QM", 0.0, {**LEAK, "mem_neg": 0.01}),
        ("D w=0.3, leakage", "D", 0.3, LEAK),
        ("D w=0.05, no leakage", "D", 0.05, {}),
    ]
    for name, model, w, kw in cases:
        r = simulate_leaky(model, phi, rng, w=w, **kw)
        st = tagged_statistics(r)
        c, a = st["clean"], st["all"]
        table[name] = {"clean": {k: [round(v[0], 5), round(v[1], 5)] for k, v in c.items()},
                       "all": {k: [round(v[0], 5), round(v[1], 5)] for k, v in a.items()},
                       "fraction_checks_leaked": round(st["fraction_checks_leaked"], 5),
                       "fraction_blocks_clean": round(st["fraction_blocks_clean"], 4)}
        if "g1" in kw:
            chk.true(f"{name}: leakage raises the unselected cross-block covariance",
                     a["cross_block"][0] > 5 * a["cross_block"][1])
            chk.within(f"{name}: clean cross-block covariance removed", *c["cross_block"], 0.0)
            chk.true(f"{name}: leaked checks detected", st["fraction_checks_leaked"] > 0.005)
        contrast = (1.0 - kw.get("e0", 0.0) - kw.get("e1", 0.0)) ** 2   # readout error scales covariances
        want1 = w * contrast if model == "D" else 0.0
        if "mem_pos" in kw:
            chk.true(f"{name}: positive readout memory survives selection", c["lag1"][0] > 5 * c["lag1"][1])
        elif "mem_neg" in kw:
            chk.true(f"{name}: negative readout memory survives selection", c["lag1"][0] < -5 * c["lag1"][1])
        else:
            chk.within(f"{name}: clean lag-1 covariance", *c["lag1"], want1)
            chk.within(f"{name}: clean lag-2 covariance", *c["lag2"], w * w * contrast if model == "D" else 0.0)
        if model == "D":
            chk.within(f"{name}: clean cross-block covariance", *c["cross_block"], w ** 6 * contrast)
    # Leaked-episode persistence across one block: (1 - g_dn)^6 if no return to the
    # computational levels went unnoticed; the estimate brackets it.
    r = simulate_leaky("QM", phi, rng, **LEAK)
    st = tagged_statistics(r)
    stay_true = (1.0 - LEAK["g_dn"]) ** BLOCK
    chk.true("persistence after two leaked checks near (1 - g_dn)^6 q2^3",
             abs(st["p_leaked_next_check_given_two_leaked"] - stay_true * LEAK["q2"] ** 3) < 0.03)
    result = {"status": "PASS", "checks": chk.count, "simulated_cases": table,
              "leak_persistence": {"estimated_after_two_leaked": round(st["p_leaked_next_check_given_two_leaked"], 4),
                                   "estimated_after_one_leaked": round(st["p_leaked_next_check_given_leaked"], 4),
                                   "model": round(stay_true * LEAK["q2"] ** 3, 4)},
              "budget": budget(), "layout": LAYOUT,
              "note": "Simulated design checks for the follow-up; nothing here is experimental data."}
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    print(f"PASS: {chk.count} checks")
    print(json.dumps(result["budget"], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        return run(args.output)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
