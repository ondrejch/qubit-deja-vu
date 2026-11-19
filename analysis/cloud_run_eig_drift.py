#!/usr/bin/env python3
"""Exploratory: test the E0 null's first-readout calibration against in-chain readouts.

For the E0 run of 19 November 2026 this is exploratory, chosen after its data.
For the repeat E0R (notes/CLOUD_RUN_NEXT_DESIGN.md Section 2H) it is the
pre-registered primary reading, committed before that run's data.

Chosen after seeing the 19 November 2026 E0 data (notes/CLOUD_RUN_RESULTS.md
Section 10.4, limit 2); not pre-registered. The pre-registered null takes each
qubit's static assignment errors e0, e1 from AS, the first readout of a shot.
The contrast flip_agree_1 - repeat_agree_1 depends on them (about -1.9 per unit
e0 and +1.0 per unit e1), so a readout that behaves differently inside the
chain moves it as memory would.

In the chain simulator (cloud_run_followup.simulate_leaky) memory acts only
after a reported "1", and T1 relaxation acts only on a held |1>. Two of the
four eigenstate fidelities are then free of both:

    repeat_agree_0  F2 read "0", R repeats |0>     depends on e0 (and e1 weakly)
    flip_agree_0    R read "0", P flips to a fresh |1>  depends on e1 (and e0 weakly)

so they give the in-chain static errors de0, de1 relative to AS. The contrast
deviation is then recomputed against a null with those in-chain errors
(T1 cancels in the contrast to first order). Derivatives come from per-qubit
simulations with common random numbers around the report's null inputs; data
errors from a bootstrap over parameter sets. The same estimates per block
(9 per shot, block 0 starting at record 0) test drift along the chain.

Attribution is conditional on that memory model. Memory that also acts after a
reported "0" would shift the state-0 fidelities as well, and this run cannot
tell such memory from an in-chain change of the static errors. A recovery
check on a synthetic chain with shifted static errors, positive memory and a
T1 mismatch runs first; the script exits nonzero if it fails.

  python3 analysis/cloud_run_eig_drift.py data/cloudrun/2026-11-19_eig \\
      --output data/cloudrun/2026-11-19_eig/drift_exploratory.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cloud_run_eigenstate as eig  # noqa: E402
import cloud_run_followup as followup  # noqa: E402

RATES = ("repeat_agree_0", "flip_agree_0", "repeat_agree_1", "flip_agree_1")
STATIC = ("de0", "de1")          # in-chain static-error shifts relative to AS
STEP = 0.003                     # finite-difference step for the derivatives
SIM_SHOTS = 300_000
BOOT = 400
MASKS = ("clean", "clean_next")
K_SIGMA = eig.K_SIGMA


def rate_counts(r: np.ndarray, mask: str) -> dict:
    """r (N, 72) -> for each rate, (agree, conditioned) boolean arrays of shape (N, 9) on the mask."""
    m = eig.block_masks(r)[mask]
    b = r.reshape(r.shape[0], -1, eig.BLOCK)
    f2, rr, p = b[..., eig.F2], b[..., eig.R_POS], b[..., eig.P_POS]
    out = {}
    for name, cond, agree in (("repeat_agree_0", f2 == 1, rr == f2), ("repeat_agree_1", f2 == -1, rr == f2),
                              ("flip_agree_0", rr == 1, p == -rr), ("flip_agree_1", rr == -1, p == -rr)):
        den = m & cond
        out[name] = (den & agree, den)
    return out


def rates(counts: dict, axis=None) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array([counts[k][0].sum(axis=axis) / counts[k][1].sum(axis=axis) for k in RATES])


def simulate(inputs: dict, n: int, seed: int, de0: float = 0.0, de1: float = 0.0, m_pos: float = 0.0,
             m_neg: float = 0.0, t1_s: float | None = None) -> np.ndarray:
    """One qubit's conventional chain from the report's null inputs, with optional shifts.

    The same seed gives common random numbers, so finite differences are precise.
    """
    rng = np.random.default_rng(seed)
    _, _, phi = eig.tagged_eig_settings(n, 72, rng)
    t1 = inputs["t1_s"] if t1_s is None else t1_s
    return followup.simulate_leaky(
        "QM", phi, rng, e0=inputs["e0"] + de0, e1=inputs["e1"] + de1,
        g0=inputs["leak_rate"], g1=inputs["leak_rate"], g_dn=inputs["g_dn"], q2=inputs["q2"],
        mem_pos=m_pos, mem_neg=m_neg, t1=t1, period_s=eig.FEZ_PERIOD_S if t1 else None)


def model(inputs: dict, seed: int, n: int = SIM_SHOTS) -> dict:
    """Null rates (overall and per block) and their derivatives in de0, de1, per mask."""
    base = simulate(inputs, n, seed)
    pert = {u: simulate(inputs, n, seed, **{u: STEP}) for u in STATIC}
    out = {}
    for mask in MASKS:
        cb = rate_counts(base, mask)
        r0 = rates(cb)
        den = np.array([cb[k][1].sum() for k in RATES])
        jac = np.column_stack([(rates(rate_counts(pert[u], mask)) - r0) / STEP for u in STATIC])  # (4, 2)
        blk = rates(cb, axis=0)
        den_b = np.array([cb[k][1].sum(axis=0) for k in RATES])
        out[mask] = {"rates": r0, "se": np.sqrt(r0 * (1 - r0) / den), "jac": jac,
                     "block_rates": blk, "block_se": np.sqrt(blk * (1 - blk) / den_b)}
    return out


# Linear maps from the four rate deviations to (de0, de1, corrected contrast deviation).
def _maps(jac: np.ndarray) -> np.ndarray:
    """3 x 4 matrix L with (de0, de1, dc) = L @ delta, delta = data - null for RATES."""
    a = jac[[0, 1], :]                  # repeat_agree_0, flip_agree_0 rows vs (de0, de1)
    ainv = np.linalg.inv(a)
    sel = np.zeros((2, 4))
    sel[0, 0] = sel[1, 1] = 1.0
    static = ainv @ sel                 # (de0, de1) from the two state-0 rates
    c_row = np.array([0.0, 0.0, -1.0, 1.0])          # contrast = flip_agree_1 - repeat_agree_1
    c_jac = jac[3, :] - jac[2, :]                       # contrast derivative in (de0, de1)
    corrected = c_row - c_jac @ static
    return np.vstack([static, corrected])


def estimate(data: np.ndarray, cov: np.ndarray, mod: dict) -> dict:
    L = _maps(mod["jac"])
    x = L @ (data - mod["rates"])
    xc = L @ (cov + np.diag(mod["se"] ** 2)) @ L.T
    se = np.sqrt(np.diag(xc))
    raw_c = (data[3] - data[2]) - (mod["rates"][3] - mod["rates"][2])
    return {"de0": [float(x[0]), float(se[0])], "de1": [float(x[1]), float(se[1])],
            "contrast_deviation_in_chain_null": [float(x[2]), float(se[2])],
            "z_in_chain_null": float(x[2] / se[2]),
            "contrast_deviation_first_readout_null": float(raw_c)}


def decompose(r_sets: np.ndarray, mod: dict, rng) -> dict:
    """r_sets (sets, shots, 72) of one qubit."""
    sets, shots, K = r_sets.shape
    out = {}
    for mask in MASKS:
        c = rate_counts(r_sets.reshape(-1, K), mask)
        per_set = {k: (c[k][0].reshape(sets, -1).sum(1), c[k][1].reshape(sets, -1).sum(1)) for k in RATES}
        data = np.array([per_set[k][0].sum() / per_set[k][1].sum() for k in RATES])
        boot = np.empty((BOOT, 4))
        for b in range(BOOT):
            idx = rng.integers(0, sets, sets)
            boot[b] = [per_set[k][0][idx].sum() / per_set[k][1][idx].sum() for k in RATES]
        m = mod[mask]
        res = estimate(data, np.cov(boot.T), m)
        res["data_rates"] = dict(zip(RATES, map(float, data)))
        res["null_rates"] = dict(zip(RATES, map(float, m["rates"])))
        # Per block: same estimates, then a weighted straight line in block index (drift along the chain).
        blk = rates(c, axis=0)
        blk_se = np.sqrt(blk * (1 - blk) / np.array([c[k][1].sum(axis=0) for k in RATES]))
        L = _maps(m["jac"])
        bx, bse = [], []
        for j in range(blk.shape[1]):
            xj = L @ (blk[:, j] - m["block_rates"][:, j])
            cj = L @ np.diag(blk_se[:, j] ** 2 + m["block_se"][:, j] ** 2) @ L.T
            bx.append(xj)
            bse.append(np.sqrt(np.diag(cj)))
        bx, bse = np.array(bx), np.array(bse)
        idx = np.arange(bx.shape[0], dtype=float)
        trend = {}
        for i, name in enumerate(("de0", "de1", "contrast_deviation_in_chain_null")):
            w = 1.0 / bse[:, i] ** 2
            xm = np.sum(w * idx) / w.sum()
            slope = float(np.sum(w * (idx - xm) * bx[:, i]) / np.sum(w * (idx - xm) ** 2))
            se = float(1.0 / math.sqrt(np.sum(w * (idx - xm) ** 2)))
            trend[name] = {"slope_per_block": slope, "se": se, "z": slope / se,
                           "per_block": [[round(float(v), 6), round(float(s), 6)] for v, s in zip(bx[:, i], bse[:, i])]}
        res["trend"] = trend
        out[mask] = res
    both = [out[mk]["z_in_chain_null"] for mk in MASKS]
    if all(z > K_SIGMA for z in both):
        out["reading_in_chain_null"] = "negative memory on eigenstate readouts"
    elif all(z < -K_SIGMA for z in both):
        out["reading_in_chain_null"] = "positive memory on eigenstate readouts"
    else:
        out["reading_in_chain_null"] = "consistent with the in-chain null"
    return out


def recovery_check() -> dict:
    """Synthetic qubit: in-chain static errors shifted from the calibration, +0.4 percent positive
    memory and a T1 shorter than the null assumes. The estimates must recover the shifts and the
    memory's contrast effect, and show no trend along the chain."""
    inputs = {"e0": 0.003, "e1": 0.009, "t1_s": 120e-6, "leak_rate": 4e-5, "g_dn": 0.015, "q2": 0.99}
    truth = {"de0": 0.001, "de1": -0.004}
    sets, shots, mem = 1000, 400, 0.004
    syn = simulate(inputs, sets * shots, 777, m_pos=mem, t1_s=70e-6, **truth).reshape(sets, shots, 72)
    mod = model(inputs, 20261120)
    # The contrast effect of the memory alone, with the true static errors and T1 in the null.
    ref = model({**inputs, "e0": inputs["e0"] + truth["de0"], "e1": inputs["e1"] + truth["de1"]}, 20261121)
    with_mem = simulate({**inputs, "e0": inputs["e0"] + truth["de0"], "e1": inputs["e1"] + truth["de1"]},
                        SIM_SHOTS, 20261121, m_pos=mem)
    c_mem = rate_counts(with_mem, "clean")
    rm = rates(c_mem)
    want_dc = (rm[3] - rm[2]) - (ref["clean"]["rates"][3] - ref["clean"]["rates"][2])
    res = decompose(syn, mod, np.random.default_rng(5))
    est = res["clean"]
    for u, want in truth.items():
        v, s = est[u]
        if not abs(v - want) <= 4 * s + 2e-4:
            raise AssertionError(f"recovery: {u} = {v:.5f} +- {s:.5f}, want {want}")
    v, s = est["contrast_deviation_in_chain_null"]
    if not abs(v - want_dc) <= 4 * s + 5e-4:
        raise AssertionError(f"recovery: corrected contrast deviation {v:.5f} +- {s:.5f}, want {want_dc:.5f}")
    if res["reading_in_chain_null"] != "positive memory on eigenstate readouts":
        raise AssertionError(f"recovery: reading {res['reading_in_chain_null']}")
    for name, tr in est["trend"].items():
        if abs(tr["z"]) > 4:
            raise AssertionError(f"recovery: spurious trend in {name}, z = {tr['z']:.2f}")
    return {"truth": {**truth, "m_pos": mem, "t1_true_s": 70e-6, "t1_null_s": 120e-6,
                      "contrast_deviation_from_memory": round(float(want_dc), 6)},
            "estimate": {k: [round(est[k][0], 6), round(est[k][1], 6)]
                         for k in ("de0", "de1", "contrast_deviation_in_chain_null")},
            "first_readout_null_deviation": round(est["contrast_deviation_first_readout_null"], 6),
            "reading": res["reading_in_chain_null"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--output")
    args = ap.parse_args()
    try:
        check = recovery_check()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("recovery check passed", check, flush=True)
    with open(os.path.join(args.run, "report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    layout = report["layout"]
    nulls = report["conditions"]["E0"]["per_qubit_null"]
    with np.load(os.path.join(args.run, "E0.npz")) as z:
        r = z["r"]                                       # (sets, shots, qubits, 72)
    out = {"note": "Exploratory for the 19 November 2026 E0 run (chosen after its data); the pre-registered primary "
                   "reading for its repeat E0R (notes/CLOUD_RUN_NEXT_DESIGN.md Section 2H). de0, de1 are the "
                   "in-chain static assignment errors minus the AS (first-readout) values, from repeat_agree_0 and "
                   "flip_agree_0, which the chain model makes free of memory and T1. The contrast deviation is "
                   "recomputed against a null with those in-chain errors and read with the pre-registered "
                   "threshold on both masks. Attribution assumes memory acts only after a reported '1'.",
           "recovery_check": check, "settings": {"step": STEP, "sim_shots": SIM_SHOTS, "bootstrap": BOOT},
           "qubits": {}}
    rng = np.random.default_rng(20261119)
    for q, phys in enumerate(layout):
        inputs = nulls[q]["null"]["inputs"]
        res = decompose(r[:, :, q, :], model(inputs, 20261200 + q), rng)
        res["pre_registered_reading"] = nulls[q]["reading"]["contrast_reading"]
        out["qubits"][str(phys)] = res
        c, cn = res["clean"], res["clean_next"]
        print(phys, f"de0 {c['de0'][0]:+.5f}±{c['de0'][1]:.5f} de1 {c['de1'][0]:+.5f}±{c['de1'][1]:.5f}",
              f"dc first {c['contrast_deviation_first_readout_null']:+.5f} in-chain "
              f"{c['contrast_deviation_in_chain_null'][0]:+.5f}±{c['contrast_deviation_in_chain_null'][1]:.5f}",
              f"z {c['z_in_chain_null']:+.2f}/{cn['z_in_chain_null']:+.2f} | {res['reading_in_chain_null']}",
              "| trend z", " ".join(f"{c['trend'][k]['z']:+.1f}" for k in c["trend"]), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
