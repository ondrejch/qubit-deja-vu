#!/usr/bin/env python3
"""Heralded assignment calibration (AH): where does the first-readout offset come from?

notes/CLOUD_RUN_NEXT_DESIGN.md Section 2I. The 19 November E0 and E0R runs found
that the first readout of a shot misreads |1> 0.1 to 0.6 percentage points more
often than in-chain readouts of a freshly flipped |1> (CLOUD_RUN_RESULTS.md
Sections 10.5 and 11). AH reads each qubit twice per shot:

  record 0: phi = pi (stay in |0>), read: the herald
  record 1: phi = pi (stay, |0>) or phi = 0 (flip, |1>), alternating over qubits
            and sets as in AS, read

Among shots whose herald read "0", record 1 gives the heralded errors
e0_H = P("1" | |0>) and e1_H = P("0" | |1>), measured as the chain measures a
fresh flip. AS (one readout per shot, the first after reset) runs in the same
session. Two questions, fixed in advance:

  1. Does heralding remove the first-readout excess in e1?  d1 = e1_AS - e1_H.
  2. Is the excess an initialization population p (excited state left after
     reset)? p inflates the AS e0 and e1 alike, so initialization alone predicts
     d1 = d0 = e0_AS - e0_H; d1 - d0 > 0 means something else adds to e1.

This module holds the settings, the estimator, a simulation with an
initialization population and separate first-readout and in-chain errors, the
design checks, the budget, and the pre-registered evaluation. NumPy only.

  python3 analysis/cloud_run_herald.py --output analysis/cloud_run_herald.json
  python3 analysis/cloud_run_herald.py --evaluate data/cloudrun/<run> \\
      --inchain data/cloudrun/2026-11-19_eigrepeat/inchain_reading.json \\
      --output data/cloudrun/<run>/herald_reading.json
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

import cloud_run_submit as submit  # noqa: E402  (its module level imports no Qiskit)

TWO_PI = 2.0 * math.pi
K_SIGMA = 3.0
# label, kind, added delay before each step (s), records per shot, QPU seconds.
# No amplitude check: the run asks only about assignment errors.
HERALD_CONDITIONS = [("AH", "assign_herald", 0.0, 2, 14.0), ("AS", "assign", 0.0, 1, 12.0)]
HERALD_LAYOUT = (124, 144, 6, 50, 23, 59, 114, 132)  # the E0 and E0R qubits
WINDOW_LEFT_S = 49.0
FEZ_PERIOD_S = 2 * 24e-9 + 1.34e-6


def herald_settings(n_sets: int, n_qubits: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """alpha, beta, phi (n_sets * n_qubits, 2): record 0 stays in |0> (phi = pi);
    record 1 stays (phi = pi) where set + qubit is even, flips (phi = 0) where odd."""
    j, q = np.meshgrid(np.arange(n_sets), np.arange(n_qubits), indexing="ij")
    phi = np.empty((n_sets * n_qubits, 2))
    phi[:, 0] = math.pi
    phi[:, 1] = np.where((j + q) % 2 == 0, math.pi, 0.0).reshape(-1)
    alpha = rng.uniform(0.0, TWO_PI, phi.shape)
    return alpha, alpha + phi, phi


def herald_errors(r: np.ndarray, phi: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """One qubit. r (sets, shots, 2) outcomes (+1 = "0"), phi (sets, 2).

    Returns the heralded errors e0_H, e1_H (record 1 among shots whose herald read
    "0"), the herald's "1" fraction (initialization population plus first-readout
    e0), and the kept fraction, each with a binomial standard error.
    """
    herald0 = r[:, :, 0] == 1
    out = {}
    h1 = float((~herald0).mean())
    out["herald_reads_1"] = (h1, math.sqrt(h1 * (1 - h1) / herald0.size))
    stay = np.cos(phi[:, 1]) < 0          # phi = pi keeps |0>
    for name, sel, bad in (("e0_H", stay, -1), ("e1_H", ~stay, 1)):
        keep = herald0[sel]
        x = r[sel][:, :, 1][keep]
        p = float((x == bad).mean()) if x.size else float("nan")
        out[name] = (p, math.sqrt(p * (1 - p) / x.size) if x.size else float("nan"))
    return out


def simulate(n_sets: int, shots: int, rng, p_init: float, e0_first: float, e1_first: float,
             e0_chain: float, e1_chain: float, kind: str = "assign_herald") -> Tuple[np.ndarray, np.ndarray]:
    """Outcomes (sets, shots, K) and phi (sets, K) for one qubit.

    After reset the qubit is |1> with probability p_init. The first readout of a
    shot uses (e0_first, e1_first), later readouts (e0_chain, e1_chain). Readouts
    leave the state in place; steps are ideal. kind "assign" is AS, one readout.
    """
    if kind == "assign":
        _, _, ph = _assign_like(n_sets, rng)
        phi = ph[:, None]
    else:
        _, _, phi = herald_settings(n_sets, 1, rng)
    K = phi.shape[1]
    state = (rng.random((n_sets, shots)) < p_init).astype(int)       # 1 = |1>
    out = np.empty((n_sets, shots, K), np.int8)
    for k in range(K):
        flip = (np.cos(phi[:, k]) > 0)[:, None]                         # phi = 0 flips
        state = np.where(flip, 1 - state, state)
        e0, e1 = (e0_first, e1_first) if k == 0 else (e0_chain, e1_chain)
        wrong = rng.random(state.shape) < np.where(state == 0, e0, e1)
        reported1 = np.where(wrong, 1 - state, state)
        out[:, :, k] = np.where(reported1 == 1, -1, 1)
    return out, phi


def _assign_like(n_sets: int, rng):
    j = np.arange(n_sets)
    phi = np.where(j % 2 == 0, math.pi, 0.0)
    alpha = rng.uniform(0.0, TWO_PI, phi.shape)
    return alpha, alpha + phi, phi


def as_errors(r: np.ndarray, phi: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """AS errors of one qubit, as cloud_run_eigenstate.assign_errors computes them."""
    prep0 = np.cos(phi) < 0
    out = {}
    for name, sel, bad in (("e0", prep0, -1), ("e1", ~prep0, 1)):
        x = r[sel].reshape(-1)
        p = float((x == bad).mean())
        out[name] = (p, math.sqrt(p * (1 - p) / x.size))
    return out


def compare(as_e: Dict[str, Tuple[float, float]], h: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """d0 = e0_AS - e0_H, d1 = e1_AS - e1_H, and d1 - d0, with standard errors."""
    d0 = (as_e["e0"][0] - h["e0_H"][0], math.hypot(as_e["e0"][1], h["e0_H"][1]))
    d1 = (as_e["e1"][0] - h["e1_H"][0], math.hypot(as_e["e1"][1], h["e1_H"][1]))
    return {"d0": d0, "d1": d1, "d1_minus_d0": (d1[0] - d0[0], math.hypot(d0[1], d1[1]))}


def pool(values) -> Tuple[float, float]:
    """Unweighted mean over qubits and the standard error of that mean."""
    v = np.array([x[0] for x in values])
    s = np.array([x[1] for x in values])
    return float(v.mean()), float(math.sqrt(float(np.sum(s ** 2))) / len(v))


def verdicts(per_qubit: list) -> Dict[str, str]:
    """The pre-registered readings of Section 2I on the pooled values."""
    d1 = pool([q["d1"] for q in per_qubit])
    dd = pool([q["d1_minus_d0"] for q in per_qubit])
    out = {}
    out["question_1"] = ("heralding removes a first-readout excess in e1" if d1[0] > K_SIGMA * d1[1] else
                         "heralding raises e1" if d1[0] < -K_SIGMA * d1[1] else
                         "no first-readout excess in e1 resolved")
    out["question_2"] = ("the e1 excess exceeds the e0 excess: initialization alone does not explain it"
                         if dd[0] > K_SIGMA * dd[1] else
                         "the e0 excess exceeds the e1 excess" if dd[0] < -K_SIGMA * dd[1] else
                         "consistent with an initialization population alone")
    out["pooled_d1"] = d1
    out["pooled_d1_minus_d0"] = dd
    return out


# ---------------------------------------------------------------- design checks

class Check:
    def __init__(self):
        self.count = 0

    def true(self, label, cond):
        self.count += 1
        if not cond:
            raise AssertionError(label)

    def within(self, label, v, se, want, k=5.0):
        self.count += 1
        if not abs(v - want) <= k * se + 1e-12:
            raise AssertionError(f"{label}: {v:.6f} +- {se:.6f}, want {want:.6f}")


def _budget() -> Dict[str, object]:
    rep_delay, shots = 250e-6, 400
    entries = []
    rows = []
    for label, kind, delay, K, seconds in HERALD_CONDITIONS:
        t_exec = rep_delay + K * (FEZ_PERIOD_S + delay)
        sets = max(1, int(round(seconds / (t_exec * shots))))
        per_set = t_exec * shots
        entries.append((label, sets, per_set))
        rows.append({"label": label, "kind": kind, "records": K, "sets": sets, "qpu_s": round(sets * per_set, 2)})
    plan = submit.job_plan(entries, submit.MAX_JOB_S)
    per = {lab: ps for lab, _, ps in entries}
    est = [round(sum((b - a) * per[lab] for lab, (a, b) in job), 2) for job in plan]
    total = sum(est) + submit.JOB_OVERHEAD_S * len(plan)
    return {"conditions": rows, "jobs": [[lab for lab, _ in job] for job in plan], "job_estimates_s": est,
            "total_with_overhead_s": round(total, 1), "usage_margin_s": submit.USAGE_MARGIN * WINDOW_LEFT_S,
            "total_at_5.1s_overhead_s": round(sum(est) + 5.1 * len(plan), 1)}


def run(output: str | None) -> int:
    chk = Check()
    rng = np.random.default_rng(20261120)
    b = _budget()
    chk.true("every job under the per-job cap", max(b["job_estimates_s"]) < submit.MAX_JOB_S)
    chk.true("estimate with overhead under the usage margin of the 49 s left",
             b["total_with_overhead_s"] < submit.USAGE_MARGIN * WINDOW_LEFT_S)
    chk.true("charge at the worst October overhead under the 49 s left", b["total_at_5.1s_overhead_s"] < WINDOW_LEFT_S)
    sets = {row["label"]: row["sets"] for row in b["conditions"]}
    # Settings: record 0 stays, record 1 alternates.
    _, _, ph = herald_settings(10, 3, rng)
    chk.true("record 0 always stays in |0>", np.allclose(ph[:, 0], math.pi))
    chk.true("record 1 alternates stay and flip", set(np.round(ph[:, 1], 6)) == {0.0, round(math.pi, 6)})
    cases = {
        "initialization only": dict(p_init=0.003, e0_first=0.002, e1_first=0.006, e0_chain=0.002, e1_chain=0.006),
        "first-readout e1 excess": dict(p_init=0.0, e0_first=0.002, e1_first=0.009, e0_chain=0.002, e1_chain=0.006),
        "no offset": dict(p_init=0.0, e0_first=0.002, e1_first=0.006, e0_chain=0.002, e1_chain=0.006),
    }
    want_q1 = {"initialization only": "heralding removes a first-readout excess in e1",
               "first-readout e1 excess": "heralding removes a first-readout excess in e1",
               "no offset": "no first-readout excess in e1 resolved"}
    want_q2 = {"initialization only": "consistent with an initialization population alone",
               "first-readout e1 excess": "the e1 excess exceeds the e0 excess: initialization alone does not explain it",
               "no offset": "consistent with an initialization population alone"}
    table = {}
    for name, kw in cases.items():
        per_q = []
        for q in range(8):
            r_h, ph_h = simulate(sets["AH"], 400, rng, **kw)
            r_a, ph_a = simulate(sets["AS"], 400, rng, kind="assign", **kw)
            h = herald_errors(r_h, ph_h)
            a = as_errors(r_a[:, :, 0], ph_a[:, 0])
            if q == 0:
                chk.within(f"{name}: heralded e0 is the in-chain e0", *h["e0_H"], kw["e0_chain"])
                chk.within(f"{name}: heralded e1 is the in-chain e1", *h["e1_H"], kw["e1_chain"])
                chk.within(f"{name}: herald reads 1 at p + e0", *h["herald_reads_1"],
                           kw["p_init"] * (1 - kw["e1_first"]) + (1 - kw["p_init"]) * kw["e0_first"])
            per_q.append(compare(a, h))
        v = verdicts(per_q)
        chk.true(f"{name}: question 1 reads '{want_q1[name]}'", v["question_1"] == want_q1[name])
        chk.true(f"{name}: question 2 reads '{want_q2[name]}'", v["question_2"] == want_q2[name])
        table[name] = {k: (list(map(lambda x: round(x, 6), val)) if isinstance(val, tuple) else val)
                       for k, val in v.items()}
    # Reach: pooled d1 standard error at the planned sizes.
    reach = table["no offset"]["pooled_d1"][1]
    chk.true("pooled d1 resolves 0.1 percentage point at 3 sigma", 3 * reach < 1e-3)
    result = {"status": "PASS", "checks": chk.count, "budget": b, "simulated_cases": table,
              "pooled_d1_standard_error": reach,
              "note": "Simulated design checks; nothing here is experimental data. Readings fixed in "
                      "notes/CLOUD_RUN_NEXT_DESIGN.md Section 2I."}
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    print(f"PASS: {chk.count} checks")
    print(json.dumps(b, indent=1))
    return 0


def evaluate(run_dir: str, inchain: str, output: str | None) -> int:
    """Apply the Section 2I readings to a saved AH run and compare with E0R's in-chain e1."""
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as fh:
        rep = json.load(fh)
    layout = rep["layout"]
    ah = rep["conditions"]["AH"]["per_qubit"]
    asr = rep["conditions"]["AS"]["per_qubit"]
    with open(inchain, encoding="utf-8") as fh:
        ic = json.load(fh)["qubits"]
    per_q, rows = [], {}
    for i, phys in enumerate(layout):
        h = {k: tuple(v) for k, v in ah[i].items()}
        a = {k: tuple(v) for k, v in asr[i].items()}
        c = compare(a, h)
        per_q.append(c)
        row = {"heralded": h, "AS": a, **{k: list(v) for k, v in c.items()}}
        if str(phys) in ic:
            e1_chain = ic[str(phys)]["clean"]["de1"][0] + _e1_null(ic, phys, run_dir)
            se = math.hypot(ic[str(phys)]["clean"]["de1"][1], h["e1_H"][1])
            z = (h["e1_H"][0] - e1_chain) / se if e1_chain is not None else float("nan")
            row["e0r_in_chain_e1"] = e1_chain
            row["heralded_minus_in_chain_e1_z"] = z
            row["heralded_matches_in_chain_e1"] = bool(abs(z) <= K_SIGMA)
        rows[str(phys)] = row
    v = verdicts(per_q)
    out = {"note": "Pre-registered readings of notes/CLOUD_RUN_NEXT_DESIGN.md Section 2I.",
           "qubits": rows, "readings": {k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()},
           "matches_in_chain_e1": sum(r.get("heralded_matches_in_chain_e1", False) for r in rows.values())}
    print(json.dumps(out["readings"], indent=1), "| heralded e1 matches E0R in-chain e1 on",
          out["matches_in_chain_e1"], "of", len(rows))
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    return 0


def _e1_null(ic: dict, phys: int, run_dir: str) -> float | None:
    """E0R's AS e1 for the qubit (the base the in-chain shift de1 is measured from)."""
    path = os.path.join(os.path.dirname(run_dir.rstrip("/")), "2026-11-19_eigrepeat", "report.json")
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    i = rep["layout"].index(int(phys))
    return rep["conditions"]["E0"]["per_qubit_null"][i]["null"]["inputs"]["e1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output")
    ap.add_argument("--evaluate", metavar="RUN_DIR")
    ap.add_argument("--inchain", default="data/cloudrun/2026-11-19_eigrepeat/inchain_reading.json")
    args = ap.parse_args()
    try:
        if args.evaluate:
            return evaluate(args.evaluate, args.inchain, args.output)
        return run(args.output)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
