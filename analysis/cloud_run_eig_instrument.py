"""Eigenstate readout memory against a general memoryless readout instrument.

Reanalysis of the 19 November 2026 eigenstate runs (E0, E0R) after an outside
referee report (20 November 2026) showed that the in-chain null of
analysis/cloud_run_eig_drift.py is biased. Its simulator lets a |1> decay only
after it has been reported, never during its own readout, so it treats the whole
first-readout e1 as a misread that leaves the state in place. A memoryless chain
in which part of e1 is decay during the readout (report "0", state |0>) then
reads as "negative memory" at 6 to 10 standard errors (legacy subcommand,
analysis/cloud_run_eig_instrument_legacy.json). This method was chosen
after seeing the E0, E0R and AH data and that critique; it was fixed, with its
synthetic tests and its reading rule, before it was applied to the records.

Data. Each eight-record block B B B B F F R P ends in five eigenstate-linked
records: x (record 3, the last equator readout, whose outcome sets the state),
F1 and F2 (flips), R (repeat, identity step) and P (flip). The 32 outcome
patterns of (x, F1, F2, R, P) are counted per set, on two masks that use only
the neighboring blocks' checks, so the block's own records stay unselected:

  prev       the previous block's check (x, F1, F2) is normal
  prev_next  the previous and the next block's checks are both normal

Model (populations, Z-twirled; levels 0, 1 and leaked 2). The initial joint
distribution of x's report and the level after it is free (p1, a0, a1, lam,
q2). Before each later step the level after the previous readout changes:
0 -> 1 with c0, 1 -> 0 with c1 (transitions after the report, including T1
during the rest of the period), 2 -> 1 with G_DN. The pulse flips 0 <-> 1 at
F1, F2 and P and leaves the level at R. The readout of level s leaks to 2 with g_s (reported "1" with q2);
otherwise it reports the wrong value and leaves s in place with e_s, or reports
the wrong value and moves to the other level with d_s at the flip steps and
d_s r at R (decay or excitation during the readout, or a pulse error). Memory is the dependence of e_s on the previous report in
the block: e_s|0 after a "0", e_s|1 = e_s|0 + m_s after a "1". m1 > 0 is
negative memory (a "1" makes the next readout of |1> report "0"), m0 > 0
positive memory, as in analysis/cloud_run_eigenstate.py. In the ideal sequence
e1|1 enters only at R, so m_s is the excess state-preserving error of the
repeat readout over the flip readouts of the same state, beyond everything the
memoryless instrument (including decay during and after readout, pulse errors
and leakage) can produce. A state-flipping excess at R (d_s r above d_s) is not
separable from a pulse error at R and is not called memory here.

The prev_next mask down-weights blocks that end leaked by a free factor rho.

Gauge. A level change after the previous report (c), at the pulse (eps) and
during the readout before its report (d) all act between two reports, and the
patterns fix only their composite per step type: six parameters, four
identifiable combinations (the Fisher matrix has two null directions). The fit
sets c0 = c1 = 0, so d carries every transition between reports. Decay
during a readout and decay before it are therefore the same thing to these
data. A failed or spurious pulse is likewise identical to a readout that
reports, and moves to, the other level, so pulse errors are carried by d too,
which therefore takes its own values at the flip steps (d0, d1) and at the
repeat step R (d0r, d1r). Quantities reported as results do not depend on the gauge: m0, m1, e0,
e1 and the composite misread of a fresh |1> (a flip step after a readout that
left |0> and reported "0", reporting "0"), which is the in-chain counterpart of
the first-readout AS e1.

Reading rule (fixed before the data): per qubit and state, memory is read when
|m_s| exceeds 3 standard errors on both masks with the same sign. The standard
error is the larger of the Fisher error and the jackknife error over 16
contiguous groups of sets, which carries drift within the run. The fit also
gives the in-chain total misread of a fresh |1> after a "0", e1|0 + d1 (what a
first readout of a prepared |1> measures, apart from initialization), and its
offset from the run's first-readout AS value.

Usage:
  python analysis/cloud_run_eig_instrument.py selftest [--output FILE]
  python analysis/cloud_run_eig_instrument.py analyze data/cloudrun/<run> [--output FILE]

NumPy and SciPy. Nothing here contacts IBM. The synthetic chains of selftest are
simulations, not data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_run_eigenstate as eig  # noqa: E402
import cloud_run_followup as followup  # noqa: E402

MASKS = ("prev", "prev_next")
POS = (3, eig.F1, eig.F2, eig.R_POS, eig.P_POS)       # x, F1, F2, R, P within a block
STEPS = ("flip", "flip", "id", "flip")                  # the pulse before F1, F2, R, P
G_DN = 0.015                                            # leak return per step, fixed as in the null
GROUPS = 16
K_SIGMA = 3.0

# name, scale (typical size, for the optimizer), lower and upper bound
PARAMS = [("p1", 0.5, 0.2, 0.8), ("a0", 1e-3, 0.0, 0.1), ("a1", 1e-3, 0.0, 0.1), ("lam", 1e-3, 0.0, 0.2),
          ("q2", 0.5, 0.3, 1.0), ("c0", 1e-3, 0.0, 0.05), ("c1", 1e-3, 0.0, 0.05), ("g0", 1e-3, 0.0, 0.05), ("g1", 1e-3, 0.0, 0.05),
          ("e0", 1e-3, 0.0, 0.05), ("e1", 1e-3, 0.0, 0.05), ("d0", 1e-3, 0.0, 0.05), ("d1", 1e-3, 0.0, 0.05),
          ("d0r", 1e-3, 0.0, 0.05), ("d1r", 1e-3, 0.0, 0.05),
          ("m0", 1e-3, -0.05, 0.05), ("m1", 1e-3, -0.05, 0.05), ("rho", 0.5, 0.0, 1.0)]
NAMES = [p[0] for p in PARAMS]
SCALE = np.array([p[1] for p in PARAMS])
START = dict(p1=0.5, a0=2e-3, a1=4e-3, lam=3e-3, q2=0.9, c0=0.0, c1=0.0, g0=1e-4, g1=2e-4, e0=2e-3, e1=3e-3,
             d0=2e-4, d1=3e-3, d0r=2e-4, d1r=3e-3, m0=0.0, m1=0.0, rho=0.2)


# ---------------------------------------------------------------- counting

def pattern_counts(r_sets: np.ndarray) -> dict:
    """r_sets (sets, shots, 72) of one qubit (+1 = "0") -> counts (sets, 32) per mask."""
    sets, shots, K = r_sets.shape
    nb = K // eig.BLOCK
    bits = (r_sets.reshape(sets, shots, nb, eig.BLOCK)[..., list(POS)] == -1).astype(np.int64)
    idx = bits @ np.array([16, 8, 4, 2, 1])                            # (sets, shots, nb)
    normal = followup.check_states(r_sets, eig.BLOCK) == 0             # (sets, shots, nb)
    prev = np.zeros_like(normal)
    prev[..., 1:] = normal[..., :-1]
    nxt = np.zeros_like(normal)
    nxt[..., :-1] = normal[..., 1:]
    off = (np.arange(sets) * 32)[:, None, None]
    out = {}
    for name, m in (("prev", prev), ("prev_next", prev & nxt)):
        out[name] = np.bincount((idx + off)[m], minlength=sets * 32).reshape(sets, 32)
    return out


def conditional_table(counts: np.ndarray) -> dict:
    """Model-free table: P(R, P | F2) from the 32 pattern counts (summed over sets)."""
    c = counts.reshape(2, 2, 2, 2, 2).sum(axis=(0, 1))                 # (F2, R, P)
    out = {}
    for f2 in (0, 1):
        n = c[f2].sum()
        out[f"F2={f2}"] = {"n": int(n), **{f"R={r},P={p}": round(float(c[f2, r, p] / n), 6)
                                           for r in (0, 1) for p in (0, 1)}}
    return out


# ---------------------------------------------------------------- model

def probs(th: dict, mask: str) -> np.ndarray:
    """Pattern probabilities (32,), index x*16 + F1*8 + F2*4 + R*2 + P, with "1" = 1."""
    lam, q2, p1, a0, a1 = th["lam"], th["q2"], th["p1"], th["a0"], th["a1"]
    alpha = np.zeros((2, 3))
    nl = 1.0 - lam
    alpha[0, 0], alpha[1, 0] = nl * (1 - p1) * (1 - a0), nl * (1 - p1) * a0
    alpha[1, 1], alpha[0, 1] = nl * p1 * (1 - a1), nl * p1 * a1
    alpha[1, 2], alpha[0, 2] = lam * q2, lam * (1 - q2)
    C = np.array([[1 - th["c0"], th["c0"], 0.0], [th["c1"], 1 - th["c1"], 0.0], [0.0, G_DN, 1 - G_DN]])
    pulse = {"flip": np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]), "id": np.eye(3)}
    M = {}
    for step, suffix in (("flip", ""), ("id", "r")):
        Ms = np.zeros((2, 2, 3, 3))                                    # [prev report, report, in, out]
        for rp in (0, 1):
            for s in (0, 1):
                g, d = th[f"g{s}"], th[f"d{s}{suffix}"]
                e = max(th[f"e{s}"] + (th[f"m{s}"] if rp == 1 else 0.0), 0.0)
                Ms[rp, 1, s, 2], Ms[rp, 0, s, 2] = g * q2, g * (1 - q2)
                Ms[rp, s, s, s] = (1 - g) * (1 - e - d)
                Ms[rp, 1 - s, s, s] = (1 - g) * e
                Ms[rp, 1 - s, s, 1 - s] = (1 - g) * d
            Ms[rp, 1, 2, 2], Ms[rp, 0, 2, 2] = q2, 1 - q2
        M[step] = Ms
    for step in STEPS:
        beta = alpha @ (C @ pulse[step])
        n = beta.shape[0]
        rp = np.arange(n) % 2
        new = np.empty((2 * n, 3))
        new[0::2] = np.einsum("ni,nij->nj", beta, M[step][rp, 0])
        new[1::2] = np.einsum("ni,nij->nj", beta, M[step][rp, 1])
        alpha = new
    w = np.array([1.0, 1.0, th["rho"] if mask == "prev_next" else 1.0])
    p = alpha @ w
    return p / p.sum()


GAUGE_FIXED = ("c0", "c1")


def fresh_misread(th: dict) -> float:
    """P(report "0") at a flip step whose previous readout left |0> and reported "0" (gauge-free)."""
    C = np.array([[1 - th["c0"], th["c0"]], [th["c1"], 1 - th["c1"]]])
    ef = th.get("eps_f", 0.0)                       # pulse failure: synthetic truth only, zero in the fit
    lev = np.array([1.0, 0.0]) @ C @ np.array([[ef, 1 - ef], [1 - ef, ef]])
    p0 = 0.0
    for s in (0, 1):
        g, d, e = th[f"g{s}"], th[f"d{s}"], max(th[f"e{s}"], 0.0)
        rep0 = (1 - g) * (1 - e - d) if s == 0 else (1 - g) * (e + d)
        p0 += lev[s] * (rep0 + g * (1 - th["q2"]))
    return p0


def _free(mask: str, memory: bool) -> list:
    out = [n for n in NAMES if (n != "rho" or mask == "prev_next") and n not in GAUGE_FIXED]
    return [n for n in out if memory or n not in ("m0", "m1")]


def fit(counts: np.ndarray, mask: str, memory: bool, start: dict | None = None) -> dict:
    """Maximum-likelihood fit to pattern counts (32,). Returns parameters and deviance."""
    free = _free(mask, memory)
    base = {**START, **(start or {})}
    if not memory:
        base["m0"] = base["m1"] = 0.0
    if mask == "prev":
        base["rho"] = 1.0
    for n in GAUGE_FIXED:
        base[n] = 0.0
    sc = np.array([SCALE[NAMES.index(n)] for n in free])
    bounds = [(PARAMS[NAMES.index(n)][2] / s, PARAMS[NAMES.index(n)][3] / s) for n, s in zip(free, sc)]
    n_tot = counts.sum()
    frac = counts / n_tot

    def nll(u):
        th = dict(base)
        th.update(zip(free, u * sc))
        p = probs(th, mask)
        return -float(np.sum(frac * np.log(np.maximum(p, 1e-300))))

    u0 = np.array([base[n] for n in free]) / sc
    u0 = np.clip(u0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(nll, u0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000, "maxfun": 200000, "ftol": 1e-15, "gtol": 1e-11})
    th = dict(base)
    th.update(zip(free, res.x * sc))
    p = probs(th, mask)
    nz = counts > 0
    dev = 2.0 * float(np.sum(counts[nz] * np.log(counts[nz] / (n_tot * p[nz]))))
    return {"theta": th, "free": free, "deviance": dev, "dof": 31 - len(free), "n": int(n_tot),
            "converged": bool(res.success)}


def fisher_se(fr: dict, mask: str) -> dict:
    """Fisher standard errors of the free parameters away from their bounds."""
    th, n_tot = fr["theta"], fr["n"]
    free = [n for n in fr["free"]
            if PARAMS[NAMES.index(n)][2] + 1e-7 < th[n] < PARAMS[NAMES.index(n)][3] - 1e-7]
    p0 = probs(th, mask)
    J = np.empty((32, len(free)))
    for j, n in enumerate(free):
        h = 1e-3 * SCALE[NAMES.index(n)]
        tp, tm = dict(th), dict(th)
        tp[n] += h
        tm[n] -= h
        J[:, j] = (probs(tp, mask) - probs(tm, mask)) / (2 * h)
    info = n_tot * (J / p0[:, None]).T @ J
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = dict(zip(free, map(float, se)))
    out["_condition"] = float(np.linalg.cond(info))
    grad = np.zeros(len(free))
    for j, n in enumerate(free):
        h = 1e-3 * SCALE[NAMES.index(n)]
        tp, tm = dict(th), dict(th)
        tp[n] += h
        tm[n] -= h
        grad[j] = (fresh_misread(tp) - fresh_misread(tm)) / (2 * h)
    out["_fresh1"] = float(math.sqrt(max(grad @ cov @ grad, 0.0)))
    return out


def analyze_qubit(counts: dict, label: str = "") -> dict:
    """Fits on both masks, memoryless and with memory, plus jackknife errors over contiguous set groups."""
    out = {}
    for mask in MASKS:
        cs = counts[mask]
        tot = cs.sum(axis=0)
        f0 = fit(tot, mask, memory=False)
        f1 = fit(tot, mask, memory=True, start=f0["theta"])
        fse = fisher_se(f1, mask)
        groups = np.array_split(np.arange(cs.shape[0]), GROUPS)
        jk = []
        for g in groups:
            keep = np.ones(cs.shape[0], bool)
            keep[g] = False
            fj = fit(cs[keep].sum(axis=0), mask, memory=True, start=f1["theta"])
            jk.append([fj["theta"]["m0"], fj["theta"]["m1"], fresh_misread(fj["theta"])])
        jk = np.array(jk)
        jse = np.sqrt((GROUPS - 1) / GROUPS * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0))
        th = f1["theta"]
        res = {"n_blocks": f1["n"], "table": conditional_table(tot), "counts": [int(v) for v in tot],
               "memoryless": {"deviance": round(f0["deviance"], 2), "dof": f0["dof"], "converged": f0["converged"]},
               "with_memory": {"deviance": round(f1["deviance"], 2), "dof": f1["dof"], "converged": f1["converged"]},
               "lr_statistic": round(f0["deviance"] - f1["deviance"], 2),
               "params": {n: round(float(th[n]), 7) for n in NAMES if n not in GAUGE_FIXED},
               "fisher_se": {n: round(v, 7) for n, v in fse.items()}}
        for i, nm in enumerate(("m0", "m1")):
            se = max(fse.get(nm, float("nan")), float(jse[i]))
            res[nm] = {"value": round(float(th[nm]), 6), "se": round(se, 6), "fisher_se": round(fse.get(nm, float("nan")), 6),
                       "jackknife_se": round(float(jse[i]), 6), "z": round(float(th[nm] / se), 2)}
        tot1 = fresh_misread(th)
        se1 = max(fse.get("_fresh1", float("nan")), float(jse[2]))
        res["fresh1_misread"] = {"value": round(float(tot1), 6), "se": round(se1, 6),
                                 "note": "flip step after a readout that left |0> and reported '0', reporting '0'"}
        out[mask] = res
        if label:
            print(f"  {label} {mask:9s} n={f1['n']:>8d} dev0 {f0['deviance']:7.1f} dev1 {f1['deviance']:7.1f} "
                  f"dof {f1['dof']:2d} | m0 {th['m0']:+.5f}±{res['m0']['se']:.5f} m1 {th['m1']:+.5f}±{res['m1']['se']:.5f}"
                  f" | e1 {th['e1']:.4f} d1 {th['d1']:.4f} c1 {th['c1']:.4f} e0 {th['e0']:.4f} d0 {th['d0']:.4f}",
                  flush=True)
    for nm, word in (("m1", "negative memory"), ("m0", "positive memory")):
        zs = [out[mk][nm]["z"] for mk in MASKS]
        if all(z > K_SIGMA for z in zs):
            out[f"reading_{nm}"] = f"{word} (state-preserving error after a '1' exceeds that after a '0')"
        elif all(z < -K_SIGMA for z in zs):
            out[f"reading_{nm}"] = "opposite sign: state-preserving error after a '1' below that after a '0'"
        else:
            out[f"reading_{nm}"] = "consistent with the memoryless instrument"
    return out


# ---------------------------------------------------------------- synthetic chains

def simulate(n: int, seed: int, e0=0.002, e1=0.003, d0=2e-4, d1=0.0045, c0=1e-4, c1=0.0045,
             g0=4e-5, g1=2e-4, q2=0.99, eps_f=2e-4, eps_id=1e-4, m0=0.0, m1=0.0, sets=400) -> np.ndarray:
    """Synthetic (sets, n // sets, 72) chain of the E0 design with the instrument of the model.

    Memory follows the previous report in the whole chain (B records included), so
    the fit's restriction to the block's own previous report is tested too. Every
    step applies the transitions of the model; equator steps rotate to the equator.
    """
    rng = np.random.default_rng(seed)
    _, _, phi = eig.tagged_eig_settings(n, 72, rng)
    level = np.zeros(n, int)
    prev = np.zeros(n, bool)                       # previous report was "1"
    out = np.empty((n, 72), np.int8)
    for k in range(72):
        u = rng.random(n)
        level = np.where(level == 2, np.where(u < G_DN, 1, 2),
                         np.where(level == 0, np.where(u < c0, 1, 0), np.where(u < c1, 0, 1)))
        ph = phi[:, k]
        flip = np.isclose(ph, 0.0)
        ident = np.isclose(ph, math.pi)
        u = rng.random(n)
        comp = level < 2
        p_flip = np.where(flip, 1 - eps_f, np.where(ident, eps_id, 0.5))
        newlev = np.where(u < p_flip, 1 - level, level)
        equator = ~(flip | ident)
        u2 = rng.random(n)
        newlev = np.where(equator, (u2 < 0.5).astype(int), newlev)
        level = np.where(comp, newlev, level)
        u = rng.random(n)
        g = np.where(level == 0, g0, g1)
        leak = (level < 2) & (u < g)
        level = np.where(leak, 2, level)
        e = np.where(level == 0, e0 + m0 * prev, e1 + m1 * prev)
        d = np.where(level == 0, d0, d1)
        u = rng.random(n)
        wrong_keep = (level < 2) & (u < e)
        wrong_flip = (level < 2) & (u >= e) & (u < e + d)
        rep1 = np.where(level == 2, rng.random(n) < q2, np.where(wrong_keep | wrong_flip, level == 0, level == 1))
        level = np.where(wrong_flip, 1 - level, level)
        out[:, k] = np.where(rep1, -1, 1)
        prev = rep1
    return out.reshape(sets, n // sets, 72)


def selftest(output: str | None) -> int:
    cases = [
        ("decay during readout, no memory (the referee's case)", dict(), {"m0": 0.0, "m1": 0.0}),
        ("decay during readout, T1 short, no memory", dict(d1=0.007, c1=0.007, e1=0.002), {"m0": 0.0, "m1": 0.0}),
        ("negative memory on |1>", dict(m1=0.003), {"m0": 0.0, "m1": 0.003}),
        ("positive memory on |0> with negative on |1>", dict(m0=0.002, m1=-0.0015, g1=8e-4),
         {"m0": 0.002, "m1": -0.0015}),
    ]
    failures, results = [], {}
    for i, (name, kw, truth) in enumerate(cases):
        t0 = time.time()
        r = simulate(240_000, 9100 + i, **kw)
        res = analyze_qubit(pattern_counts(r), label=f"case {i + 1}")
        row = {"truth": {**dict(e0=0.002, e1=0.003, d0=2e-4, d1=0.0045, c0=1e-4, c1=0.0045, g0=4e-5, g1=2e-4,
                                q2=0.99, eps_f=2e-4, eps_id=1e-4, m0=0.0, m1=0.0), **kw}, "fit": {}}
        for mask in MASKS:
            row["fit"][mask] = {k: res[mask][k] for k in ("m0", "m1", "memoryless", "with_memory", "fresh1_misread")}
            for nm in ("m0", "m1"):
                v, se = res[mask][nm]["value"], res[mask][nm]["se"]
                if abs(v - truth[nm]) > 4 * se + 2e-4:
                    failures.append(f"{name}, {mask}: {nm} = {v:+.5f} ± {se:.5f}, want {truth[nm]:+.5f}")
            wm = res[mask]["with_memory"]
            if wm["deviance"] > wm["dof"] + 5 * math.sqrt(2 * wm["dof"]):
                failures.append(f"{name}, {mask}: deviance {wm['deviance']} on {wm['dof']} dof")
            want1 = fresh_misread({**START, **row["truth"]})
            v, se = res[mask]["fresh1_misread"]["value"], res[mask]["fresh1_misread"]["se"]
            if abs(v - want1) > 4 * se + 3e-4:
                failures.append(f"{name}, {mask}: fresh |1> misread = {v:.5f} ± {se:.5f}, want {want1:.5f}")
        for nm in ("m0", "m1"):
            want_read = truth[nm] != 0.0
            got = res[f"reading_{nm}"] != "consistent with the memoryless instrument"
            if want_read != got and abs(truth[nm]) >= 0.0015:
                failures.append(f"{name}: reading of {nm} is '{res[f'reading_{nm}']}'")
            if not want_read and got:
                failures.append(f"{name}: false reading of {nm}: '{res[f'reading_{nm}']}'")
        row["readings"] = {nm: res[f"reading_{nm}"] for nm in ("m0", "m1")}
        results[name] = row
        print(f"  case {i + 1} done in {time.time() - t0:.0f} s", flush=True)
    status = "pass" if not failures else "fail"
    print(f"selftest: {status}", *failures, sep="\n  ")
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump({"note": "Synthetic chains of the E0 design (simulations, not data), 240,000 shots each; "
                               "the fit must recover the memory of each case within 4 standard errors plus 2e-4, "
                               "read none where there is none, and fit within 5 sigma of its deviance.",
                       "status": status, "failures": failures, "cases": results}, fh, indent=1)
            fh.write("\n")
    return 0 if not failures else 1


# ---------------------------------------------------------------- the withdrawn in-chain estimator

def legacy_check(output: str | None) -> int:
    """Memoryless synthetic chains with decay during the readout, read by the in-chain estimator of
    analysis/cloud_run_eig_drift.py unchanged (the outside referee's test of 20 November 2026).

    Part f of the T1 decay probability per period happens early in the readout (reports "0",
    leaves |0>), the rest after the report. A first-readout calibration would see e1 = a + early
    decay, which the estimator's null treats as a misread that leaves |1>.
    """
    import cloud_run_eig_drift as drift

    def chain(phi, rng, e0, a, t1, f, g, g_dn=0.015, q2=0.99):
        n, K = phi.shape
        p_relax = -math.expm1(-eig.FEZ_PERIOD_S / t1)
        p_early, p_late = f * p_relax, (1 - f) * p_relax
        level = np.zeros(n, int)
        out = np.empty((n, K), np.int8)
        for k in range(K):
            level = np.where((level == 2) & (rng.random(n) < g_dn), 1, level)
            level = np.where((level == 1) & (rng.random(n) < p_late), 0, level)
            b = np.where(level == 0, 1.0, -1.0)
            sz = b * (-np.cos(phi[:, k]))
            comp = np.where(rng.random(n) < 0.5 * (1.0 + sz), 0, 1)
            level = np.where(level < 2, comp, level)
            level = np.where(((level == 0) | (level == 1)) & (rng.random(n) < g), 2, level)
            level = np.where((level == 1) & (rng.random(n) < p_early), 0, level)
            p1 = np.where(level == 0, e0, np.where(level == 1, 1.0 - a, q2))
            out[:, k] = np.where(rng.random(n) < p1, -1, 1)
        return out, p_early

    rows = []
    for t1, f, a, e0 in ((150e-6, 0.5, 0.003, 0.003), (100e-6, 0.5, 0.003, 0.003), (200e-6, 0.4, 0.002, 0.002)):
        sets, shots = 700, 400
        rng = np.random.default_rng(1)
        _, _, phi = eig.tagged_eig_settings(sets * shots, 72, rng)
        r, b = chain(phi, rng, e0, a, t1, f, g=4e-5)
        inputs = {"e0": e0, "e1": a + b, "t1_s": t1, "leak_rate": 4e-5, "g_dn": 0.015, "q2": 0.99}
        res = drift.decompose(r.reshape(sets, shots, 72), drift.model(inputs, 12345), np.random.default_rng(5))
        c = res["clean"]
        new = analyze_qubit(pattern_counts(r.reshape(sets, shots, 72)))
        rows.append({"t1_us": t1 * 1e6, "early_fraction": f, "state_preserving_e1": a, "early_decay": round(b, 6),
                     "first_readout_e1": round(a + b, 6),
                     "in_chain_estimator": {"de1": c["de1"], "contrast_deviation": c["contrast_deviation_in_chain_null"],
                                            "z_clean": round(c["z_in_chain_null"], 2),
                                            "z_clean_next": round(res["clean_next"]["z_in_chain_null"], 2),
                                            "reading": res["reading_in_chain_null"],
                                            "first_readout_null_deviation": round(c["contrast_deviation_first_readout_null"], 6)},
                     "instrument_fit": {mk: {"m0": new[mk]["m0"], "m1": new[mk]["m1"]} for mk in MASKS} |
                                       {"reading_m1": new["reading_m1"], "reading_m0": new["reading_m0"]}})
        print(rows[-1], flush=True)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump({"note": "Simulations, not data: memoryless chains (700 sets x 400 shots, E0R size) with T1 "
                               "decay split between early in the readout and after the report; the in-chain "
                               "estimator of analysis/cloud_run_eig_drift.py reads them as negative memory, the "
                               "instrument fit does not.", "cases": rows}, fh, indent=1)
            fh.write("\n")
    return 0


# ---------------------------------------------------------------- records

def analyze(run: str, output: str | None) -> int:
    with open(os.path.join(run, "report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    layout = report["layout"]
    as_pq = report["conditions"]["AS"]["per_qubit"]
    with np.load(os.path.join(run, "E0.npz")) as z:
        r = z["r"]                                                   # (sets, shots, qubits, 72)
    out = {"note": "Reanalysis chosen after the E0, E0R and AH data and an outside referee report; method, "
                   "synthetic tests and reading rule fixed before this was run (analysis/cloud_run_eig_instrument.py). "
                   "Memory m_s is the excess state-preserving misread after a reported '1' over that after a '0', "
                   "beyond a memoryless instrument with decay during and after readout, pulse errors and leakage.",
           "run": report.get("run"), "layout": layout, "groups": GROUPS, "k_sigma": K_SIGMA, "g_dn": G_DN,
           "qubits": {}}
    for q, phys in enumerate(layout):
        t0 = time.time()
        res = analyze_qubit(pattern_counts(r[:, :, q, :]), label=f"q{phys}")
        e1_as, e1_as_se = as_pq[q]["e1"]
        for mask in MASKS:
            f = res[mask]["fresh1_misread"]
            off = f["value"] - e1_as
            se = math.hypot(f["se"], e1_as_se)
            res[mask]["first_readout_offset_e1"] = {"in_chain_minus_AS": round(off, 6), "se": round(se, 6),
                                                    "z": round(off / se, 2), "AS_e1": e1_as}
        out["qubits"][str(phys)] = res
        print(f"q{phys}: m1 {res['reading_m1']} | m0 {res['reading_m0']} | "
              f"in-chain e1 - AS {res['prev']['first_readout_offset_e1']['in_chain_minus_AS']:+.5f}"
              f"±{res['prev']['first_readout_offset_e1']['se']:.5f} ({time.time() - t0:.0f} s)", flush=True)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    return 0


def derived(run: str, output: str | None) -> int:
    """Quantities derived from a saved instrument_reading.json (no refit): per qubit and mask, the 3-sigma
    upper limit of |m_s|, the largest difference between observed and fitted pattern frequencies (with-memory
    fit), the state-preserving share of the fresh-|1> misread e1 / (e1 + d1), and the T1 implied by the extra
    state-flipping error of a |1> held through one more period, -period / ln(1 - (d1r - d1)), against the
    run-day calibration T1 of the build report (d1r - d1 also contains any pulse error at R)."""
    with open(os.path.join(run, "instrument_reading.json"), encoding="utf-8") as fh:
        fit = json.load(fh)
    with open(os.path.join(run, "report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    nulls = report["conditions"]["E0"]["per_qubit_null"]
    out = {"note": "Derived from instrument_reading.json without refitting (analysis/cloud_run_eig_instrument.py derived).",
           "period_s": eig.FEZ_PERIOD_S, "qubits": {}}
    for i, phys in enumerate(fit["layout"]):
        row = {}
        for mask in MASKS:
            r = fit["qubits"][str(phys)][mask]
            th = {**START, **r["params"], "c0": 0.0, "c1": 0.0}
            c = np.array(r["counts"], float)
            resid = float(np.max(np.abs(c / c.sum() - probs(th, mask))))
            held = th["d1r"] - th["d1"]
            t1 = -eig.FEZ_PERIOD_S / math.log1p(-held) if held > 0 else float("nan")
            cal = nulls[i]["null"]["inputs"]["t1_s"]
            row[mask] = {"upper_limit_3sigma": {nm: round(abs(r[nm]["value"]) + 3 * r[nm]["se"], 6) for nm in ("m0", "m1")},
                         "max_pattern_residual": round(resid, 7),
                         "state_preserving_share_e1": round(th["e1"] / (th["e1"] + th["d1"]), 3),
                         "held_extra_flip": round(held, 6), "implied_t1_us": round(t1 * 1e6, 1),
                         "calibration_t1_us": round(cal * 1e6, 1), "implied_over_calibration": round(t1 / cal, 3)}
        out["qubits"][str(phys)] = row
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--output")
    a = sub.add_parser("analyze")
    a.add_argument("run")
    dv = sub.add_parser("derived")
    dv.add_argument("run")
    dv.add_argument("--output")
    lg = sub.add_parser("legacy")
    lg.add_argument("--output")
    a.add_argument("--output")
    a.add_argument("--g-dn", type=float, help="leak return per step for a sensitivity run (default %(default)s)",
                   default=G_DN)
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest(args.output)
    if args.cmd == "legacy":
        return legacy_check(args.output)
    if args.cmd == "derived":
        return derived(args.run, args.output)
    globals()["G_DN"] = args.g_dn
    return analyze(args.run, args.output)


if __name__ == "__main__":
    sys.exit(main())
