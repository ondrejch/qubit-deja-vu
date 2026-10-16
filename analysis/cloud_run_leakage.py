#!/usr/bin/env python3
"""Exploratory leakage model for the cloud-run records (not pre-registered).

The 15 October 2026 main run showed a neighbor covariance above five standard
errors at all gaps, with a lag spectrum that stays flat across a shot and an
excess of long runs of "1" readings (notes/CLOUD_RUN_RESULTS.md, Section 4).
This script tests one conventional explanation: a leaked level |2> that the
readout reports as "1", entered during readouts and left by decay.

Model, per qubit (three true levels |0>, |1>, |2>; quantum mechanics, no hidden
variable): before each step, |2> decays to |1> with probability g_dn per
record period; the step acts on |0>, |1> as designed (P(0) = (1 - b cos phi)/2
from |b>) and leaves |2> alone; during each readout |0> and |1> leak to |2>
with probabilities g0 and g1; the readout reports "1" from |0> with
probability e0, "0" from |1> with probability e1, and "1" from |2> with
probability q2; the post-readout level is the true level.

The parameters are fitted by maximum likelihood (forward algorithm) to the B0
probe chain, whose flip and repeat steps separate leakage from readout error.
The fitted model then predicts the balanced chains' lag spectra at the three
gaps (A0, A1, A2), scaling g_dn with the record period. The difference between
observed and predicted neighbor covariance is the residual that a
detector-anchored persistence would have to fit into.

  python analysis/cloud_run_leakage.py data/cloudrun/2026-10-15_main --output data/cloudrun/2026-10-15_main/leakage_exploratory.json

Exploratory: the model and the residual statistic were chosen after seeing
the data. Needs NumPy and SciPy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_run_analyze as cra  # noqa: E402

NAMES = ("e0", "e1", "g0", "g1", "g_dn", "q2")


def _probs(theta: np.ndarray) -> dict:
    p = 1.0 / (1.0 + np.exp(-theta))
    return dict(zip(NAMES, p))


def _step_matrices(cphi: np.ndarray) -> np.ndarray:
    """(sets, K) cos(phi) -> (sets, K, 3, 3) row-stochastic step matrices."""
    M = np.zeros(cphi.shape + (3, 3))
    M[..., 0, 0] = 0.5 * (1.0 - cphi)
    M[..., 0, 1] = 0.5 * (1.0 + cphi)
    M[..., 1, 0] = 0.5 * (1.0 + cphi)
    M[..., 1, 1] = 0.5 * (1.0 - cphi)
    M[..., 2, 2] = 1.0
    return M


def neg_loglik(theta: np.ndarray, r: np.ndarray, cphi: np.ndarray) -> float:
    """r (sets, shots, K) in {+1, -1}; cphi (sets, K)."""
    p = _probs(theta)
    sets, shots, K = r.shape
    M = _step_matrices(cphi)
    G = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, p["g_dn"], 1.0 - p["g_dn"]]])
    Lk = np.array([[1.0 - p["g0"], 0.0, p["g0"]], [0.0, 1.0 - p["g1"], p["g1"]], [0.0, 0.0, 1.0]])
    read1 = np.array([p["e0"], 1.0 - p["e1"], p["q2"]])          # P(report "1" | level)
    alpha = np.zeros((sets, shots, 3))
    alpha[..., 0] = 1.0
    ll = 0.0
    for k in range(K):
        alpha = alpha @ G
        alpha = np.matmul(alpha, M[:, k])
        alpha = alpha @ Lk
        is1 = (r[:, :, k] < 0)[..., None]
        alpha = alpha * np.where(is1, read1, 1.0 - read1)
        c = alpha.sum(axis=-1, keepdims=True)
        ll += float(np.log(c).sum())
        alpha = alpha / c
    return -ll


def fit_qubit(r: np.ndarray, cphi: np.ndarray, sets_max: int = 400, shots_max: int = 200) -> dict:
    r = r[:sets_max, :shots_max]
    cphi = cphi[:sets_max]
    start = np.log(np.array([0.005, 0.02, 0.0005, 0.002, 0.02, 0.95]) / (1 - np.array([0.005, 0.02, 0.0005, 0.002, 0.02, 0.95])))
    res = minimize(neg_loglik, start, args=(r, cphi), method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-3})
    p = _probs(res.x)
    return {"params": {k: float(v) for k, v in p.items()}, "neg_loglik": float(res.fun),
            "converged": bool(res.success), "records": int(r.size)}


def simulate_balanced(p: dict, n_shots: int, K: int, g_dn: float, rng) -> np.ndarray:
    """Outcomes of a balanced chain (cos phi = 0) under the fitted model."""
    level = np.zeros(n_shots, int)
    out = np.empty((n_shots, K), np.int8)
    read1 = np.array([p["e0"], 1.0 - p["e1"], p["q2"]])
    for k in range(K):
        u = rng.random(n_shots)
        level = np.where((level == 2) & (u < g_dn), 1, level)
        comp = level < 2
        level = np.where(comp, np.where(rng.random(n_shots) < 0.5, 0, 1), level)
        u = rng.random(n_shots)
        leak = np.where(level == 0, u < p["g0"], np.where(level == 1, u < p["g1"], False))
        level = np.where(leak, 2, level)
        out[:, k] = np.where(rng.random(n_shots) < read1[level], -1, 1)
    return out


def lag_spectrum(r: np.ndarray, max_lag: int = 8):
    """r (N, K) -> list of (cov, se) for lags 1..max_lag."""
    out = []
    for L in range(1, min(max_lag, r.shape[1] - 1) + 1):
        out.append(cra._cov_se(r[:, :-L], r[:, L:]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest, chains, _ = cra.load(args.run)
    readout_s = manifest["build_report"].get("readout_s", 1.38e-6)
    sx_s = manifest["build_report"].get("sx_s", 32e-9)
    period0 = 2 * sx_s + readout_s
    rng = np.random.default_rng(20261015)
    b0 = chains["B0"]
    nq = b0["r"].shape[2]
    report = {"run": args.run, "status": "exploratory, chosen after seeing the data",
              "model": __doc__.split("Model, per qubit")[1].split("The parameters")[0].strip(),
              "record_period_B0_s": period0, "qubits": []}
    with ProcessPoolExecutor(max_workers=nq) as pool:
        fits = list(pool.map(fit_qubit, [b0["r"][:, :, q, :] for q in range(nq)],
                             [np.cos(b0["phi"][:, q, :]) for q in range(nq)]))
    for q in range(nq):
        fit = fits[q]
        p = fit["params"]
        entry = {"qubit_index": q, "physical_qubit": manifest["build_report"]["layout"][q], "fit": fit,
                 "leaked_lifetime_us": period0 / max(p["g_dn"], 1e-9) * 1e6, "conditions": {}}
        for label in ("A0", "A1", "A2"):
            a = chains[label]
            K = a["r"].shape[3]
            delay = 0.0 if label == "A0" else (4e-6 if label == "A1" else 40e-6)
            period = period0 + delay
            g_dn = 1.0 - (1.0 - p["g_dn"]) ** (period / period0)
            obs = a["r"][:, :, q, :].reshape(-1, K)
            sim = simulate_balanced(p, 400_000, K, g_dn, rng)
            o, s = lag_spectrum(obs), lag_spectrum(sim)
            resid = o[0][0] - s[0][0]
            se = math.hypot(o[0][1], s[0][1])
            entry["conditions"][label] = {
                "record_period_us": round(period * 1e6, 3),
                "observed_lags": [round(v, 5) for v, _ in o],
                "predicted_lags": [round(v, 5) for v, _ in s],
                "observed_minus_predicted_lag1": [round(resid, 5), round(se, 5)],
                "lag2to8_observed_minus_predicted_mean": round(float(np.mean([a_[0] - b_[0] for a_, b_ in zip(o[1:], s[1:])])), 5),
            }
        report["qubits"].append(entry)
        print(f"q{q} ({entry['physical_qubit']}): " + " ".join(f"{k}={v:.4g}" for k, v in p.items())
              + f" | lifetime {entry['leaked_lifetime_us']:.0f} us", flush=True)
        for label, c in entry["conditions"].items():
            print(f"   {label}: obs {c['observed_lags'][:4]} pred {c['predicted_lags'][:4]} "
                  f"lag1 resid {c['observed_minus_predicted_lag1']}", flush=True)
    # Pooled lag-1 residual per condition (mean over qubits).
    pooled = {}
    for label in ("A0", "A1", "A2"):
        vals = [e["conditions"][label]["observed_minus_predicted_lag1"] for e in report["qubits"]]
        v = float(np.mean([x[0] for x in vals]))
        s = float(math.sqrt(sum(x[1] ** 2 for x in vals)) / len(vals))
        pooled[label] = {"lag1_residual": [round(v, 6), round(s, 6)], "upper_5sigma": round(v + 5 * s, 6)}
    report["pooled_lag1_residual"] = pooled
    print(json.dumps(pooled, indent=1))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
