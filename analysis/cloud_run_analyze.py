#!/usr/bin/env python3
"""Save and analyze records of the IBM cloud run (notes/CLOUD_RUN_DESIGN.md).

save_job() writes one job's outcomes and drawn settings to a directory: a
manifest (manifest.json), one NPZ file per chain condition (outcomes as int8,
+1 = |0>, shape (sets, shots, qubits, records), with the phases), and the
amplitude check (amplitude.npz). analyze() computes the statistics fixed in
the design note (Section 7) per qubit and pooled, applies its decision rules
and writes a JSON report. The analysis needs only NumPy and groups records by
parameter set, so a full main job fits in memory.

  python analysis/cloud_run_analyze.py analyze RUN_DIR --output RUN_DIR/report.json
  python analysis/cloud_run_analyze.py selftest          # NumPy only
  python analysis/cloud_run_analyze.py selftest --aer    # also Qiskit and Aer

The self-tests write synthetic runs to a temporary directory. They inject the
two completions and conventional effects (pulse amplitude error, readout error,
readout memory), check that analyze() recovers them, and check the grouped
estimators against the record-level ones in analysis/cloud_run_design.py. With
--aer, circuits from analysis/cloud_run_circuits.py run on Qiskit Aer with a
coherent sx over-rotation and readout error, and the real Qiskit results go
through save_job(). Nothing here contacts IBM or submits anything, and no
self-test output is experimental data.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_run_design as design  # noqa: E402
import cloud_run_followup as followup  # noqa: E402

FIVE = 5.0
PHI_WINDOW = 0.1  # rad: probe steps this close to pi repeat the state, to 0 flip it


# ---------------------------------------------------------------- storage

def _register(data):
    """The one classical register of a PUB result (named "c" by the builder)."""
    if hasattr(data, "c"):
        return data.c
    return next(iter(data.values()))


def bits_to_outcomes(bool_bits: np.ndarray, n_qubits: int, K: int) -> np.ndarray:
    """Bits (..., shots, n_qubits*K), clbit q*K + k -> int8 outcomes (..., shots, qubits, K)."""
    return (1 - 2 * bool_bits.astype(np.int8)).reshape(bool_bits.shape[:-1] + (n_qubits, K))


def save_arrays(directory: str, manifest: dict, chains: Dict[str, dict], amplitude: dict | None) -> None:
    directory = os.path.expanduser(directory)
    os.makedirs(directory, exist_ok=True)
    for label, arrays in chains.items():
        np.savez_compressed(os.path.join(directory, f"{label}.npz"), **arrays)
    if amplitude is not None:
        np.savez_compressed(os.path.join(directory, "amplitude.npz"), **amplitude)
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


def save_job(directory: str, built: dict, result, job=None) -> dict:
    """Write a finished job to `directory`.

    built: the dictionary returned by cloud_run_circuits.build_all() for the job.
    result: the job's result (one PUB result per entry of built["pubs"]).
    job: the job object, optional, for its identifier.
    """
    chains, amp_m, amp_r = {}, [], []
    n_qubits = len(built["report"]["layout"])
    for i, label in enumerate(built["labels"]):
        bits = _register(result[i].data).to_bool_array(order="little")
        if label.startswith("amplitude_"):
            amp_m.append(int(label.split("_")[1]))
            amp_r.append(bits_to_outcomes(bits, n_qubits, 1)[..., 0])
            continue
        meta = built["meta"][label]
        arrays = {"r": bits_to_outcomes(bits, n_qubits, meta["records"]),
                  "kind": np.array(meta["kind"]), "records": np.array(meta["records"])}
        for key in ("alpha", "beta", "phi", "relabel"):
            if key in meta:
                arrays[key] = np.asarray(meta[key])
        chains[label] = arrays
    amplitude = {"m": np.array(amp_m), "r": np.stack(amp_r)} if amp_m else None
    job_id = None
    if job is not None:
        job_id = job.job_id() if callable(getattr(job, "job_id", None)) else getattr(job, "job_id", None)
    manifest = {
        "saved": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "job_id": job_id,
        "build_report": built["report"],
        "chains": {label: {"kind": str(a["kind"]), "records": int(a["records"]), "shape": list(a["r"].shape)}
                   for label, a in chains.items()},
        "amplitude_pulses": amp_m,
    }
    save_arrays(directory, manifest, chains, amplitude)
    return manifest


def load(directory: str) -> Tuple[dict, Dict[str, dict], dict | None]:
    directory = os.path.expanduser(directory)
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    chains = {}
    for label in manifest["chains"]:
        with np.load(os.path.join(directory, f"{label}.npz")) as z:
            chains[label] = {k: z[k] for k in z.files}
    amplitude = None
    path = os.path.join(directory, "amplitude.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            amplitude = {k: z[k] for k in z.files}
    return manifest, chains, amplitude


# ---------------------------------------------------------------- statistics

def _cov_se(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    prod = (x - x.mean()) * (y - y.mean())
    return float(prod.mean()), float(prod.std() / math.sqrt(prod.size))


def amplitude_fit(m: np.ndarray, r: np.ndarray) -> Dict[str, float]:
    """Pulse amplitude error from one readout after m sx pulses, m = 1 (mod 4).

    The mean outcome is b - c sin(m pi eps / 2): b absorbs the readout bias
    and c the readout contrast (bounded to [0.5, 1.05]). For every eps on a
    grid, b and c follow by weighted least squares; the profile chi^2 picks
    eps. The returned A = c^2 sin^2(pi eps / 2) is the amplitude term in the
    reported neighbor covariance; its error is the half-width of A over the
    region chi^2 <= chi^2_min + 1, which stays narrow where c and eps are
    degenerate (small m eps), because A depends only on their product there.
    """
    m = np.asarray(m, dtype=np.float64)
    mean = r.astype(np.float64).mean(axis=1)
    wgt = r.shape[1] / np.clip(1.0 - mean ** 2, 1e-6, None)
    grid = np.linspace(-0.1, 0.1, 8001)
    S = np.sin(np.outer(grid, m) * math.pi / 2.0)          # (grid, points)
    sw, sy = np.sum(wgt), np.sum(wgt * mean)
    ss, ssy, s1 = S ** 2 @ wgt, S @ (wgt * mean), S @ wgt
    det = sw * ss - s1 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        c = -(sw * ssy - s1 * sy) / det                   # model b - c S
    c = np.where(np.isfinite(c), np.clip(c, 0.5, 1.05), 1.0)
    b = (sy + c * s1) / sw
    resid = mean[None, :] - (b[:, None] - c[:, None] * S)
    chi2 = np.sum(wgt[None, :] * resid ** 2, axis=1)
    i = int(np.argmin(chi2))
    A = c ** 2 * np.sin(math.pi * grid / 2.0) ** 2
    band = chi2 <= chi2[i] + 1.0
    step = grid[1] - grid[0]
    se_eps = max(0.5 * (grid[band].max() - grid[band].min()), step)
    se_A = max(0.5 * (A[band].max() - A[band].min()), 1e-12)
    return {"eps": float(grid[i]), "se_eps": float(se_eps), "contrast": float(c[i]),
            "A": float(A[i]), "se_A": float(se_A), "chi2_min": float(chi2[i]), "points": int(len(m))}


def lag2_sign(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """t = n_(k-1).n_(k+1) per set: alpha, beta (sets, K) -> (sets, K-2)."""
    K = alpha.shape[1]
    return np.stack([design.z_of_steps([alpha[:, k], alpha[:, k + 1]], [beta[:, k], beta[:, k + 1]])
                     for k in range(1, K - 1)], axis=1)


def balanced_one_qubit(r: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Dict[str, object]:
    """r (sets, shots, K) of one qubit; alpha, beta (sets, K)."""
    x = r.astype(np.float64)
    t = lag2_sign(alpha, beta)
    out = {
        "d_lag1": _cov_se(x[:, :, :-1], x[:, :, 1:]),
        "d_lag2": _cov_se(x[:, :, :-2], x[:, :, 2:]),
        "q_lag2": _cov_se(t[:, None, :] * x[:, :, :-2], x[:, :, 2:]),
        "t_values": sorted(set(np.round(t, 6).ravel().tolist())),
    }
    # Diagnostic: centering by the mean over sets at each shot index removes
    # drift slower than one pass over the sets (execution is column-wise).
    xc = x - x.mean(axis=0, keepdims=True)
    out["d_lag1_drift_centered"] = _cov_se(xc[:, :, :-1], xc[:, :, 1:])
    out["lag_spectrum"] = [round(_cov_se(x[:, :, :-L], x[:, :, L:])[0], 6) for L in range(1, min(9, x.shape[2]))]
    return out


def probe_groups(r: np.ndarray, phi: np.ndarray):
    """Group (B -> P) and (P -> B) pairs by parameter set and record.

    Returns ybar, count, phi, is_bp for every group."""
    x = r.astype(np.float64)
    sets, shots, K = x.shape
    ybar, cnt, ph, bp = [], [], [], []
    for k in range(1, K, 2):
        ybar.append((x[:, :, k - 1] * x[:, :, k]).mean(axis=1))
        ph.append(phi[:, k])
        bp.append(np.ones(sets, bool))
        if k + 1 < K:
            ybar.append((x[:, :, k] * x[:, :, k + 1]).mean(axis=1))
            ph.append(phi[:, k])
            bp.append(np.zeros(sets, bool))
    ybar = np.concatenate(ybar)
    return ybar, np.full(ybar.shape, float(shots)), np.concatenate(ph), np.concatenate(bp)


def probe_fit_grouped(ybar, cnt, phi, is_bp, max_order: int = 2) -> Dict[str, Tuple[float, float]]:
    """design.probe_fit on grouped pairs: identical estimates and errors,
    because outcomes are +-1 and the design depends on the group only."""
    orders = list(range(max_order + 1))
    n_nuis = 2 * max_order + 1
    nuis = np.stack(design._harmonics(phi, orders), axis=1)
    X = np.zeros((len(ybar), 2 * n_nuis + 2))
    X[is_bp, :n_nuis] = nuis[is_bp]
    X[~is_bp, n_nuis:2 * n_nuis] = nuis[~is_bp]
    X[:, -2] = design.d_template(phi)
    X[is_bp, -1] = design.q_template(phi[is_bp])
    A = X[:, :-2]
    mu = A @ np.linalg.solve(A.T @ (A * cnt[:, None]), A.T @ (cnt * ybar))
    wgt = cnt / np.clip(1.0 - mu ** 2, 0.02, None)
    G = X.T @ (X * wgt[:, None])
    coef = np.linalg.solve(G, X.T @ (wgt * ybar))
    fit = X @ coef
    # Record-level weighted residual sum: sum over records of (y - fit)^2 with y^2 = 1.
    rss = np.sum((wgt / cnt) * cnt * (1.0 - 2.0 * fit * ybar + fit ** 2))
    scale = float(rss / (cnt.sum() - X.shape[1]))
    se = np.sqrt(np.diag(np.linalg.inv(G)) * scale)
    return {"w_D": (float(coef[-2]), float(se[-2])), "w_Q": (float(coef[-1]), float(se[-1]))}


def harmonic_fit(ybar: np.ndarray, cnt: np.ndarray, phi: np.ndarray, max_order: int = 6) -> Dict[str, object]:
    """Weighted harmonic fit of grouped +-1 outcomes; amplitudes per order."""
    A = np.stack(design._harmonics(phi, list(range(max_order + 1))), axis=1)
    mu = A @ np.linalg.solve(A.T @ (A * cnt[:, None]), A.T @ (cnt * ybar))
    wgt = cnt / np.clip(1.0 - mu ** 2, 0.02, None)
    G = A.T @ (A * wgt[:, None])
    coef = np.linalg.solve(G, A.T @ (wgt * ybar))
    cov = np.linalg.inv(G)
    out = {"constant": [float(coef[0]), float(math.sqrt(cov[0, 0]))], "orders": {}}
    for j in range(1, max_order + 1):
        c, s = coef[2 * j - 1], coef[2 * j]
        amp = math.hypot(c, s)
        z = float(np.sqrt([c, s] @ np.linalg.solve(cov[2 * j - 1:2 * j + 1, 2 * j - 1:2 * j + 1], [c, s])))
        out["orders"][str(j)] = {"amplitude": round(float(amp), 6), "sqrt_chi2": round(z, 2)}
    return out


def readout_characterization(r: np.ndarray, phi: np.ndarray) -> Dict[str, object]:
    """Probe steps near phi = pi repeat the previous state; near 0 they flip it."""
    x = r.astype(np.float64)
    out = {}
    for name, target in (("repeat", math.pi), ("flip", 0.0)):
        vals = []
        for k in range(1, x.shape[2], 2):
            d = np.abs(((phi[:, k] - target + math.pi) % (2 * math.pi)) - math.pi)
            sel = d < PHI_WINDOW
            if sel.any():
                vals.append((x[sel, :, k - 1] * x[sel, :, k]).ravel())
        if vals:
            v = np.concatenate(vals)
            out[name] = {"mean_product": round(float(v.mean()), 5), "pairs": int(v.size)}
    return out


def cross_qubit(r: np.ndarray) -> Dict[str, object]:
    """Covariances between qubits read in parallel (same record, and lag 1)."""
    sets, shots, nq, K = r.shape
    rows = []
    for a in range(nq):
        for b in range(a + 1, nq):
            c0 = _cov_se(r[:, :, a, :], r[:, :, b, :])
            c1 = _cov_se(r[:, :, a, :-1], r[:, :, b, 1:])
            rows.append({"qubits": [a, b], "lag0": [round(c0[0], 6), round(c0[1], 6)],
                         "lag1": [round(c1[0], 6), round(c1[1], 6)]})
    zmax = max((abs(row[k][0]) / row[k][1] for row in rows for k in ("lag0", "lag1")), default=0.0)
    return {"pairs": rows, "max_abs_z": round(float(zmax), 2)}


def cross_execution(r: np.ndarray) -> Tuple[float, float]:
    """Last record of set b with the first record of set b + 1 at the same shot
    index (consecutive executions if execution is column-wise). Exploratory."""
    return _cov_se(r[:-1, :, :, -1], r[1:, :, :, 0])


def _upper(value: float, se: float) -> float:
    return value + FIVE * se


def _pool(items: List[Tuple[float, float]]) -> Tuple[float, float]:
    vals = np.array([v for v, _ in items])
    ses = np.array([s for _, s in items])
    return float(vals.mean()), float(math.sqrt(np.sum(ses ** 2)) / len(items))


def analyze(directory: str) -> dict:
    manifest, chains, amplitude = load(directory)
    report = {"run": directory, "job_id": manifest.get("job_id"),
              "backend": manifest.get("build_report", {}).get("backend"),
              "layout": manifest.get("build_report", {}).get("layout"),
              "note": "Statistics fixed in notes/CLOUD_RUN_DESIGN.md Section 7.",
              "amplitude": None, "conditions": {}, "flags": []}
    n_qubits = None
    amp_terms = None
    if amplitude is not None:
        m, r = amplitude["m"], amplitude["r"]  # r (n_m, shots, qubits)
        n_qubits = r.shape[2]
        fits = [amplitude_fit(m, r[:, :, q]) for q in range(n_qubits)]
        amp_terms = [(f["A"], f["se_A"]) for f in fits]
        report["amplitude"] = [{k: (round(v, 7) if isinstance(v, float) else v) for k, v in f.items()} for f in fits]
    for label, arrays in chains.items():
        kind = str(arrays["kind"])
        r = arrays["r"]
        nq = r.shape[2]
        entry: Dict[str, object] = {"kind": kind, "shape": list(r.shape)}
        if kind == "balanced":
            per_q = [balanced_one_qubit(r[:, :, q, :], arrays["alpha"][:, q, :], arrays["beta"][:, q, :])
                     for q in range(nq)]
            d_fix = []
            for q, st in enumerate(per_q):
                A, sA = amp_terms[q] if amp_terms else (0.0, 0.0)
                v, s = st["d_lag1"]
                d_fix.append((v - A, math.hypot(s, sA)))
            q_fix = [st["q_lag2"] for st in per_q]
            pooled_d, pooled_q = _pool(d_fix), _pool(q_fix)
            entry["per_qubit"] = [{
                "D_fix": [round(d[0], 6), round(d[1], 6)], "D_fix_upper_5sigma": round(_upper(*d), 6),
                "Q_fix": [round(qq[0], 6), round(qq[1], 6)], "Q_fix_upper_5sigma": round(_upper(*qq), 6),
                "d_lag1": [round(st["d_lag1"][0], 6), round(st["d_lag1"][1], 6)],
                "d_lag2": [round(st["d_lag2"][0], 6), round(st["d_lag2"][1], 6)],
                "d_lag1_drift_centered": [round(st["d_lag1_drift_centered"][0], 6),
                                          round(st["d_lag1_drift_centered"][1], 6)],
                "lag_spectrum": st["lag_spectrum"], "t_values": st["t_values"],
            } for st, d, qq in zip(per_q, d_fix, q_fix)]
            entry["pooled"] = {"D_fix": [round(pooled_d[0], 6), round(pooled_d[1], 6)],
                               "D_fix_upper_5sigma": round(_upper(*pooled_d), 6),
                               "Q_fix": [round(pooled_q[0], 6), round(pooled_q[1], 6)],
                               "Q_fix_upper_5sigma": round(_upper(*pooled_q), 6)}
            entry["cross_qubit"] = cross_qubit(r) if nq > 1 else None
            ce = cross_execution(r)
            entry["cross_execution_exploratory"] = [round(ce[0], 6), round(ce[1], 6)]
            if pooled_q[0] > FIVE * pooled_q[1]:
                report["flags"].append(f"{label}: Q_fix above five standard errors; follow decision rule 2")
            if pooled_d[0] > FIVE * pooled_d[1]:
                report["flags"].append(f"{label}: D_fix above five standard errors; follow decision rule 3")
            if not amp_terms:
                report["flags"].append(f"{label}: no amplitude check in this run; D_fix is d_lag1 uncorrected")
        elif kind == "probe":
            per_q = []
            groups_all = []
            for q in range(nq):
                g = probe_groups(r[:, :, q, :], arrays["phi"][:, q, :])
                groups_all.append(g)
                fit2 = probe_fit_grouped(*g, max_order=2)
                fit4 = probe_fit_grouped(*g, max_order=4)
                per_q.append({"order2": fit2, "order4": fit4,
                              "readout": readout_characterization(r[:, :, q, :], arrays["phi"][:, q, :])})
            cat = [np.concatenate([g[i] for g in groups_all]) for i in range(4)]
            pooled = {"order2": probe_fit_grouped(*cat, max_order=2), "order4": probe_fit_grouped(*cat, max_order=4)}
            entry["per_qubit"] = [{o: {k: [round(v[0], 6), round(v[1], 6)] for k, v in pq[o].items()}
                                   for o in ("order2", "order4")} | {"readout": pq["readout"]} for pq in per_q]
            entry["pooled"] = {o: {k: [round(v[0], 6), round(v[1], 6)] for k, v in pooled[o].items()}
                               for o in ("order2", "order4")}
            wq, sq = pooled["order2"]["w_Q"]
            if wq > FIVE * sq:
                report["flags"].append(f"{label}: w_Q above five standard errors; follow decision rule 2")
            wd2, sd2 = pooled["order2"]["w_D"]
            wd4, sd4 = pooled["order4"]["w_D"]
            if wd2 > FIVE * sd2 and wd4 > FIVE * sd4:
                report["flags"].append(f"{label}: cusp w_D above five standard errors at orders 2 and 4")
        elif kind == "tagged":
            per_q, d_clean, cross = [], [], []
            for q in range(nq):
                st = followup.tagged_statistics(r[:, :, q, :].reshape(-1, r.shape[3]))
                A, sA = amp_terms[q] if amp_terms else (0.0, 0.0)
                v, se = st["clean"]["lag1"]
                d_clean.append((v - A, math.hypot(se, sA)))
                cross.append(st["clean"]["cross_block"] if "cross_block" in st["clean"] else (float("nan"), float("nan")))
                per_q.append({
                    "D_fix_clean": [round(v - A, 6), round(math.hypot(se, sA), 6)],
                    "D_fix_clean_upper_5sigma": round(_upper(v - A, math.hypot(se, sA)), 6),
                    "clean": {k: [round(x[0], 6), round(x[1], 6)] for k, x in st["clean"].items()},
                    "all": {k: [round(x[0], 6), round(x[1], 6)] for k, x in st["all"].items()},
                    **{k: (round(x, 6) if isinstance(x, float) else x) for k, x in st.items() if k not in ("clean", "all")},
                })
            pooled_d = _pool(d_clean)
            entry["per_qubit"] = per_q
            entry["pooled"] = {"D_fix_clean": [round(pooled_d[0], 6), round(pooled_d[1], 6)],
                               "D_fix_clean_upper_5sigma": round(_upper(*pooled_d), 6)}
            if r.shape[3] > followup.BLOCK:
                pooled_x = _pool(cross)
                entry["pooled"]["cross_block_clean"] = [round(pooled_x[0], 6), round(pooled_x[1], 6)]
                if abs(pooled_x[0]) > FIVE * pooled_x[1]:
                    report["flags"].append(f"{label}: clean cross-block covariance nonzero; tagging did not remove the background")
            for q, (v, se) in enumerate(d_clean):
                if abs(v) > FIVE * se:
                    report["flags"].append(f"{label}: qubit {q} clean D_fix = {v:.5f} +- {se:.5f} (readout memory or persistence)")
        elif kind == "phase":
            per_q = []
            for q in range(nq):
                ybar = r[:, :, q, 0].astype(np.float64).mean(axis=1)
                cnt = np.full(ybar.shape, float(r.shape[1]))
                per_q.append(harmonic_fit(ybar, cnt, arrays["phi"][:, q, 0]))
            entry["per_qubit"] = per_q
            zmax = max(per_q[q]["orders"][str(j)]["sqrt_chi2"] for q in range(nq) for j in range(3, 7))
            entry["max_sqrt_chi2_above_order_2"] = zmax
            if zmax > FIVE:
                report["flags"].append(f"{label}: phase generation shows harmonics above order 2 "
                                       f"(sqrt chi^2 = {zmax})")
        report["conditions"][label] = entry
    if not report["flags"]:
        report["flags"].append("none: every primary statistic is within five standard errors of zero")
    return report


# ---------------------------------------------------------------- self-tests

class Check:
    def __init__(self) -> None:
        self.count = 0

    def within(self, label: str, value: float, se: float, want: float, k: float = FIVE) -> None:
        self.count += 1
        if not abs(value - want) <= k * se + 1e-12:
            raise AssertionError(f"{label}: got {value:.6f} +- {se:.6f}, want {want:.6f}")

    def true(self, label: str, cond: bool) -> None:
        self.count += 1
        if not cond:
            raise AssertionError(label)


def _synthetic_run(directory: str, model: str, w: float, rng, sets: int, shots: int, nq: int,
                   K: int = 8, **noise) -> None:
    """Write a synthetic run in save_job's format, generated by design.simulate."""
    chains = {}
    for label, kind in (("A0", "balanced"), ("B0", "probe"), ("C2", "phase")):
        n = sets * nq
        if kind == "balanced":
            alpha, beta, extra = design.balanced_settings(n, K, rng)
            extra_key = "relabel"
        elif kind == "probe":
            alpha, beta, extra = design.probe_settings(n, K, rng)
            extra_key = "phi"
        else:
            alpha = rng.uniform(0.0, 2 * math.pi, (n, 1))
            extra = rng.uniform(0.0, 2 * math.pi, (n, 1))
            beta = alpha + extra
            extra_key = "phi"
        Kc = alpha.shape[1]
        a_rep = np.repeat(alpha.reshape(sets, 1, nq, Kc), shots, axis=1).reshape(-1, Kc)
        b_rep = np.repeat(beta.reshape(sets, 1, nq, Kc), shots, axis=1).reshape(-1, Kc)
        out = design.simulate(model, a_rep, b_rep, w, rng, **noise)
        chains[label] = {"r": out.reshape(sets, shots, nq, Kc).astype(np.int8), "kind": np.array(kind),
                         "records": np.array(Kc), "alpha": alpha.reshape(sets, nq, Kc),
                         "beta": beta.reshape(sets, nq, Kc), extra_key: extra.reshape(sets, nq, Kc)}
    eps = noise.get("eps", 0.0) if model == "QM" else 0.0
    e0, e1 = noise.get("e0", 0.0), noise.get("e1", 0.0)
    ms = np.array([1, 9, 25, 49, 97])
    amp_shots = 20 * shots
    z = np.cos(ms * 0.5 * math.pi * (1.0 + eps))                     # true <Z> after m pulses
    p1 = 0.5 * (1.0 - z)                                            # P(|1>)
    p_report1 = p1 * (1 - e1) + (1 - p1) * e0
    r_amp = np.where(rng.random((len(ms), amp_shots, nq)) < p_report1[:, None, None], -1, 1).astype(np.int8)
    manifest = {"saved": "synthetic", "job_id": None, "build_report": {"backend": f"synthetic {model}",
                "layout": list(range(nq))},
                "chains": {k: {"kind": str(v["kind"]), "records": int(v["records"]), "shape": list(v["r"].shape)}
                           for k, v in chains.items()}, "amplitude_pulses": ms.tolist()}
    save_arrays(directory, manifest, chains, {"m": ms, "r": r_amp})


def selftest_numpy(chk: Check, rng) -> dict:
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        # 1. Grouped estimators equal the record-level ones of the design script.
        path = os.path.join(tmp, "consistency")
        _synthetic_run(path, "D", 0.3, rng, sets=60, shots=40, nq=2)
        _, chains, _ = load(path)
        a = chains["A0"]
        for q in range(2):
            mine = balanced_one_qubit(a["r"][:, :, q, :], a["alpha"][:, q, :], a["beta"][:, q, :])
            K = a["r"].shape[3]
            rows = a["r"][:, :, q, :].reshape(-1, K).astype(np.float64)
            al = np.repeat(a["alpha"][:, None, q, :], a["r"].shape[1], axis=1).reshape(-1, K)
            be = np.repeat(a["beta"][:, None, q, :], a["r"].shape[1], axis=1).reshape(-1, K)
            ref = design.balanced_statistics(rows, al, be)
            for key in ("d_lag1", "d_lag2", "q_lag2"):
                chk.true(f"grouped {key} equals record-level (qubit {q})",
                         abs(mine[key][0] - ref[key][0]) < 1e-12 and abs(mine[key][1] - ref[key][1]) < 1e-12)
        b = chains["B0"]
        for q in range(2):
            g = probe_groups(b["r"][:, :, q, :], b["phi"][:, q, :])
            mine = probe_fit_grouped(*g)
            K = b["r"].shape[3]
            rows = b["r"][:, :, q, :].reshape(-1, K).astype(np.float64)
            ph = np.repeat(b["phi"][:, None, q, :], b["r"].shape[1], axis=1).reshape(-1, K)
            ref = design.probe_fit(rows, ph)
            for key in ("w_D", "w_Q"):
                chk.true(f"grouped shape fit {key} equals record-level (qubit {q})",
                         abs(mine[key][0] - ref[key][0]) < 1e-9 and abs(mine[key][1] / ref[key][1] - 1) < 1e-6)

        # 2. Recovery of injected effects through save, load and analyze.
        cases = [
            ("QM, amplitude error 8%, readout 1%/3%", "QM", 0.0, dict(eps=0.08, e0=0.01, e1=0.03)),
            ("QM, readout memory 3%", "QM", 0.0, dict(e0=0.01, e1=0.03, mem=0.03)),
            ("D, w = 0.3", "D", 0.3, {}),
            ("Q, w = 0.3", "Q", 0.3, {}),
        ]
        for name, model, w, noise in cases:
            path = os.path.join(tmp, model + name[-4:].replace(" ", "_"))
            _synthetic_run(path, model, w, rng, sets=300, shots=60, nq=2, **noise)
            rep = analyze(path)
            pa = rep["conditions"]["A0"]["pooled"]
            pb = rep["conditions"]["B0"]["pooled"]["order2"]
            d_fix, q_fix = pa["D_fix"], pa["Q_fix"]
            w_d, w_q = pb["w_D"], pb["w_Q"]
            if model == "QM" and "amplitude" in name:
                want_A = 0.96 ** 2 * math.sin(math.pi * 0.08 / 2) ** 2
                for q in range(2):
                    f = rep["amplitude"][q]
                    chk.within(f"{name}: eps recovered (qubit {q})", f["eps"], f["se_eps"], 0.08)
                    chk.within(f"{name}: amplitude term recovered (qubit {q})", f["A"], f["se_A"], want_A)
                raw = rep["conditions"]["A0"]["per_qubit"][0]["d_lag1"]
                chk.true(f"{name}: raw d_lag1 moved by the amplitude term", raw[0] > FIVE * raw[1])
                chk.within(f"{name}: D_fix after subtraction", *d_fix, 0.0)
            elif model == "QM":
                chk.true(f"{name}: D_fix flagged", d_fix[0] > FIVE * d_fix[1])
            if model == "D":
                chk.within(f"{name}: D_fix", *d_fix, w)
                chk.within(f"{name}: shape w_D", *w_d, w)
            else:
                chk.within(f"{name}: shape w_D", *w_d, 0.0)
            chk.within(f"{name}: Q_fix", *q_fix, w * w if model == "Q" else 0.0)
            chk.within(f"{name}: shape w_Q", *w_q, w if model == "Q" else 0.0)
            phase = rep["conditions"]["C2"]["max_sqrt_chi2_above_order_2"]
            chk.true(f"{name}: phase check clean (max z {phase})", phase < FIVE)
            summary[name] = {"D_fix": d_fix, "Q_fix": q_fix, "w_D": w_d, "w_Q": w_q, "flags": rep["flags"]}
    return summary


def selftest_tagged(chk: Check, rng) -> dict:
    """Tagged follow-up chains with leakage, through save, load and analyze."""
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, model, w in (("QM, leakage", "QM", 0.0), ("D w=0.3, leakage", "D", 0.3)):
            sets, shots, nq, K = 200, 60, 2, 66
            alpha, beta, phi = followup.tagged_settings(sets * nq, K, rng)
            rep_phi = np.repeat(phi.reshape(sets, 1, nq, K), shots, axis=1).reshape(-1, K)
            out = followup.simulate_leaky(model, rep_phi, rng, w=w, **followup.LEAK)
            chains = {"T0": {"r": out.reshape(sets, shots, nq, K).astype(np.int8), "kind": np.array("tagged"),
                             "records": np.array(K), "alpha": alpha.reshape(sets, nq, K),
                             "beta": beta.reshape(sets, nq, K), "phi": phi.reshape(sets, nq, K)}}
            manifest = {"saved": "synthetic", "job_id": None,
                        "build_report": {"backend": f"synthetic {model}", "layout": list(range(nq))},
                        "chains": {"T0": {"kind": "tagged", "records": K, "shape": list(chains["T0"]["r"].shape)}},
                        "amplitude_pulses": []}
            path = os.path.join(tmp, model)
            save_arrays(path, manifest, chains, None)
            rep = analyze(path)
            pooled = rep["conditions"]["T0"]["pooled"]
            contrast = (1.0 - followup.LEAK["e0"] - followup.LEAK["e1"]) ** 2
            chk.within(f"tagged {name}: clean D_fix", *pooled["D_fix_clean"], w * contrast)
            chk.within(f"tagged {name}: clean cross-block covariance", *pooled["cross_block_clean"],
                       w ** 6 * contrast)
            raw = rep["conditions"]["T0"]["per_qubit"][0]["all"]["cross_block"]
            chk.true(f"tagged {name}: leakage visible before selection", raw[0] > FIVE * raw[1])
            summary[name] = {"pooled": pooled, "flags": rep["flags"]}
    return summary


def selftest_aer(chk: Check) -> dict:
    """Circuits of cloud_run_circuits.py on Qiskit Aer, through save_job()."""
    import cloud_run_circuits as circ
    from qiskit import transpile
    from qiskit.circuit.library import RXGate
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, coherent_unitary_error
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    eps = 0.06
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(coherent_unitary_error(RXGate(0.5 * math.pi * eps).to_matrix()), ["sx"])
    noise.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.03, 0.97]]))
    sim = AerSimulator()
    sampler = AerSampler(options={"backend_options": {"noise_model": noise, "seed_simulator": 7}})
    n_qubits, shots = 2, 100
    plan = [("A0", "balanced", 8, 160), ("B0", "probe", 8, 320), ("C2", "phase", 1, 400)]
    pubs, labels, meta = [], [], {}
    for i, (label, kind, K, sets) in enumerate(plan):
        qc = transpile(circ.chain_circuit(K, n_qubits, 0, False), sim, optimization_level=0)
        vals, m = circ.draw_settings(kind, K, n_qubits, sets, 100 + i)
        pubs.append((qc, vals[:, circ.order_for(qc, K, n_qubits)], shots))
        labels.append(label)
        meta[label] = {"kind": kind, "records": K, **m}
    for c in circ.amplitude_circuits(n_qubits, False):
        pubs.append((transpile(c, sim, optimization_level=0), None, 20 * shots))
        labels.append(c.name)
    built = {"pubs": pubs, "labels": labels, "meta": meta,
             "report": {"backend": "aer with sx over-rotation 6% and readout error 1%/3%",
                        "layout": list(range(n_qubits))}}
    result = sampler.run([(c, v, s) for c, v, s in pubs]).result()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = save_job(tmp, built, result)
        chk.true("save_job wrote every chain condition", set(manifest["chains"]) == {"A0", "B0", "C2"})
        rep = analyze(tmp)
    want_A = 0.96 ** 2 * math.sin(math.pi * eps / 2) ** 2
    for q in range(n_qubits):
        f = rep["amplitude"][q]
        chk.within(f"Aer: eps recovered (qubit {q})", f["eps"], f["se_eps"], eps)
        chk.within(f"Aer: amplitude term recovered (qubit {q})", f["A"], f["se_A"], want_A)
    pa = rep["conditions"]["A0"]["pooled"]
    chk.within("Aer: D_fix after amplitude subtraction", *pa["D_fix"], 0.0)
    chk.within("Aer: Q_fix", *pa["Q_fix"], 0.0)
    pb = rep["conditions"]["B0"]["pooled"]["order2"]
    chk.within("Aer: shape w_D", *pb["w_D"], 0.0)
    chk.within("Aer: shape w_Q", *pb["w_Q"], 0.0)
    chk.true("Aer: phase check clean", rep["conditions"]["C2"]["max_sqrt_chi2_above_order_2"] < FIVE)
    raw = rep["conditions"]["A0"]["per_qubit"][0]["d_lag1"]
    return {"amplitude": rep["amplitude"], "A0_pooled": pa, "B0_order2": pb, "raw_d_lag1_qubit0": raw,
            "flags": rep["flags"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="analyze a saved run directory")
    pa.add_argument("directory")
    pa.add_argument("--output", help="write the report as JSON")
    ps = sub.add_parser("selftest", help="synthetic self-tests")
    ps.add_argument("--aer", action="store_true", help="also run circuits on Qiskit Aer (needs Qiskit)")
    ps.add_argument("--output", help="write the self-test summary as JSON")
    args = parser.parse_args()
    try:
        if args.cmd == "analyze":
            report = analyze(args.directory)
            text = json.dumps(report, indent=2, default=str)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
            print(text)
            return 0
        chk = Check()
        summary = {"numpy": selftest_numpy(chk, np.random.default_rng(20261015)),
                   "tagged": selftest_tagged(chk, np.random.default_rng(20261016))}
        if args.aer:
            summary["aer"] = selftest_aer(chk)
        summary["checks"] = chk.count
        summary["status"] = "PASS"
        summary["note"] = "Synthetic self-tests of the analysis code; nothing here is experimental data."
        text = json.dumps(summary, indent=2, default=str)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        print(f"PASS: {chk.count} checks")
        return 0
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
