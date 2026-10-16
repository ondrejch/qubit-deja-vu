#!/usr/bin/env python3
"""Design checks for the IBM cloud run planned in notes/CLOUD_RUN_DESIGN.md.

The run reads one qubit many times in a reset-free chain. Every step is two
pi/2 pulses whose phases are frame phases (virtual Z), followed by a readout:

  step(alpha, beta) = P(beta) P(alpha),
  P(g) = rotation by pi/2 about the axis (cos g, sin g, 0).

From a post-readout eigenstate b = +-1 (b = +1 is |0>), a step leaves
<Z> = -b cos(beta - alpha). Balanced steps use beta - alpha = pi/2, or 3 pi/2
to relabel the readout; probe steps draw beta - alpha uniformly.

The script simulates these chains under quantum mechanics with conventional
noise (pulse amplitude error, T1 decay, readout assignment error, readout
memory) and under the two completions of the theory note
(theory/theorynote.pdf) with renewal persistence. It checks that each
planned statistic responds to what the design says it responds to, and it
computes shape-test sensitivities and the QPU-time budget. The analysis
functions take arrays of outcomes and nominal phases, so the same code applies
to hardware records. The script exits nonzero on any failed check. It
validates the design only; nothing here is experimental data.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequential_dh_model as sdm  # noqa: E402

# q_frozen_pair_formula recomputes its Gauss-Legendre nodes on every call.
np.polynomial.legendre.leggauss = functools.lru_cache(maxsize=None)(np.polynomial.legendre.leggauss)

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------- geometry


def p_half(v: np.ndarray, gamma: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Rotate vectors v (n, 3) by (pi/2)(1 + eps) about (cos gamma, sin gamma, 0)."""
    t = 0.5 * math.pi * (1.0 + eps)
    ca, sa = np.cos(gamma), np.sin(gamma)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    cross = np.stack([sa * vz, -ca * vz, ca * vy - sa * vx], axis=1)
    along = (ca * vx + sa * vy) * (1.0 - math.cos(t))
    axis = np.stack([ca, sa, np.zeros_like(ca)], axis=1)
    return v * math.cos(t) + cross * math.sin(t) + axis * along[:, None]


def apply_step(v: np.ndarray, alpha: np.ndarray, beta: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return p_half(p_half(v, alpha, eps), beta, eps)


def z_of_steps(alphas: List[np.ndarray], betas: List[np.ndarray]) -> np.ndarray:
    """z component of (S_last ... S_first) applied to +z, for nominal pulses."""
    v = np.zeros((len(alphas[0]), 3))
    v[:, 2] = 1.0
    for a, b in zip(alphas, betas):
        v = apply_step(v, a, b)
    return v[:, 2]


def isotropic(n: int, rng) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- settings

def balanced_settings(n: int, K: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating balanced chain S, S^-1, S, S^-1, ... with one random frame
    offset per shot and an independent random relabel bit per step.

    S = step(a, a + pi/2); S^-1 = step(a + 3 pi/2, a + pi). A relabel adds pi
    to the second phase. Returns alpha, beta (n, K) and relabel (n, K) bool.
    """
    a0 = rng.uniform(0.0, TWO_PI, n)
    alpha = np.empty((n, K))
    beta = np.empty((n, K))
    for k in range(K):
        if k % 2 == 0:
            alpha[:, k], beta[:, k] = a0, a0 + 0.5 * math.pi
        else:
            alpha[:, k], beta[:, k] = a0 + 1.5 * math.pi, a0 + math.pi
    relabel = rng.random((n, K)) < 0.5
    return alpha, beta + math.pi * relabel, relabel


def probe_settings(n: int, K: int, rng, phi_fixed: float | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced and probe steps alternate (B, P, B, P, ...). Every step has a
    random frame offset; balanced steps carry a random relabel; probe steps
    have beta - alpha = phi, uniform on [0, 2 pi) unless phi_fixed is given.
    Returns alpha, beta (n, K) and phi (n, K) (relative phase of every step).
    """
    alpha = rng.uniform(0.0, TWO_PI, (n, K))
    phi = np.empty((n, K))
    for k in range(K):
        if k % 2 == 0:
            phi[:, k] = 0.5 * math.pi + math.pi * (rng.random(n) < 0.5)
        elif phi_fixed is None:
            phi[:, k] = rng.uniform(0.0, TWO_PI, n)
        else:
            phi[:, k] = phi_fixed
    return alpha, alpha + phi, phi


# ---------------------------------------------------------------- simulation

def simulate(model: str, alpha: np.ndarray, beta: np.ndarray, w: float, rng,
             eps: float = 0.0, t1: float = 0.0, e0: float = 0.0, e1: float = 0.0,
             mem: float = 0.0) -> np.ndarray:
    """Reported outcomes (n, K) in {+1, -1} (+1 = |0>) of a chain from |0>.

    model "QM": quantum mechanics with pulse amplitude error eps, decay of the
    excited state with probability t1 during each pre-step delay (amplitude
    damping of the Bloch vector), assignment errors e0 (|0> read as 1) and e1,
    and readout memory: after a reported 1, a |0> is misread with extra
    probability mem. Models "D" and "Q" are the theory note's completions with
    gap persistence w; their readouts are ideal.
    """
    n, K = alpha.shape
    s = np.zeros((n, 3))
    s[:, 2] = 1.0
    lam = isotropic(n, rng)
    u = rng.random(n)
    r_prev = np.ones(n)
    out = np.empty((n, K))
    for k in range(K):
        if k > 0 and model != "QM":
            redraw = rng.random(n) >= w
            m = int(redraw.sum())
            lam[redraw] = isotropic(m, rng)
            u[redraw] = rng.random(m)
        if t1 > 0.0:
            s = s * np.array([math.sqrt(1 - t1), math.sqrt(1 - t1), 1 - t1]) + np.array([0.0, 0.0, t1])
        s = apply_step(s, alpha[:, k], beta[:, k], eps if model == "QM" else 0.0)
        sz = s[:, 2]
        if model == "QM":
            b = np.where(rng.random(n) < 0.5 * (1.0 + sz), 1.0, -1.0)
        elif model == "D":
            b = np.where(u < 0.5 * (1.0 + sz), 1.0, -1.0)
        elif model == "Q":
            lam = apply_step(lam, alpha[:, k], beta[:, k])
            b = np.where(lam[:, 2] + sz > 0.0, 1.0, -1.0)
        else:
            raise ValueError(model)
        p_err = np.where(b > 0, e0 + mem * (r_prev < 0), e1)
        r = np.where(rng.random(n) < p_err, -b, b)
        out[:, k] = r
        s = np.zeros((n, 3))
        s[:, 2] = b
        r_prev = r
    return out


# ---------------------------------------------------------------- analysis

def _cov(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Covariance and its standard error (records treated as independent)."""
    xc, yc = x - x.mean(), y - y.mean()
    prod = xc * yc
    return float(prod.mean()), float(prod.std() / math.sqrt(len(prod)))


def balanced_statistics(r: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Dict[str, object]:
    """Planned fixed-angle statistics of a balanced chain.

    d_lag1: physical-label covariance of neighbors (detector-anchored test).
    d_lag2: physical-label covariance at lag 2.
    q_lag2: lag-2 covariance after multiplying by t = n_(k-1).n_(k+1), the
            overlap of the two readout axes in the frame that co-rotates with
            the qubit (qubit-anchored test). t is computed from nominal phases.
    """
    n, K = r.shape
    d1 = _cov(r[:, :-1].ravel(), r[:, 1:].ravel())
    d2 = _cov(r[:, :-2].ravel(), r[:, 2:].ravel())
    t = np.stack([z_of_steps([alpha[:, k], alpha[:, k + 1]], [beta[:, k], beta[:, k + 1]])
                  for k in range(1, K - 1)], axis=1)
    q2 = _cov((t * r[:, :-2]).ravel(), r[:, 2:].ravel())
    return {"d_lag1": d1, "d_lag2": d2, "q_lag2": q2, "t_values": np.unique(np.round(t, 6))}


@functools.lru_cache(maxsize=None)
def _q_pair_table(points: int = 2001) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, math.pi, points)
    val = np.array([sdm.q_frozen_pair_formula(th) if th <= 0.5 * math.pi
                    else -sdm.q_frozen_pair_formula(math.pi - th) for th in theta])
    return theta, val


def q_pair(theta: np.ndarray) -> np.ndarray:
    """Frozen completion-Q pair correlation, balanced first readout."""
    grid, val = _q_pair_table()
    return np.interp(theta, grid, val)


def d_template(phi: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(np.cos(phi))


def q_template(phi: np.ndarray) -> np.ndarray:
    """Deviation of the completion-Q pair correlation from quantum mechanics."""
    c = -np.cos(phi)
    return q_pair(np.arccos(np.clip(c, -1.0, 1.0))) - c


def _harmonics(phi: np.ndarray, orders: List[int]) -> List[np.ndarray]:
    cols = []
    for j in orders:
        if j == 0:
            cols.append(np.ones_like(phi))
        else:
            cols += [np.cos(j * phi), np.sin(j * phi)]
    return cols


def probe_fit(r: np.ndarray, phi: np.ndarray, max_order: int = 2) -> Dict[str, Tuple[float, float]]:
    """Shape test on a B, P, B, P chain.

    Pairs (B -> P) and (P -> B) are fitted together. Each pair type gets its
    own quantum-allowed nuisance: all harmonics of the probe phase up to
    max_order (2 for a qubit with one leakage level and a level-diagonal
    readout, 4 otherwise). The detector-anchored cusp is shared by both pair
    types; the qubit-anchored template enters the (B -> P) pairs only.
    Weighted least squares with variance 1 - mu^2 from a nuisance-only pass.
    Returns {name: (estimate, standard error)} for w_D and w_Q.
    """
    n, K = r.shape
    rows_y, rows_X = [], []
    orders = list(range(max_order + 1))
    n_nuis = 2 * max_order + 1
    for kind in ("BP", "PB"):
        ks = range(1, K, 2) if kind == "BP" else range(1, K - 1, 2)
        y = np.concatenate([r[:, k - 1] * r[:, k] if kind == "BP" else r[:, k] * r[:, k + 1] for k in ks])
        p = np.concatenate([phi[:, k] for k in ks])
        nuis = np.stack(_harmonics(p, orders), axis=1)
        block = np.zeros((len(y), 2 * n_nuis + 2))
        off = 0 if kind == "BP" else n_nuis
        block[:, off:off + n_nuis] = nuis
        block[:, -2] = d_template(p)
        if kind == "BP":
            block[:, -1] = q_template(p)
        rows_y.append(y)
        rows_X.append(block)
    y = np.concatenate(rows_y)
    X = np.concatenate(rows_X)
    nuis_only = X[:, :-2]
    mu = nuis_only @ np.linalg.lstsq(nuis_only, y, rcond=None)[0]
    wgt = 1.0 / np.clip(1.0 - mu ** 2, 0.02, None)
    G = X.T @ (X * wgt[:, None])
    coef = np.linalg.solve(G, X.T @ (wgt * y))
    cov = np.linalg.inv(G)
    resid = y - X @ coef
    scale = float(np.sum(wgt * resid ** 2) / (len(y) - X.shape[1]))
    se = np.sqrt(np.diag(cov) * scale)
    return {"w_D": (float(coef[-2]), float(se[-2])), "w_Q": (float(coef[-1]), float(se[-1]))}


def q_repeated_preparation(variant: str, n: int, rng) -> float:
    """Frozen completion-Q correlation of balanced repeated preparation, with
    re-preparation by pulses as on IBM hardware.

    sx | readout x1 | [unwinding sx^-1] | reset | sx | readout x2. variant
    "ibm reset": reset = readout (Q rule) followed by X if the outcome is 1,
    with lambda rotated by the X; "unwind, ibm reset": the same after an
    unwinding pulse; "unwind, pulse-free reset": the state is reset and lambda
    is left alone (the theory note's implicit assumption).
    """
    z = np.zeros((n, 3))
    z[:, 2] = 1.0

    def rot(v, sign):
        """sx (sign +1) or its inverse (sign -1): pi/2 about +x or -x."""
        return p_half(v, np.full(len(v), 0.0 if sign > 0 else math.pi))

    lam = rot(isotropic(n, rng), +1)
    s = rot(z, +1)
    x1 = np.where(lam[:, 2] + s[:, 2] > 0, 1.0, -1.0)
    s = x1[:, None] * z
    if variant.startswith("unwind"):
        lam, s = rot(lam, -1), rot(s, -1)
    if variant.endswith("ibm reset"):
        flip = lam[:, 2] + s[:, 2] <= 0
        lam[flip] = rot(rot(lam[flip], +1), +1)
    s = rot(z.copy(), +1)
    lam = rot(lam, +1)
    x2 = np.where(lam[:, 2] + s[:, 2] > 0, 1.0, -1.0)
    return float(np.mean(x1 * x2))


# ---------------------------------------------------------------- sensitivity

def shape_sensitivity(contrast: float = 0.95, grid: int = 4096) -> Dict[str, Dict[str, float]]:
    """sigma_w * sqrt(N) for the shape templates, with N probe records.

    (B -> P) pairs: per-pair variance 1 - (contrast cos phi)^2. (P -> B) pairs:
    variance 1. Nuisance: harmonics up to order 2 or 4 (all orders; the
    templates have definite parity, so only same-parity orders matter).
    """
    phi = np.linspace(0.0, TWO_PI, grid, endpoint=False)
    var_bp = 1.0 - (contrast * np.cos(phi)) ** 2
    out: Dict[str, Dict[str, float]] = {}

    def info(template: np.ndarray, var: np.ndarray, orders: List[int]) -> float:
        A = np.stack(_harmonics(phi, orders), axis=1)
        W = 1.0 / var
        coef = np.linalg.solve(A.T @ (A * W[:, None]), A.T @ (W * template))
        res = template - A @ coef
        return float(np.mean(res ** 2 * W))

    for k_max in (2, 4):
        orders = list(range(k_max + 1))
        i_d = info(d_template(phi), var_bp, orders) + info(d_template(phi), np.ones_like(phi), orders)
        i_q = info(q_template(phi), var_bp, orders)
        i_flat = info(d_template(phi), np.ones_like(phi), orders)
        out[f"max_order={k_max}"] = {
            "w_D_sigma_sqrtN": round(1.0 / math.sqrt(i_d), 2),
            "w_Q_sigma_sqrtN": round(1.0 / math.sqrt(i_q), 2),
            "note_flat_variance_w_D_sigma_sqrtN": round(1.0 / math.sqrt(i_flat), 2),
        }
    return out


# ---------------------------------------------------------------- budget

HARDWARE = {
    "source": "ibm_kingston fake-backend snapshot (qiskit-ibm-runtime 0.50.0) and IBM docs, checked 13 October 2026",
    "rep_delay_s": 250e-6,
    "measure_2_s": 1.76e-6,
    "sx_s": 32e-9,
    "subjob_overhead_s": 2.0,
    "open_plan_s_per_28_days": 600.0,
}

PLAN = [
    # label, kind, extra delay before each step (s), records per shot, QPU seconds
    # Job 1 (pilot): the full-size probe circuit at low shot count, plus C.
    ("pilot (B0 circuit, 64 sets x 300 shots)", "probe", 0.0, 64, 7.0),
    ("pilot amplitude check", "amplitude", 0.0, 1, 0.5),
    ("pilot phase check", "phase", 0.0, 1, 2.4),
    # Job 2 (main): all conditions as PUBs of one job, equal shots per PUB.
    ("A0 balanced, no added delay", "balanced", 0.0, 64, 60.0),
    ("A1 balanced, 4 us", "balanced", 4e-6, 64, 60.0),
    ("A2 balanced, 40 us", "balanced", 40e-6, 16, 60.0),
    ("B0 probe, no added delay", "probe", 0.0, 64, 180.0),
    ("C amplitude check", "amplitude", 0.0, 1, 0.5),
    ("C2 phase check", "phase", 0.0, 1, 3.2),
]
JOBS = 2


def budget(qubits: int, sens: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    hw = HARDWARE
    rows = []
    total = 0.0
    for label, kind, delay, K, seconds in PLAN:
        step = 2 * hw["sx_s"] + hw["measure_2_s"] + delay
        t_exec = hw["rep_delay_s"] + K * step
        if kind == "amplitude":
            t_exec = hw["rep_delay_s"] + 100 * hw["sx_s"] + hw["measure_2_s"]
        execs = seconds / t_exec
        row = {"label": label, "kind": kind, "records_per_shot": K, "added_delay_us": delay * 1e6,
               "qpu_s": seconds, "us_per_execution": round(t_exec * 1e6, 1), "executions": int(execs)}
        if kind == "balanced":
            pairs = execs * (K - 1) * qubits
            row["neighbor_pairs"] = f"{pairs:.2e}"
            row["five_sigma_bound_w_D"] = round(5.0 / math.sqrt(pairs), 5)
            row["five_sigma_bound_w1w2_Q"] = round(5.0 / math.sqrt(execs * (K - 2) * qubits), 5)
        elif kind == "probe":
            probes = execs * (K // 2) * qubits
            s2 = sens["max_order=2"]
            s4 = sens["max_order=4"]
            row["probe_records"] = f"{probes:.2e}"
            row["five_sigma_w_D_cusp"] = [round(5 * s2["w_D_sigma_sqrtN"] / math.sqrt(probes), 4),
                                          round(5 * s4["w_D_sigma_sqrtN"] / math.sqrt(probes), 4)]
            row["five_sigma_w_Q_shape"] = [round(5 * s2["w_Q_sigma_sqrtN"] / math.sqrt(probes), 4),
                                           round(5 * s4["w_Q_sigma_sqrtN"] / math.sqrt(probes), 4)]
        rows.append(row)
        total += seconds
    total += JOBS * hw["subjob_overhead_s"]
    return {"hardware": hw, "qubits_in_parallel": qubits, "conditions": rows,
            "total_qpu_s_with_overheads": round(total, 1),
            "fits_open_plan": total <= hw["open_plan_s_per_28_days"]}


# ---------------------------------------------------------------- checks

class Check:
    def __init__(self) -> None:
        self.count = 0

    def within(self, label: str, value: float, se: float, want: float, k: float = 5.0) -> None:
        self.count += 1
        if abs(value - want) > k * se + 1e-12:
            raise AssertionError(f"{label}: got {value:.5f} +- {se:.5f}, want {want:.5f}")

    def true(self, label: str, cond: bool) -> None:
        self.count += 1
        if not cond:
            raise AssertionError(label)


def run(output: str | None, qubits: int) -> int:
    rng = np.random.default_rng(20261013)
    chk = Check()
    table: Dict[str, Dict[str, object]] = {}

    # 1. Balanced chain: which statistic each model moves.
    n, K = 300_000, 8
    for model, w in (("QM", 0.0), ("D", 0.5), ("Q", 0.5)):
        al, be, _ = balanced_settings(n, K, rng)
        st = balanced_statistics(simulate(model, al, be, w, rng), al, be)
        chk.true(f"{model}: lag-2 axes return (t = +-1)", set(st["t_values"].tolist()) <= {-1.0, 1.0})
        want = {"QM": (0.0, 0.0, 0.0), "D": (w, w * w, 0.0), "Q": (0.0, 0.0, w * w)}[model]
        for key, target in zip(("d_lag1", "d_lag2", "q_lag2"), want):
            chk.within(f"balanced {model} {key}", *st[key], target)
        table[f"balanced chain, {model}" + (f" w={w}" if model != "QM" else "")] = {
            key: [round(st[key][0], 4), round(st[key][1], 4)] for key in ("d_lag1", "d_lag2", "q_lag2")}

    # 2. Conventional effects in the balanced chain (relabel bits random).
    eps = 0.06
    amp_term = math.cos(0.5 * math.pi * (1 + eps)) ** 2
    cases = {
        "pulse amplitude error 6%": dict(eps=eps),
        "T1 decay 10% per gap": dict(t1=0.10),
        "assignment error 1% / 3%": dict(e0=0.01, e1=0.03),
        "readout memory 2%": dict(e0=0.01, e1=0.03, mem=0.02),
        "all of the above": dict(eps=eps, t1=0.10, e0=0.01, e1=0.03, mem=0.02),
    }
    for name, kw in cases.items():
        al, be, _ = balanced_settings(n, K, rng)
        st = balanced_statistics(simulate("QM", al, be, 0.0, rng, **kw), al, be)
        chk.within(f"{name}: q_lag2 unmoved", *st["q_lag2"], 0.0)
        if name.startswith("pulse"):
            chk.within(f"{name}: d_lag1 equals cos^2 of the pulse angle", *st["d_lag1"], amp_term)
        elif name.startswith(("T1", "assignment")):
            chk.within(f"{name}: d_lag1 unmoved", *st["d_lag1"], 0.0)
        elif name.startswith("readout"):
            chk.true(f"{name}: d_lag1 moved", st["d_lag1"][0] > 5 * st["d_lag1"][1])
        table[f"balanced chain, QM with {name}"] = {
            key: [round(st[key][0], 4), round(st[key][1], 4)] for key in ("d_lag1", "d_lag2", "q_lag2")}

    # 3. Probe pairs at fixed relative phase: closed forms at w = 1.
    for phi0 in (0.6, 1.2, 1.9, 2.5):
        c = -math.cos(phi0)
        for model, want in (("QM", c), ("D", c + 1.0 - abs(c)), ("Q", float(q_pair(np.array([math.acos(c)]))[0]))):
            al, be, ph = probe_settings(200_000, 4, rng, phi_fixed=phi0)
            r = simulate(model, al, be, 1.0, rng)
            y = np.concatenate([r[:, 0] * r[:, 1], r[:, 2] * r[:, 3]])
            se = max(float(y.std() / math.sqrt(len(y))), 1e-4)
            chk.within(f"B->P pair {model} phi={phi0}", float(y.mean()), se, want)

    # 4. Shape fits on a probe chain.
    fits = {}
    n_p, K_p = 400_000, 8
    for label, model, w, kw in (("QM ideal", "QM", 0.0, {}),
                                ("QM with all conventional effects", "QM", 0.0,
                                 dict(eps=0.03, t1=0.05, e0=0.01, e1=0.03, mem=0.02)),
                                ("D w=0.3", "D", 0.3, {}), ("Q w=0.3", "Q", 0.3, {})):
        al, be, ph = probe_settings(n_p, K_p, rng)
        fit = probe_fit(simulate(model, al, be, w, rng, **kw), ph)
        want_d = w if model == "D" else 0.0
        want_q = w if model == "Q" else 0.0
        chk.within(f"shape fit {label}: w_D", *fit["w_D"], want_d)
        chk.within(f"shape fit {label}: w_Q", *fit["w_Q"], want_q)
        fits[label] = {k: [round(v[0], 4), round(v[1], 4)] for k, v in fit.items()}

    # 5. Completion Q through re-preparation made of pulses (Section 2 of the
    #    design note): zero through IBM-style resets, w only if lambda is left alone.
    q_reset = {}
    n_r = 400_000
    for variant, want in (("ibm reset", 0.0), ("unwind, ibm reset", 0.0), ("unwind, pulse-free reset", 1.0)):
        val = q_repeated_preparation(variant, n_r, rng)
        chk.within(f"Q repeated preparation, {variant}", val, 1.0 / math.sqrt(n_r), want)
        q_reset[variant] = round(val, 4)

    # 6. Sensitivities; the flat-variance cusp value must match the note.
    sens = shape_sensitivity()
    flat = sens["max_order=2"]["note_flat_variance_w_D_sigma_sqrtN"]
    chk.true("flat-variance cusp sensitivity matches the theory note (0.073 at 1e6)",
             abs(5 * flat / 1000.0 - 0.073) < 0.0015)

    plan = budget(qubits, sens)
    chk.true("plan fits the open-plan allowance", bool(plan["fits_open_plan"]))

    result = {
        "status": "PASS",
        "checks": chk.count,
        "what_each_statistic_sees": table,
        "shape_fits_on_simulated_probe_chains": fits,
        "q_frozen_repeated_preparation_by_reset_variant": q_reset,
        "shape_sensitivity_sigma_w_sqrtN": sens,
        "budget": plan,
        "not_checked": [
            "leakage to |2> and higher levels (covered only by the harmonic-degree bound)",
            "measurement-induced state transitions and non-QND readout",
            "crosstalk between qubits read in parallel",
            "drifts within a job",
            "diffusive hidden-variable dynamics",
        ],
        "note": "Simulated design checks only; nothing here is experimental data.",
    }
    text = json.dumps(result, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(f"PASS: {chk.count} checks")
    print(json.dumps({"budget": plan["conditions"], "total_s": plan["total_qpu_s_with_overheads"]}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="write results as JSON")
    parser.add_argument("--qubits", type=int, default=6, help="qubits read in parallel (default 6)")
    args = parser.parse_args()
    try:
        return run(args.output, args.qubits)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
