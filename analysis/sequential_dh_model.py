#!/usr/bin/env python3
"""Consistency checks for the sequential extension of the Donadi-Hossenfelder
two-outcome rule (theory note, theory/theorynote.pdf).

Two ordering-free completions of the rule "outcome 1 iff u < Born probability":

  qubit-anchored (Q):    hidden unit vector lam, isotropic on the sphere and
                         transformed by the same pulses as the qubit; outcome
                         x = sgn(lam.n + s.n) for Bloch vector s and axis n.
  detector-anchored (D): hidden u, uniform on [0, 1]; the eigenstate sigma*n is
                         mapped to the bright detector outcome; bright iff
                         u < (1 + sigma s.n)/2, and x = sigma if bright.

Renewal persistence: between consecutive measurements the hidden variable is
kept with probability w (gap persistence) and otherwise redrawn independently.

The "prep" operation sets the qubit state and leaves lam alone. For model Q
that is a transparent re-preparation: the preparation pulse follows a reset
that applies no pulse, with its inverse applied before that reset. Section 10
of the checks covers re-preparation made of pulses, which turns lam.

Frozen-block statistics are computed exactly (model D, interval arithmetic) or
by one-dimensional quadrature over lam_z with the exact arcsine law for lam_x
(model Q), and cross-checked by Monte Carlo. The script exits nonzero on any
failed check. It validates formulas only; nothing here is experimental data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

Vec = Tuple[float, float, float]
Op = Tuple  # ("prep", s) or ("meas", n, sigma)

Z: Vec = (0.0, 0.0, 1.0)
X: Vec = (1.0, 0.0, 0.0)


def axis(theta: float) -> Vec:
    """Unit vector in the x-z plane at polar angle theta from Z."""
    return (math.sin(theta), 0.0, math.cos(theta))


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def scale(k: float, a: Vec) -> Vec:
    return (k * a[0], k * a[1], k * a[2])


# ---------------------------------------------------------------- frozen blocks

def frozen_block_D(state: Vec, ops: Sequence[Op]) -> Dict[Tuple[int, ...], Tuple[float, Vec]]:
    """Exact joint law of one frozen block for the detector-anchored model.

    Returns {outcomes: (probability, final state)}; u is uniform on [0, 1].
    """
    items = [((0.0, 1.0), state, ())]
    for op in ops:
        if op[0] == "prep":
            items = [(iv, op[1], out) for iv, _, out in items]
            continue
        _, n, sigma = op
        new = []
        for (lo, hi), s, out in items:
            p_bright = 0.5 * (1.0 + sigma * dot(s, n))
            cut = min(max(p_bright, lo), hi)
            if cut > lo:
                new.append(((lo, cut), scale(sigma, n), out + (sigma,)))
            if hi > cut:
                new.append(((cut, hi), scale(-sigma, n), out + (-sigma,)))
        items = new
    law: Dict[Tuple[int, ...], Tuple[float, Vec]] = {}
    for (lo, hi), s, out in items:
        p, _ = law.get(out, (0.0, s))
        law[out] = (p + (hi - lo), s)
    return law


def _arcsine_cdf(x: float, rho: float) -> float:
    if rho <= 0.0:
        return 0.0 if x < 0.0 else 1.0
    t = min(max(x / rho, -1.0), 1.0)
    return 1.0 - math.acos(t) / math.pi


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)
_GL400 = np.polynomial.legendre.leggauss(400)


def frozen_block_Q(state: Vec, ops: Sequence[Op], panels: int = 1000) -> Dict[Tuple[int, ...], Tuple[float, Vec]]:
    """Joint law of one frozen block for the qubit-anchored model.

    All states and axes must lie in the x-z plane. For lam_z = a the component
    lam_x is arcsine-distributed on [-rho, rho], rho = sqrt(1 - a^2); the block
    is split exactly in lam_x and integrated over a by composite Gauss-Legendre.
    """
    for op in ops:
        vec = op[1]
        if abs(vec[1]) > 1e-12:
            raise ValueError("model Q evaluator needs x-z plane vectors")
    if abs(state[1]) > 1e-12:
        raise ValueError("model Q evaluator needs x-z plane vectors")

    # Panel edges include every lam_z at which a threshold meets the edge of
    # the lam_x range (square-root kinks) or a Z-axis test changes sign, so the
    # quadrature never straddles a singular point.
    axes = [op[1] for op in ops if op[0] == "meas"]
    states = [state] + [op[1] for op in ops if op[0] == "prep"]
    states += [scale(k, n) for n in axes for k in (1, -1)]
    breaks = {-1.0, 1.0}
    for n in axes:
        for s in states:
            k = dot(s, n)
            if abs(n[0]) < 1e-15:
                if abs(n[2]) > 0:
                    breaks.add(-k / n[2])
                continue
            root = abs(n[0]) * math.sqrt(max(0.0, 1.0 - k * k))
            breaks.update((-n[2] * k - root, -n[2] * k + root))
    knots = sorted(b for b in breaks if -1.0 <= b <= 1.0)
    edges = [knots[0]]
    for lo, hi in zip(knots[:-1], knots[1:]):
        if hi - lo < 1e-14:
            continue
        m = max(2, int(math.ceil(panels * (hi - lo) / 2.0)))
        edges.extend(np.linspace(lo, hi, m + 1)[1:])
    edges = np.array(edges)
    law: Dict[Tuple[int, ...], List] = {}
    for left, right in zip(edges[:-1], edges[1:]):
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            a = mid + half * node
            rho = math.sqrt(max(0.0, 1.0 - a * a))
            items = [((-rho, rho), state, ())]
            for op in ops:
                if op[0] == "prep":
                    items = [(iv, op[1], out) for iv, _, out in items]
                    continue
                _, n, _sigma = op
                new = []
                for (lo, hi), s, out in items:
                    offset = n[2] * a + dot(s, n)
                    if abs(n[0]) < 1e-15:
                        x = 1 if offset > 0 else -1
                        new.append(((lo, hi), scale(x, n), out + (x,)))
                        continue
                    cut = min(max(-offset / n[0], lo), hi)
                    plus_side = (cut, hi) if n[0] > 0 else (lo, cut)
                    minus_side = (lo, cut) if n[0] > 0 else (cut, hi)
                    if plus_side[1] > plus_side[0]:
                        new.append((plus_side, scale(1, n), out + (1,)))
                    if minus_side[1] > minus_side[0]:
                        new.append((minus_side, scale(-1, n), out + (-1,)))
                items = new
            for (lo, hi), s, out in items:
                mass = _arcsine_cdf(hi, rho) - _arcsine_cdf(lo, rho)
                entry = law.setdefault(out, [0.0, s])
                entry[0] += 0.5 * half * weight * mass
    return {out: (p, s) for out, (p, s) in law.items()}


# ---------------------------------------------------------------- renewal mixture

def split_measurements(ops: Sequence[Op]) -> List[List[Op]]:
    """Group ops so that each segment ends with exactly one measurement."""
    segments, current = [], []
    for op in ops:
        current.append(op)
        if op[0] == "meas":
            segments.append(current)
            current = []
    if current:
        raise ValueError("sequence must end with a measurement")
    return segments


def renewal_law(model: str, state: Vec, ops: Sequence[Op], gaps: Sequence[float]) -> Dict[Tuple[int, ...], float]:
    """Joint outcome law with gap persistence probabilities `gaps`."""
    segments = split_measurements(ops)
    if len(gaps) != len(segments) - 1:
        raise ValueError("need one persistence probability per gap")
    block = frozen_block_D if model == "D" else frozen_block_Q
    total: Dict[Tuple[int, ...], float] = {}
    for pattern in itertools.product((True, False), repeat=len(gaps)):
        weight = 1.0
        for keep, w in zip(pattern, gaps):
            weight *= w if keep else 1.0 - w
        if weight == 0.0:
            continue
        blocks, current = [], list(segments[0])
        for keep, seg in zip(pattern, segments[1:]):
            if keep:
                current.extend(seg)
            else:
                blocks.append(current)
                current = list(seg)
        blocks.append(current)
        partial = {(): (1.0, state)}
        for ops_block in blocks:
            nxt: Dict[Tuple[int, ...], Tuple[float, Vec]] = {}
            for out, (p, s) in partial.items():
                for out_b, (p_b, s_b) in block(s, ops_block).items():
                    key = out + out_b
                    q, _ = nxt.get(key, (0.0, s_b))
                    nxt[key] = (q + p * p_b, s_b)
            partial = nxt
        for out, (p, _) in partial.items():
            total[out] = total.get(out, 0.0) + weight * p
    return total


def expect(law: Dict[Tuple[int, ...], float], idx: Sequence[int]) -> float:
    return sum(p * math.prod(out[i] for i in idx) for out, p in law.items())


def carry_over(law: Dict[Tuple[int, ...], float], first: int, last: int) -> float:
    """D = [E(x_last | x_first = +1) - E(x_last | x_first = -1)] / 2."""
    num = {1: 0.0, -1: 0.0}
    den = {1: 0.0, -1: 0.0}
    for out, p in law.items():
        num[out[first]] += p * out[last]
        den[out[first]] += p
    return 0.5 * (num[1] / den[1] - num[-1] / den[-1])


# ---------------------------------------------------------------- sequences

def triad(theta: float, sig: Tuple[int, int, int] = (1, 1, 1)) -> List[Op]:
    return [("meas", Z, sig[0]), ("meas", axis(theta), sig[1]), ("meas", Z, sig[2])]


def causal_break(theta: float, s0: Vec, sig: Tuple[int, int, int] = (1, 1, 1)) -> List[Op]:
    return [("meas", Z, sig[0]), ("prep", s0), ("meas", axis(theta), sig[1]), ("meas", Z, sig[2])]


def pair(theta: float) -> List[Op]:
    return [("meas", Z, 1), ("meas", axis(theta), 1)]


def repeated_prep(s0: Vec, sig: Tuple[int, int] = (1, 1)) -> List[Op]:
    return [("meas", Z, sig[0]), ("prep", s0), ("meas", Z, sig[1])]


def cusp_scan(phi: float, sig: Tuple[int, int] = (1, 1)) -> List[Op]:
    """Repeated preparation whose second state sits at polar angle phi."""
    return [("meas", Z, sig[0]), ("prep", axis(phi)), ("meas", Z, sig[1])]


# ---------------------------------------------------------------- conventional memory

def _random_unitary(dim: int, rng) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_kraus(dim: int, outputs: int, rng) -> List[np.ndarray]:
    """Kraus operators of a random instrument or channel (outputs blocks)."""
    iso = _random_unitary(dim * outputs, rng)[:, :dim]
    return [iso[k * dim:(k + 1) * dim, :] for k in range(outputs)]


def conventional_memory_law(d: int, phis: np.ndarray, rng, mode: str = "rotation") -> np.ndarray:
    """P(b1, b2 | phi) for a random quantum model with memory.

    A d-level system (qubit plus d - 2 leakage levels) and a two-level memory:
    random preparation, random joint readout instrument, random joint reset
    channel, a second preparation in which phi enters only through the frame
    phase F = exp(-i phi n), and a second readout instrument. Modes:
      "rotation":        F acts once between two fixed pulses;
      "virtual_diag":    F shifts the later pulse (F P F^dag) and the second
                         readout is diagonal in the level basis;
      "virtual_general": F shifts the later pulse and the second readout is a
                         random joint instrument.
    Returns an array of shape (len(phis), 2, 2).
    """
    da = 2
    dim = d * da
    rho_a = _random_kraus(da, 1, rng)[0]
    rho_a = rho_a @ rho_a.conj().T
    rho_a /= np.trace(rho_a)
    ground = np.zeros((d, d), complex)
    ground[0, 0] = 1.0
    rho = np.kron(ground, rho_a)
    u1 = np.kron(_random_unitary(d, rng), np.eye(da))
    rho = u1 @ rho @ u1.conj().T
    read1 = _random_kraus(dim, 2, rng)
    reset = _random_kraus(dim, 3, rng)
    pre = np.kron(_random_unitary(d, rng), np.eye(da))
    post = np.kron(_random_unitary(d, rng), np.eye(da))
    if mode == "virtual_diag":
        read2 = [np.zeros((dim, dim), complex) for _ in range(2)]
        for j in range(d):
            proj = np.zeros((d, d))
            proj[j, j] = 1.0
            for b, m in enumerate(_random_kraus(da, 2, rng)):
                read2[b] += np.kron(proj, m)
    else:
        read2 = _random_kraus(dim, 2, rng)
    charge = np.kron(np.arange(d, dtype=float), np.ones(da))
    out = np.zeros((len(phis), 2, 2))
    for b1, k1 in enumerate(read1):
        after = k1 @ rho @ k1.conj().T
        after = sum(k @ after @ k.conj().T for k in reset)
        after = pre @ after @ pre.conj().T
        for i, phi in enumerate(phis):
            frame = np.diag(np.exp(-1j * phi * charge))
            if mode == "rotation":
                step = post @ frame
            else:
                step = frame @ post @ frame.conj().T
            state = step @ after @ step.conj().T
            for b2, k2 in enumerate(read2):
                out[i, b1, b2] = float(np.real(np.trace(k2 @ state @ k2.conj().T)))
    return out


# ---------------------------------------------------------------- Monte Carlo

def monte_carlo(model: str, state: Vec, ops: Sequence[Op], gaps: Sequence[float], n: int, rng) -> np.ndarray:
    """Simulate outcome records; returns an (n, K) array of +-1."""
    def draw():
        if model == "D":
            return rng.random(n)
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    hidden = draw()
    s = np.tile(np.array(state), (n, 1))
    outs = []
    meas_seen = 0
    for op in ops:
        if op[0] == "prep":
            s = np.tile(np.array(op[1]), (n, 1))
            continue
        if meas_seen > 0:
            redraw = rng.random(n) >= gaps[meas_seen - 1]
            fresh = draw()
            if model == "D":
                hidden = np.where(redraw, fresh, hidden)
            else:
                hidden = np.where(redraw[:, None], fresh, hidden)
        _, nvec, sigma = op
        nv = np.array(nvec)
        sn = s @ nv
        if model == "D":
            bright = hidden < 0.5 * (1.0 + sigma * sn)
            x = np.where(bright, sigma, -sigma)
        else:
            x = np.where(hidden @ nv + sn > 0, 1, -1)
        outs.append(x)
        s = x[:, None] * nv[None, :]
        meas_seen += 1
    return np.stack(outs, axis=1)


# ---------------------------------------------------------------- checks

class Check:
    def __init__(self) -> None:
        self.count = 0
        self.worst = 0.0

    def close(self, label: str, got: float, want: float, tol: float) -> None:
        self.count += 1
        err = abs(got - want)
        self.worst = max(self.worst, err) if tol <= 1e-6 else self.worst
        if err > tol:
            raise AssertionError(f"{label}: got {got:.9f}, want {want:.9f} (tol {tol:g})")


def q_frozen_triad_formula(theta: float) -> float:
    """C = 1 - 4 P(0 < a < c, b < -c) for model Q, frozen, balanced start."""
    c, s = math.cos(theta), math.sin(theta)
    a_max = min(c, s * s - c * c)
    if a_max <= 0.0 or s == 0.0:
        return 1.0
    nodes, weights = _GL400
    a = 0.5 * a_max * (nodes + 1.0)
    arg = np.clip((c / s) * np.sqrt((1.0 + a) / (1.0 - a)), -1.0, 1.0)
    prob = 0.5 * a_max * np.sum(weights * np.arccos(arg)) / (2.0 * math.pi)
    return 1.0 - 4.0 * prob


def q_frozen_pair_formula(theta: float) -> float:
    """E(x1 x2) = 1 - 4 P(a > 0, b < -c) for model Q, frozen, balanced start."""
    c, s = math.cos(theta), math.sin(theta)
    a_max = min(1.0, s * s - c * c)
    if a_max <= 0.0 or s == 0.0:
        return 1.0
    nodes, weights = _GL400
    a = 0.5 * a_max * (nodes + 1.0)
    arg = np.clip((c / s) * np.sqrt((1.0 + a) / (1.0 - a)), -1.0, 1.0)
    prob = 0.5 * a_max * np.sum(weights * np.arccos(arg)) / (2.0 * math.pi)
    return 1.0 - 4.0 * prob


def run(output: str | None) -> int:
    rng = np.random.default_rng(20261011)
    chk = Check()
    tol_exact = 1e-12
    tol_quad = 1e-6
    plus_x = X

    # 1. Born marginals for one measurement on a fresh hidden variable.
    for _ in range(40):
        t_s, t_n = rng.uniform(0, math.pi, size=2)
        s, n = axis(t_s), axis(t_n)
        born = 0.5 * (1.0 + dot(s, n))
        for model, tol in (("D", tol_exact), ("Q", tol_quad)):
            law = renewal_law(model, s, [("meas", n, 1)], [])
            chk.close(f"Born {model}", law.get((1,), 0.0), born, tol)

    # 2. No persistence reproduces the ideal triplet law and a zero carry-over.
    for theta in np.linspace(0.05, math.pi / 2, 7):
        c = math.cos(theta)
        for model, tol in (("D", tol_exact), ("Q", tol_quad)):
            law = renewal_law(model, plus_x, triad(theta), [0.0, 0.0])
            for out, p in law.items():
                z0, r, z2 = out
                chk.close(f"ideal triplet {model}", p, 0.5 * 0.5 * (1 + z0 * r * c) * 0.5 * (1 + r * z2 * c), tol)
            law = renewal_law(model, plus_x, causal_break(theta, plus_x), [0.0, 0.0])
            chk.close(f"null carry-over {model}", carry_over(law, 0, 2), 0.0, tol)

    # 3. Frozen orthogonal-setting closed forms, general start c0.
    for t0 in np.linspace(0.0, math.pi, 9):
        s0 = axis(t0)
        c0 = math.cos(t0)
        for sigma in (1, -1):
            law = renewal_law("D", s0, triad(math.pi / 2, (1, sigma, 1)), [1.0, 1.0])
            chk.close("D frozen C(pi/2)", expect(law, (0, 2)), 1 - abs(c0), tol_exact)
            chk.close("D frozen E(z0 r)", expect(law, (0, 1)), sigma * (1 - abs(c0)), tol_exact)
            chk.close("D frozen E(r z2)", expect(law, (1, 2)), sigma, tol_exact)
        law = renewal_law("Q", s0, triad(math.pi / 2), [1.0, 1.0])
        chk.close("Q frozen C(pi/2)", expect(law, (0, 2)), 1 - abs(c0), tol_quad)
        chk.close("Q frozen E(z0 r)", expect(law, (0, 1)), 0.0, tol_quad)
        chk.close("Q frozen E(r z2)", expect(law, (1, 2)), 0.0, tol_quad)

    # 4. Renewal at the orthogonal setting, balanced start.
    for w1, w2 in [(0.3, 0.7), (0.05, 0.05), (0.9, 0.2)]:
        for model, tol in (("D", tol_exact), ("Q", tol_quad)):
            law = renewal_law(model, plus_x, triad(math.pi / 2), [w1, w2])
            chk.close(f"renewal C {model}", expect(law, (0, 2)), w1 * w2, tol)
            law = renewal_law(model, plus_x, causal_break(math.pi / 2, plus_x), [w1, w2])
            chk.close(f"renewal causal break {model}", carry_over(law, 0, 2), w1 * w2, tol)
        law = renewal_law("D", plus_x, triad(math.pi / 2), [w1, w2])
        chk.close("renewal D E(z0 r)", expect(law, (0, 1)), w1, tol_exact)
        chk.close("renewal D E(r z2)", expect(law, (1, 2)), w2, tol_exact)

    # 5. Repeated preparation and the readout-relabeling test.
    for w in (0.0, 0.01, 0.4, 1.0):
        for sig2 in (1, -1):
            law = renewal_law("D", plus_x, repeated_prep(plus_x, (1, sig2)), [w])
            chk.close("repeated prep D", expect(law, (0, 1)), sig2 * w, tol_exact)
            law = renewal_law("Q", plus_x, repeated_prep(plus_x, (1, sig2)), [w])
            chk.close("repeated prep Q", expect(law, (0, 1)), w, tol_quad)
        law = renewal_law("D", plus_x, triad(math.pi / 2, (1, 1, -1)), [w, w])
        chk.close("relabeled endpoint D", expect(law, (0, 2)), -w * w, tol_exact)

    # 6. Model Q frozen formulas at general angle.
    for theta in np.linspace(0.02, math.pi / 2, 12):
        law = renewal_law("Q", plus_x, triad(theta), [1.0, 1.0])
        chk.close("Q frozen triad formula", expect(law, (0, 2)), q_frozen_triad_formula(theta), tol_quad)
        law = renewal_law("Q", plus_x, pair(theta), [1.0])
        chk.close("Q frozen pair formula", expect(law, (0, 1)), q_frozen_pair_formula(theta), tol_quad)
        if theta <= math.pi / 4:
            chk.close("Q frozen pair = 1 below pi/4", expect(law, (0, 1)), 1.0, tol_quad)

    # 7. Randomized readout labels (each sigma independent and uniform).
    def label_average(model, s0, make, k, gaps):
        sigs = list(itertools.product((1, -1), repeat=k))
        avg: Dict[Tuple[int, ...], float] = {}
        for sg in sigs:
            for out, p in renewal_law(model, s0, make(sg), gaps).items():
                avg[out] = avg.get(out, 0.0) + p / len(sigs)
        return avg

    for w in (0.2, 1.0):
        law = label_average("D", plus_x, lambda sg: repeated_prep(plus_x, sg), 2, [w])
        chk.close("random labels remove D (repeated prep)", expect(law, (0, 1)), 0.0, tol_exact)
        law = label_average("D", plus_x, lambda sg: triad(math.pi / 2, sg), 3, [w, w])
        for idx in ((0, 1), (1, 2), (0, 2)):
            chk.close("random labels remove D (orthogonal triad)", expect(law, idx), 0.0, tol_exact)
        law = label_average("Q", plus_x, lambda sg: repeated_prep(plus_x, sg), 2, [w])
        chk.close("random labels keep Q (repeated prep)", expect(law, (0, 1)), w, tol_quad)
    for theta in np.linspace(0.1, math.pi / 2, 6):
        c = math.cos(theta)
        law = label_average("D", plus_x, lambda sg: triad(theta, sg), 3, [1.0, 1.0])
        chk.close("random labels D neighbor", expect(law, (0, 1)), c, tol_exact)
        chk.close("random labels D neighbor (second)", expect(law, (1, 2)), c, tol_exact)
        chk.close("random labels D endpoint residual", expect(law, (0, 2)), c, tol_exact)

    # 8. Shape test: repeated preparation with the second state at angle phi.
    def frame_cov(model, phi, w, randomize, frame):
        """Covariance of the two outcomes in logical or physical labels."""
        sigs = list(itertools.product((1, -1), repeat=2)) if randomize else [(1, 1)]
        mix: Dict[Tuple[int, int], float] = {}
        for sg in sigs:
            for (x1, x2), p in renewal_law(model, plus_x, cusp_scan(phi, sg), [w]).items():
                key = (x1, x2) if frame == "logical" else (sg[0] * x1, sg[1] * x2)
                mix[key] = mix.get(key, 0.0) + p / len(sigs)
        return expect(mix, (0, 1)) - expect(mix, (0,)) * expect(mix, (1,)), expect(mix, (1,))

    cusp = lambda phi: 1.0 - abs(math.cos(phi))
    for phi in np.linspace(0.0, math.pi, 9):
        for w in (0.3, 1.0):
            for model, tol in (("D", tol_exact), ("Q", tol_quad)):
                cov, marg = frame_cov(model, phi, w, False, "logical")
                chk.close(f"cusp {model} fixed labels", cov, w * cusp(phi), tol)
                chk.close(f"cusp {model} Born marginal", marg, math.cos(phi), tol)
            cov, _ = frame_cov("Q", phi, w, True, "logical")
            chk.close("cusp Q random labels, logical", cov, w * cusp(phi), tol_quad)
            cov, _ = frame_cov("Q", phi, w, True, "physical")
            chk.close("cusp Q random labels, physical", cov, 0.0, tol_quad)
            cov, _ = frame_cov("D", phi, w, True, "logical")
            chk.close("cusp D random labels, logical", cov, 0.0, tol_exact)
            cov, _ = frame_cov("D", phi, w, True, "physical")
            chk.close("cusp D random labels, physical", cov, w * cusp(phi), tol_exact)

    # Bounded harmonic degree of any quantum model with memory: harmonics of phi
    # above d - 1 vanish when phi enters only as a frame phase on d levels.
    # A virtual Z rotation shifts later pulses instead; one conjugation survives
    # when the record's last readout is diagonal in the level basis, and the
    # bound doubles to 2(d - 1) for a general readout.
    grid = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    for mode, bound in (("rotation", lambda d: d - 1), ("virtual_diag", lambda d: d - 1),
                        ("virtual_general", lambda d: 2 * (d - 1))):
        for d in (2, 3):
            k_max = bound(d)
            for _ in range(10):
                law = conventional_memory_law(d, grid, rng, mode)
                spectrum = np.abs(np.fft.rfft(law, axis=0)) / len(grid)
                chk.close(f"{mode}: no harmonics above {k_max} (d={d})",
                          float(spectrum[k_max + 1:].max()), 0.0, 1e-12)
                if spectrum[k_max].max() < 1e-6:
                    raise AssertionError(f"{mode}: harmonic {k_max} unexpectedly absent for d={d}")
                chk.count += 1

    # Harmonic content of the cusp beyond a quantum-allowed degree K.
    fine = np.linspace(0.0, 2.0 * math.pi, 1 << 16, endpoint=False)
    amp = 2.0 * np.abs(np.fft.rfft(1.0 - np.abs(np.cos(fine)))) / len(fine)
    shape = {}
    for k_max in (2, 4):
        rms = math.sqrt(0.5 * float(np.sum(amp[k_max + 1:] ** 2)))
        series = math.sqrt(0.5 * sum((4.0 / (math.pi * (4 * j * j - 1))) ** 2
                                     for j in range(k_max // 2 + 1, 200000)))
        chk.close(f"cusp residual rms K={k_max}", rms, series, 1e-6)
        shape[f"K={k_max}"] = {
            "residual_rms_per_unit_w": round(rms, 4),
            "five_sigma_w": {f"N=1e{e}": round(5.0 / (rms * math.sqrt(10.0 ** e)), 4) for e in (6, 7, 8)},
        }

    # 9. Monte Carlo cross-check of the exact and quadrature laws.
    n_mc = 400_000
    for model in ("D", "Q"):
        for theta, gaps in [(math.pi / 3, [0.6, 0.5]), (3 * math.pi / 8, [1.0, 1.0])]:
            for ops in (triad(theta, (1, -1, 1)), causal_break(theta, plus_x)):
                law = renewal_law(model, plus_x, ops, gaps)
                rec = monte_carlo(model, plus_x, ops, gaps, n_mc, rng)
                for idx in ((0, 1), (1, 2), (0, 2)):
                    want = expect(law, idx)
                    got = float(np.mean(np.prod(rec[:, list(idx)], axis=1)))
                    chk.close(f"MC {model} {idx}", got, want, 5.0 / math.sqrt(n_mc))

    # 10. Model Q when pulses turn lam (theory note v0.2).
    # (a) Balanced readouts with the second axis turned by chi:
    #     E = w (1 - 2 chi / pi); model D keeps w for any chi.
    for chi in np.linspace(0.0, math.pi, 9):
        n2 = axis(chi)
        ops = [("meas", Z, 1), ("prep", axis(chi + 0.5 * math.pi)), ("meas", n2, 1)]
        for w in (0.3, 1.0):
            law = renewal_law("Q", plus_x, ops, [w])
            chk.close("Q turned axis", expect(law, (0, 1)), w * (1.0 - 2.0 * chi / math.pi), tol_quad)
            law = renewal_law("D", plus_x, ops, [w])
            chk.close("D turned axis", expect(law, (0, 1)), w, tol_exact)
    # (b) Pulses alone to a balanced second readout: the pair at pi/2 is zero.
    law = renewal_law("Q", plus_x, pair(math.pi / 2), [1.0])
    chk.close("Q pulse re-preparation to balanced", expect(law, (0, 1)), 0.0, tol_quad)
    # (c) Relabeling pi pulses that are not undone: logical labels lose Q,
    #     frame-tracked labels keep it (Monte Carlo, three-dimensional lam).
    def q_relabel(n, w, rng):
        lam = rng.normal(size=(n, 3))
        lam /= np.linalg.norm(lam, axis=1, keepdims=True)
        sig = np.where(rng.random((n, 2)) < 0.5, 1.0, -1.0)
        flip = np.array([1.0, -1.0, -1.0])        # pi pulse about x
        lam1 = np.where(sig[:, :1] < 0, lam * flip, lam)
        x1 = sig[:, 0] * np.sign(lam1[:, 2])      # balanced start +x has s_z = 0
        keep = rng.random(n) < w                  # transparent re-preparation to +x
        fresh = rng.normal(size=(n, 3))
        fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
        lam2 = np.where(keep[:, None], lam1, fresh)
        lam2 = np.where(sig[:, 1:] < 0, lam2 * flip, lam2)
        x2 = sig[:, 1] * np.sign(lam2[:, 2])
        return float(np.mean(x1 * x2)), float(np.mean(sig[:, 0] * x1 * x2))
    for w in (0.4, 1.0):
        logical, tracked = q_relabel(n_mc, w, rng)
        chk.close("Q relabel pulses not undone: logical labels", logical, 0.0, 5.0 / math.sqrt(n_mc))
        chk.close("Q relabel pulses not undone: frame-tracked labels", tracked, w, 5.0 / math.sqrt(n_mc))
    # (d) Harmonic content of the Q pair deviation q(theta) = E_Q - cos(theta),
    #     with cos(theta) = -cos(phi) for two pi/2 pulses at relative phase phi.
    theta_grid = np.linspace(0.0, math.pi, 4001)
    pair_val = np.array([q_frozen_pair_formula(t) if t <= 0.5 * math.pi
                         else -q_frozen_pair_formula(math.pi - t) for t in theta_grid])
    kink = {}
    rms_by_grid = []
    for m_exp in (12, 14):
        fine_phi = np.linspace(0.0, 2.0 * math.pi, 1 << m_exp, endpoint=False)
        c_phi = -np.cos(fine_phi)
        dev = np.interp(np.arccos(c_phi), theta_grid, pair_val) - c_phi
        amp_q = 2.0 * np.abs(np.fft.rfft(dev)) / len(fine_phi)
        chk.close("Q deviation has only odd harmonics", float(np.max(amp_q[2:40:2])), 0.0, 1e-6)
        rms_by_grid.append([math.sqrt(0.5 * float(np.sum(amp_q[k + 1:] ** 2))) for k in (2, 4)])
    for i, k_max in enumerate((2, 4)):
        chk.close(f"Q deviation rms K={k_max} grid-independent", rms_by_grid[0][i], rms_by_grid[1][i], 1e-4)
        rms = rms_by_grid[1][i]
        kink[f"K={k_max}"] = {
            "residual_rms_per_unit_w": round(rms, 4),
            "five_sigma_w": {f"N=1e{e}": round(5.0 / (rms * math.sqrt(10.0 ** e)), 4) for e in (6, 7, 8)},
        }
    kink["first_and_third_harmonic"] = [round(float(amp_q[1]), 4), round(float(amp_q[3]), 4)]

    # Table for the note: frozen limit, balanced start (+x), all sigma = +1.
    table = []
    for label, theta in [("0", 0.0), ("pi/8", math.pi / 8), ("pi/4", math.pi / 4),
                         ("3pi/8", 3 * math.pi / 8), ("pi/2", math.pi / 2)]:
        c = math.cos(theta)
        row = {"theta": label, "qm": {"E_z0r": c, "C_zz": c * c, "D": 0.0}}
        for model in ("Q", "D"):
            tri = renewal_law(model, plus_x, triad(theta), [1.0, 1.0])
            cb = renewal_law(model, plus_x, causal_break(theta, plus_x), [1.0, 1.0])
            row[model] = {
                "E_z0r": round(expect(tri, (0, 1)), 4),
                "E_rz2": round(expect(tri, (1, 2)), 4),
                "C_zz": round(expect(tri, (0, 2)), 4),
                "D": round(carry_over(cb, 0, 2), 4),
            }
        table.append(row)

    result = {
        "status": "PASS",
        "scope": "Analytic and numerical checks of a model constructed for the theory note; no experimental data",
        "checks": chk.count,
        "monte_carlo_samples_per_case": n_mc,
        "frozen_limit_table_balanced_start": table,
        "shape_test_cusp_1_minus_abs_cos_phi": shape,
        "shape_test_q_pair_deviation_pulse_repreparation": kink,
        "not_checked": [
            "any laboratory hardware or measured record",
            "whether either completion is the one Donadi and Hossenfelder would endorse",
            "diffusive (non-renewal) hidden-variable dynamics",
        ],
    }
    print(json.dumps(result, indent=2))
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="write the JSON summary to this file")
    args = parser.parse_args()
    try:
        return run(args.output)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
