#!/usr/bin/env python3
"""Repetition-code impact of the measured readout memory and leakage (simulation, not data).

What does readout memory of the size measured in the E0 and E0R runs, and
leakage episodes of the measured kind, do to a bit-flip repetition code decoded
by minimum-weight perfect matching that assumes independent measurement errors?

Model (a phenomenological, classical simulation; the code only sees bit flips):
  - d data qubits, d - 1 ancillas, T = d noisy syndrome rounds and a final
    perfect data readout (memory experiment). Each round every data qubit flips
    with probability p. Each ancilla is reset and reports its parity check.
  - Static readout errors e0 = P("1" | 0) and e1 = P("0" | 1).
  - Readout memory: after the ancilla reported "1" in the previous round, a 0
    reads "1" with extra probability m_pos (positive memory) and a 1 reads "0"
    with extra probability m_neg (negative memory). The model assumes the
    memory survives the ancilla reset; the runs measured on reset-free chains.
  - Leakage: an ancilla leaks with probability g per round, reports "1" with
    probability q2 while leaked, and returns with probability g_dn per round
    (mean episode 1 / g_dn rounds).
  - The decoder (PyMatching) weights measurement errors as independent at the
    average misread rate observed in the same simulation. Each case is
    compared with a rate-matched control: independent readout errors with e0
    and e1 scaled so the average misread rate equals the case's, at the same
    distance. The ratio to that control isolates the correlation; the ratio
    to the static reference also contains the change in error rate.

Parameter sources (the memory and leakage scale; the code layout is generic):
  - Eigenstate memory below 3e-4 on every qubit in E0 and E0R
    (instrument_reading.json from analysis/cloud_run_eig_instrument.py), so
    m = 0.002 and 0.004 are upper scenarios, 7 to 13 times that bound. (They
    were first set from the withdrawn in-chain readings.)
  - Static errors e0 = 0.003, e1 = 0.008, inside the E0 and E0R AS ranges.
  - Leakage per readout 5.3e-5 to 1.8e-3 and lifetimes of tens of microseconds
    (IBM note, Sections 5 and 6); here g = 5e-4 per round and 1 / g_dn = 10
    rounds, a round of a few microseconds.

Requires NumPy and PyMatching (in the Qiskit virtualenv). Exits nonzero if a
self-check fails.

  ~/.venvs/qiskit/bin/python analysis/qec_memory_impact.py --output analysis/qec_memory_impact.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

SHOTS = {3: 1_000_000, 5: 2_000_000, 7: 6_000_000}  # more shots where logical errors are rare
DISTANCES = (3, 5, 7)
P_DATA = 0.01
E0, E1 = 0.003, 0.008
CASES = {
    "independent (reference)": {},
    "negative memory 0.2%": {"m_neg": 0.002},
    "negative memory 0.4%": {"m_neg": 0.004},
    "positive memory 0.4%": {"m_pos": 0.004},
    "leakage episodes": {"g": 5e-4, "g_dn": 0.1, "q2": 0.99},
    "negative memory 0.4% + leakage": {"m_neg": 0.004, "g": 5e-4, "g_dn": 0.1, "q2": 0.99},
    # Leakage is the assumption-heavy part: sweep its rate and episode length.
    "leakage, g 1e-4": {"g": 1e-4, "g_dn": 0.1, "q2": 0.99},
    "leakage, g 2e-3": {"g": 2e-3, "g_dn": 0.1, "q2": 0.99},
    "leakage, 3-round episodes": {"g": 5e-4, "g_dn": 1 / 3, "q2": 0.99},
    "leakage, 30-round episodes": {"g": 5e-4, "g_dn": 1 / 30, "q2": 0.99},
}


def simulate(d: int, rounds: int, shots: int, rng, p: float, e0: float, e1: float,
             m_pos: float = 0.0, m_neg: float = 0.0, g: float = 0.0, g_dn: float = 1.0, q2: float = 1.0):
    """Reported syndromes (shots, d - 1, rounds + 1), the final data errors, and the misread rate."""
    data = np.zeros((shots, d), bool)
    leaked = np.zeros((shots, d - 1), bool)
    prev1 = np.zeros((shots, d - 1), bool)
    synd = np.zeros((shots, d - 1, rounds + 1), np.uint8)
    wrong = 0
    for t in range(rounds):
        data ^= rng.random(data.shape) < p
        parity = data[:, :-1] ^ data[:, 1:]
        if g > 0.0:
            leaked &= ~(rng.random(leaked.shape) < g_dn)
            leaked |= rng.random(leaked.shape) < g
        err = np.where(parity, e1 + m_neg * prev1, e0 + m_pos * prev1)
        report = parity ^ (rng.random(parity.shape) < err)
        if g > 0.0:
            report = np.where(leaked, rng.random(parity.shape) < q2, report)
        wrong += int((report != parity).sum())
        prev1 = report
        synd[:, :, t] = report
    synd[:, :, rounds] = data[:, :-1] ^ data[:, 1:]
    return synd, data, wrong / (shots * (d - 1) * rounds)


def logical_error_rate(d: int, synd: np.ndarray, data: np.ndarray, p: float, q: float) -> tuple:
    """Decode with matching that assumes independent errors; return (rate, standard error)."""
    import pymatching

    H = np.zeros((d - 1, d), np.uint8)
    for i in range(d - 1):
        H[i, i] = H[i, i + 1] = 1
    reps = synd.shape[2]
    matching = pymatching.Matching.from_check_matrix(
        H, weights=math.log((1 - p) / p), repetitions=reps,
        timelike_weights=math.log((1 - q) / q) if q > 0 else 1e6)
    det = synd.copy()
    det[:, :, 1:] ^= synd[:, :, :-1]
    # PyMatching numbers detectors round by round: index = round * (d - 1) + check.
    corr = matching.decode_batch(det.transpose(0, 2, 1).reshape(det.shape[0], -1))
    residual = (data ^ corr.astype(bool))
    fail = residual[:, 0]                      # in the code space the residual is all 0 or all 1
    assert np.all(residual == residual[:, :1]), "decoder left a residual outside the code space"
    rate = float(fail.mean())
    return rate, math.sqrt(max(rate * (1 - rate), 1e-12) / fail.size)


def run(output: str | None) -> int:
    rng = np.random.default_rng(20261121)
    checks = 0

    def ok(label, cond):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(label)

    # Self-checks.
    s, dat, _ = simulate(5, 5, 2000, rng, 0.0, 0.0, 0.0)
    ok("no noise, no logical errors", logical_error_rate(5, s, dat, 1e-3, 1e-3)[0] == 0.0)
    s, dat, w = simulate(3, 3, 50_000, rng, 0.0, E0, E1)
    ok("misread rate with balanced parities near the static mean", abs(w - E0) < 1e-3)
    results = {}
    for name, kw in CASES.items():
        rows = {}
        for d in DISTANCES:
            synd, data, q = simulate(d, d, SHOTS[d], rng, P_DATA, E0, E1, **kw)
            rate, se = logical_error_rate(d, synd, data, P_DATA, q)
            rows[str(d)] = {"logical_error": rate, "se": se, "misread_rate": round(q, 6)}
        results[name] = rows
    ref = results["independent (reference)"]
    matched = {}
    for name in CASES:
        if name == "independent (reference)":
            continue
        matched[name] = {}
        for d in DISTANCES:
            k = results[name][str(d)]["misread_rate"] / ref[str(d)]["misread_rate"]
            synd, data, q = simulate(d, d, SHOTS[d], rng, P_DATA, E0 * k, E1 * k)
            rate, se = logical_error_rate(d, synd, data, P_DATA, q)
            matched[name][str(d)] = {"logical_error": rate, "se": se, "misread_rate": round(q, 6), "scale": round(k, 4)}
    ok("independent reference: logical error falls with distance",
       all(ref[str(a)]["logical_error"] > ref[str(b)]["logical_error"] for a, b in zip(DISTANCES, DISTANCES[1:])))
    ratios = {}
    for name, rows in results.items():
        if name == "independent (reference)":
            continue
        ratios[name] = {}
        for d in DISTANCES:
            r, rs = rows[str(d)]["logical_error"], rows[str(d)]["se"]
            f, fs = ref[str(d)]["logical_error"], ref[str(d)]["se"]
            ratio = r / f if f > 0 else float("nan")
            ratio_se = ratio * math.hypot(rs / r if r else 0, fs / f if f else 0) if f > 0 and r > 0 else float("nan")
            ratios[name][str(d)] = [round(ratio, 3), round(ratio_se, 3)]
    ratios_matched = {}
    for name, rows in matched.items():
        ratios_matched[name] = {}
        for d in DISTANCES:
            r, rs = results[name][str(d)]["logical_error"], results[name][str(d)]["se"]
            f, fs = rows[str(d)]["logical_error"], rows[str(d)]["se"]
            ratio = r / f if f > 0 else float("nan")
            ratio_se = ratio * math.hypot(rs / r if r else 0, fs / f if f else 0) if f > 0 and r > 0 else float("nan")
            ratios_matched[name][str(d)] = [round(ratio, 3), round(ratio_se, 3)]
    out = {"status": "PASS", "checks": checks,
           "note": "Simulation of a bit-flip repetition code with the measured memory and leakage scales; "
                   "nothing here is experimental data. See the module docstring for the model and its assumptions.",
           "settings": {"shots": {str(k): v for k, v in SHOTS.items()}, "p_data": P_DATA, "e0": E0, "e1": E1, "rounds": "d", "distances": DISTANCES},
           "results": results, "ratio_to_independent": ratios,
           "matched_controls": matched, "ratio_to_matched": ratios_matched}
    print(f"PASS: {checks} checks")
    for name, rows in results.items():
        print(f"{name:34s}", " ".join(f"d={d}: {rows[str(d)]['logical_error']:.2e}" for d in DISTANCES),
              "| ratio", ratios.get(name, {}), "| matched", ratios_matched.get(name, {}))
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output")
    args = ap.parse_args()
    try:
        return run(args.output)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
