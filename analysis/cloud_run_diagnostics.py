#!/usr/bin/env python3
"""Exploratory diagnostics for the cloud-run records (not pre-registered).

These statistics were chosen after seeing the 15 October 2026 data. They
back the descriptive numbers in notes/CLOUD_RUN_RESULTS.md (Sections 4 and 6)
and the IBM analysis note that do not come from report.json:

1. Long runs (balanced chains): the fraction of shots holding at least ten
   consecutive "1" (or "0") readings, against the same shots with their
   records shuffled within the shot (same marginals, no time order).
2. Flip failures (probe chain): probe steps with |phi| < 0.15 flip the
   previous outcome. The pattern frequencies of (previous, flip-step)
   readouts, and the rate of a "1, 1" failure after one "1" and after five
   consecutive "1".
3. Clustered standard errors: the neighbor covariance of balanced chains and
   the clean neighbor covariance of leak-tagged chains, with the standard
   error recomputed treating each shot, and each parameter set, as one
   independent unit. The ratio to the reported pair-counting error shows
   whether the five-sigma calls assume more independence than the records
   have.
4. Leak buildup (leak-tagged chains at two spacings): the pre-registered S4
   rule compares leaked-check fractions across spacings assuming a steady
   leaked population. Each shot starts unleaked, and a 66-record shot at
   1.44 us lasts about 95 us, comparable to the leak lifetimes. A one-level
   model (leak-in g per record, or g per unit time, decay with lifetime tau,
   checked at the middle flip of each block) gives the ratio of leaked
   fractions expected under readout-only and time-only leak-in for a grid of
   lifetimes, beside the observed ratio per qubit.

  python analysis/cloud_run_diagnostics.py data/cloudrun/2026-10-15_main --output data/cloudrun/2026-10-15_main/diagnostics_exploratory.json

Needs NumPy.
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
import cloud_run_analyze as cra  # noqa: E402
import cloud_run_followup as crf  # noqa: E402

RUN_LENGTH = 10
FLIP_WINDOW = 0.15
BUILDUP_TAUS_US = (20, 40, 60, 80, 100, 120)


def has_run(x: np.ndarray, val: int, length: int) -> np.ndarray:
    """x (N, K) -> True where a row holds at least `length` consecutive `val`."""
    m = (x == val).astype(np.int16)
    c = np.zeros(m.shape[0], np.int16)
    best = np.zeros(m.shape[0], np.int16)
    for k in range(m.shape[1]):
        c = (c + 1) * m[:, k]
        best = np.maximum(best, c)
    return best >= length


def long_runs(r: np.ndarray, rng) -> list:
    """r (sets, shots, qubits, K) of a balanced chain; "1" is r = -1."""
    sets, shots, nq, K = r.shape
    out = []
    for q in range(nq):
        x = r[:, :, q, :].reshape(-1, K)
        xs = np.take_along_axis(x, np.argsort(rng.random(x.shape), axis=1), axis=1)
        out.append({
            "shots_with_run_of_1": round(float(has_run(x, -1, RUN_LENGTH).mean()), 5),
            "shuffled_run_of_1": round(float(has_run(xs, -1, RUN_LENGTH).mean()), 5),
            "shots_with_run_of_0": round(float(has_run(x, 1, RUN_LENGTH).mean()), 5),
            "shuffled_run_of_0": round(float(has_run(xs, 1, RUN_LENGTH).mean()), 5),
        })
    return out


def flip_failures(r: np.ndarray, phi: np.ndarray) -> list:
    """r (sets, shots, qubits, K) and phi (sets, qubits, K) of the probe chain."""
    sets, shots, nq, K = r.shape
    out = []
    for q in range(nq):
        rows = {"11": 0, "00": 0, "10": 0, "01": 0}
        after5 = [0, 0]
        after1 = [0, 0]
        for k in range(1, K, 2):
            sel = np.abs(((phi[:, q, k] + math.pi) % (2 * math.pi)) - math.pi) < FLIP_WINDOW
            if not sel.any():
                continue
            a = r[sel, :, q, k - 1]
            b = r[sel, :, q, k]
            rows["11"] += int(((a == -1) & (b == -1)).sum())
            rows["00"] += int(((a == 1) & (b == 1)).sum())
            rows["10"] += int(((a == -1) & (b == 1)).sum())
            rows["01"] += int(((a == 1) & (b == -1)).sum())
            if k >= 5:
                hist = np.all(r[sel, :, q, k - 5:k] == -1, axis=-1)
                fail = (a == -1) & (b == -1)
                after5[0] += int((fail & hist).sum())
                after5[1] += int(hist.sum())
                prev1 = a == -1
                after1[0] += int((fail & prev1).sum())
                after1[1] += int(prev1.sum())
        n = sum(rows.values())
        out.append({
            "flip_pairs": n,
            "p_11": round(rows["11"] / n, 5),
            "p_00": round(rows["00"] / n, 5),
            "ratio_11_over_00": round(rows["11"] / max(rows["00"], 1), 3),
            "p_fail_11_given_previous_1": round(after1[0] / max(after1[1], 1), 5),
            "p_fail_11_given_previous_five_1": round(after5[0] / max(after5[1], 1), 4),
            "cases_previous_five_1": after5[1],
        })
    return out


def clustered(x: np.ndarray, y: np.ndarray, m: np.ndarray, shots: int) -> dict:
    """Covariance of x, y over mask m, all (rows, cols) with rows = sets x shots.

    Returns the estimate, the pair-counting standard error (as in the
    reports) and the errors with each shot and each parameter set taken as one
    independent unit.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    n = m.sum()
    mx = (x * m).sum() / n
    my = (y * m).sum() / n
    prod = (x - mx) * (y - my)
    mean = (prod * m).sum() / n
    dev = (prod - mean) * m
    naive = math.sqrt((dev ** 2).sum() / n) / math.sqrt(n)
    per_shot = dev.sum(axis=1)
    per_set = per_shot.reshape(-1, shots).sum(axis=1)
    se_shot = math.sqrt((per_shot ** 2).sum()) / n
    se_set = math.sqrt((per_set ** 2).sum()) / n
    return {"cov": round(float(mean), 6), "se_pairs": round(naive, 6),
            "se_shot_clustered": round(se_shot, 6), "se_set_clustered": round(se_set, 6),
            "inflation_shot": round(se_shot / naive, 3), "inflation_set": round(se_set / naive, 3)}


def balanced_clustered(r: np.ndarray) -> list:
    sets, shots, nq, K = r.shape
    out = []
    for q in range(nq):
        x = r[:, :, q, :].reshape(-1, K)
        out.append(clustered(x[:, :-1], x[:, 1:], np.ones((x.shape[0], K - 1), bool), shots))
    return out


def tagged_clustered(r: np.ndarray) -> list:
    """Clean within-block lag-1 covariance, as in cloud_run_followup.tagged_statistics."""
    sets, shots, nq, K = r.shape
    nb = K // crf.BLOCK
    out = []
    for q in range(nq):
        x = r[:, :, q, :].reshape(-1, K)
        clean = crf.clean_blocks(crf.check_states(x))
        blocks = x.reshape(x.shape[0], nb, crf.BLOCK)
        xs = np.concatenate([blocks[:, :, j] for j in range(3)], axis=1)
        ys = np.concatenate([blocks[:, :, j + 1] for j in range(3)], axis=1)
        ms = np.concatenate([clean] * 3, axis=1)
        out.append(clustered(xs, ys, ms, shots))
    return out


def _period_s(build: dict, label: str) -> float:
    return build["readout_s"] + 2 * build["sx_s"] + build[label].get("delay_dt", 0) * build["dt_s"]


def mean_leaked_at_checks(K: int, period_s: float, tau_s: float, g: float) -> float:
    """Leaked population at the middle flip of each block, shot starting unleaked."""
    p, vals = 0.0, []
    decay = math.exp(-period_s / tau_s)
    for k in range(K):
        p *= decay
        p += (1.0 - p) * g
        if k % crf.BLOCK == 4:
            vals.append(p)
    return float(np.mean(vals))


def leak_buildup(manifest: dict, chains: dict) -> dict:
    build = manifest["build_report"]
    tagged = [lab for lab, meta in manifest["chains"].items() if meta["kind"] == "tagged"]
    if len(tagged) < 2:
        return {}
    ref = min(tagged, key=lambda lab: _period_s(build, lab))
    K = manifest["chains"][ref]["records"]
    others = [lab for lab in tagged if lab != ref and manifest["chains"][lab]["records"] == K]
    if not others:
        return {}
    g = 1e-4
    t0 = _period_s(build, ref)

    def leaked(lab):
        r = chains[lab]["r"]
        return np.array([float((crf.check_states(r[:, :, q, :].reshape(-1, K)) == 1).mean())
                         for q in range(r.shape[2])])

    f_ref = leaked(ref)
    out = {"reference": ref, "reference_period_us": round(t0 * 1e6, 3), "tau_us": list(BUILDUP_TAUS_US), "conditions": {}}
    for lab in others:
        t1 = _period_s(build, lab)
        ro, ti = [], []
        for tau in BUILDUP_TAUS_US:
            base = mean_leaked_at_checks(K, t0, tau * 1e-6, g)
            ro.append(round(mean_leaked_at_checks(K, t1, tau * 1e-6, g) / base, 3))
            ti.append(round(mean_leaked_at_checks(K, t1, tau * 1e-6, g * t1 / t0) / base, 3))
        out["conditions"][lab] = {"period_us": round(t1 * 1e6, 3), "rate_factor": round(t1 / t0, 3),
                                  "expected_ratio_readout_only": ro, "expected_ratio_time_only": ti,
                                  "observed_ratio": [round(float(v), 3) for v in leaked(lab) / f_ref]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest, chains, _ = cra.load(args.run)
    rng = np.random.default_rng(0)
    report = {"run": args.run, "status": "exploratory, chosen after seeing the data",
              "layout": manifest["build_report"]["layout"], "conditions": {}}
    for label, meta in manifest["chains"].items():
        kind = meta["kind"]
        r = chains[label]["r"]
        entry = {"kind": kind}
        if kind == "balanced":
            entry["lag1_clustered"] = balanced_clustered(r)
            if label == "A0":
                entry[f"runs_of_{RUN_LENGTH}"] = long_runs(r, rng)
        elif kind == "probe":
            entry["flip_failures"] = flip_failures(r, chains[label]["phi"])
        elif kind == "tagged":
            entry["clean_lag1_clustered"] = tagged_clustered(r)
        else:
            continue
        report["conditions"][label] = entry
    buildup = leak_buildup(manifest, chains)
    if buildup:
        report["leak_buildup"] = buildup
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
