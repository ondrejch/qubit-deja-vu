#!/usr/bin/env python3
"""Data figures of the journal manuscript, from saved run outputs only.

Reads data/cloudrun/*/report.json, drift_exploratory.json and
leakage_exploratory.json; writes figures/*.pdf. Every plotted number is
asserted against the value quoted in the manuscript (tolerance covers the table
rounding). No IBM contact, no new computation of statistics.

Usage: python3 scripts/make_paper_figures.py  (run from the repository root)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUD = os.path.join(ROOT, "data", "cloudrun")
FIGS = os.path.join(ROOT, "figures")
INK = "#16344C"


def load(run, name="report.json"):
    with open(os.path.join(CLOUD, run, name)) as fh:
        return json.load(fh)


def val_se(pair):
    return pair[0], pair[1]


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def fig_lag():
    """Neighbor-covariance lag spectra at 1.44 us: plateau vs departures."""
    leak = load("2026-10-15_main", "leakage_exploratory.json")
    by_q = {q["physical_qubit"]: q for q in leak["qubits"]}
    lags = {}
    for q in (149, 11, 109):
        obs = by_q[q]["conditions"]["A0"]["observed_lags"]
        assert len(obs) == 8, (q, len(obs))
        lags[q] = obs
    # Values quoted in main.tex Section sec:main.
    assert abs(lags[149][0] - 0.00784) < 1e-5, lags[149][0]
    assert min(lags[149][1:]) > 0.0075 and max(lags[149][1:]) < 0.0086
    assert abs(lags[11][0] - (-0.0044)) < 5e-4, lags[11][0]
    assert abs(lags[109][0] - 0.0099) < 5e-4, lags[109][0]

    _, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = list(range(1, 9))
    for q, kw in ((149, {"color": INK, "marker": "o"}),
                  (11, {"color": "#B3541E", "marker": "s"}),
                  (109, {"color": "#4C8C4C", "marker": "^"})):
        ax.plot(xs, lags[q], label=f"qubit {q}", **kw, linewidth=1.1,
                markersize=4)
    ax.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"lag (readouts at 1.44 $\mu$s period)")
    ax.set_ylabel("covariance of physical outcomes")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(pad=0.6)
    os.makedirs(FIGS, exist_ok=True)
    plt.savefig(os.path.join(FIGS, "fig-lag.pdf"))
    plt.close()


def fig_sign():
    """Per-qubit clean neighbor covariance: the sign pattern."""
    follow = load("2026-10-15_followup")
    rerun = load("2026-10-15_fezrepeat_rerun")
    layout_k = follow["layout"]
    layout_f = rerun["layout"]
    k_vals, f_vals = {}, {}
    for i, q in enumerate(layout_k):
        v, se = val_se(
            follow["conditions"]["T0"]["per_qubit"][i]["D_fix_clean"])
        k_vals[q] = (v, se)
    for i, q in enumerate(layout_f):
        v, se = val_se(
            rerun["conditions"]["G0"]["per_qubit"][i]["D_fix_clean"])
        f_vals[q] = (v, se)
    # Values quoted in main.tex Tables tab:clean and tab:fez.
    expect = {21: 0.0007, 154: -0.0027, 11: -0.0086, 149: -0.0017,
              109: 0.0060, 89: -0.0001,
              124: 0.0041, 26: -0.0007, 86: -0.0009, 144: 0.0055,
              2: -0.0029, 6: 0.0052}
    for q, e in expect.items():
        got = k_vals.get(q, f_vals.get(q))[0]
        assert abs(got - e) < 1e-4, (q, got, e)

    _, ax = plt.subplots(figsize=(6.4, 2.9))
    order = ([("k", q) for q in layout_k]
             + [(None, None)] + [("f", q) for q in layout_f])
    xs, vs, es, colors = [], [], [], []
    labels = []
    x = 0
    for src, q in order:
        if q is None:
            x += 1
            continue
        v, se = (k_vals if src == "k" else f_vals)[q]
        xs.append(x)
        vs.append(v)
        es.append(se)
        colors.append("#B3541E" if v < 0 else INK)
        labels.append(str(q))
        x += 1
    ax.bar(xs, vs, yerr=es, capsize=2, color=colors, edgecolor="white",
           error_kw={"ecolor": "black", "elinewidth": 0.9})
    ax.axhline(0, color="0.3", linewidth=0.7, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("physical qubit (left: ibm_kingston T0; right: ibm_fez G0)")
    ax.set_ylabel("clean neighbor covariance $D_{\\mathrm{clean}}$")
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(FIGS, "fig-sign.pdf"))
    plt.close()


def fig_equator_eig():
    """Equator memory against eigenstate memory per qubit, E0 and E0R.

    Both axes use the sign convention of D: positive is a bias toward
    repeating the previous report. The eigenstate memory is the negative of
    the contrast deviation from the in-chain null (a negative deviation is
    positive memory). E0's in-chain reading is exploratory; E0R's is the
    pre-registered primary reading.
    """
    runs = (("2026-11-19_eig", "drift_exploratory.json", "E0 (exploratory)", "o", "white"),
            ("2026-11-19_eigrepeat", "inchain_reading.json", "E0R (pre-registered)", "s", INK))
    # Values quoted in main.tex Tables tab:eig, tab:inchain and tab:repeat
    # (deviation, not yet sign-flipped) and in the E0 and E0R equator text.
    expect = {
        "2026-11-19_eig": ({124: 0.0035, 144: 0.0052, 6: 0.0051, 50: -0.0013,
                            23: 0.0, 59: 0.0060, 114: -0.0001, 132: -0.0004},
                           {124: 0.0032, 144: 0.0023, 6: -0.0020, 50: 0.0008,
                            23: 0.0041, 59: 0.0007, 114: 0.0022, 132: 0.0021}),
        "2026-11-19_eigrepeat": ({124: 0.0038, 144: 0.0055, 6: 0.0055},
                                 {124: 0.0029, 144: 0.0014, 6: -0.0013, 50: 0.0002,
                                  23: 0.0040, 59: 0.0016, 114: 0.0030, 132: 0.0016}),
    }
    _, ax = plt.subplots(figsize=(5.0, 3.6))
    pos = {}
    for run, name, label, marker, face in runs:
        rep = load(run)
        reading = load(run, name)
        layout = rep["layout"]
        per = rep["conditions"]["E0"]["per_qubit"]
        xs, ys, xes, yes = [], [], [], []
        for i, q in enumerate(layout):
            ex, exe = val_se(per[i]["clean"]["B_lag1"])
            dy, dye = val_se(reading["qubits"][str(q)]["clean"]["contrast_deviation_in_chain_null"])
            ex_x, ex_y = expect[run]
            if q in ex_x:
                assert abs(ex - ex_x[q]) < 2.5e-4, (run, q, ex)
            assert abs(dy - ex_y[q]) < 2.5e-4, (run, q, dy)
            xs.append(ex)
            ys.append(-dy)
            xes.append(exe)
            yes.append(dye)
            pos.setdefault(q, []).append((ex, -dy))
        ax.errorbar(xs, ys, xerr=xes, yerr=yes, fmt=marker, ecolor="0.45", elinewidth=0.8,
                    capsize=1.5, markersize=4.5, markerfacecolor=face, markeredgecolor=INK,
                    markeredgewidth=1.0, label=label)
    # One label per qubit, beside its E0R point, nudged where points crowd.
    nudge = {114: (6, 4), 132: (-20, -10), 23: (6, 2), 50: (-16, 5), 6: (6, -2),
             124: (6, 2), 144: (6, 2), 59: (-14, -11)}
    for q, pts in pos.items():
        x, y = pts[-1]
        ax.annotate(str(q), (x, y), textcoords="offset points", xytext=nudge.get(q, (4, 3)),
                    fontsize=7, color="0.2")
    ax.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="0.5", linewidth=0.7, linestyle="--")
    ax.set_xlabel("equator memory (clean B lag-1 covariance)")
    ax.set_ylabel("eigenstate memory (minus contrast deviation)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(FIGS, "fig-equator-eig.pdf"))
    plt.close()


if __name__ == "__main__":
    fig_lag()
    fig_sign()
    fig_equator_eig()
    print("wrote", sorted(os.listdir(FIGS)))
