#!/usr/bin/env python3
"""Data figures of the journal manuscript, from saved run outputs only.

Reads data/cloudrun/*/report.json, instrument_reading.json and
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
    """Eigenstate repeat bias m0 - m1 against equator covariance D_clean, E0 and E0R.

    Both from the same shots: D_clean from report.json (clean B lag-1), m0 and m1
    from instrument_reading.json (prev_next mask). A readout that tends to
    repeat its previous report gives m0 - m1 = 2D (main.tex Section 2.3, item 1).
    """
    runs = (("2026-11-19_eig", "E0", "o", "white"), ("2026-11-19_eigrepeat", "E0R", "s", INK))
    # Values quoted in main.tex Table tab:eig (D_clean; m1 and m0 in units of 1e-4).
    expect = {
        "2026-11-19_eig": {124: (0.0035, 0.2, 0.1), 144: (0.0052, -1.0, -0.5), 6: (0.0051, 0.5, 0.0),
                           50: (-0.0013, 0.3, -0.3), 23: (0.0, -2.0, 1.7), 59: (0.0060, -0.3, -0.4),
                           114: (-0.0001, -1.0, 0.0), 132: (-0.0003, 0.6, -0.4)},
        "2026-11-19_eigrepeat": {124: (0.0038, -0.1, 0.0), 144: (0.0055, 1.1, 0.0), 6: (0.0055, 0.8, 0.5),
                                 50: (-0.0012, 0.4, -0.1), 23: (-0.0003, -2.8, 0.4), 59: (0.0058, -2.0, -1.1),
                                 114: (0.0001, -1.2, -0.8), 132: (0.0, 0.1, -0.6)},
    }
    _, ax = plt.subplots(figsize=(5.0, 3.6))
    pos = {}
    for run, label, marker, face in runs:
        rep = load(run)
        fit = load(run, "instrument_reading.json")
        per = rep["conditions"]["E0"]["per_qubit"]
        xs, ys, xes, yes = [], [], [], []
        for i, q in enumerate(rep["layout"]):
            ex, exe = val_se(per[i]["clean"]["B_lag1"])
            pn = fit["qubits"][str(q)]["prev_next"]
            m0, m0e = pn["m0"]["value"], pn["m0"]["se"]
            m1, m1e = pn["m1"]["value"], pn["m1"]["se"]
            wd, w1, w0 = expect[run][q]
            assert abs(ex - wd) < 6e-5, (run, q, ex)
            assert abs(m1 * 1e4 - w1) < 0.06 and abs(m0 * 1e4 - w0) < 0.06, (run, q, m1, m0)
            xs.append(ex)
            ys.append(m0 - m1)
            xes.append(exe)
            yes.append((m0e ** 2 + m1e ** 2) ** 0.5)
            pos.setdefault(q, []).append((ex, m0 - m1))
        ax.errorbar(xs, ys, xerr=xes, yerr=yes, fmt=marker, ecolor="0.45", elinewidth=0.8,
                    capsize=1.5, markersize=4.5, markerfacecolor=face, markeredgecolor=INK,
                    markeredgewidth=1.0, label=label)
    lim = (-0.002, 0.0068)
    ax.plot(lim, [2 * v for v in lim], color="0.35", linewidth=0.8, linestyle=":")
    nudge = {124: (-4, 6), 144: (-6, 6), 6: (-14, -11), 50: (4, 5), 23: (-6, 7), 59: (4, -10),
             114: (-2, -11), 132: (-14, 6)}
    for q, pts in pos.items():
        x, y = pts[-1]
        ax.annotate(str(q), (x, y), textcoords="offset points", xytext=nudge.get(q, (4, 3)),
                    fontsize=7, color="0.2")
    ax.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="0.5", linewidth=0.7, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("equator covariance $D_{\\mathrm{clean}}$")
    ax.set_ylabel("eigenstate repeat bias $m_0 - m_1$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(FIGS, "fig-equator-eig.pdf"))
    plt.close()


if __name__ == "__main__":
    fig_lag()
    fig_sign()
    fig_equator_eig()
    print("wrote", sorted(os.listdir(FIGS)))
