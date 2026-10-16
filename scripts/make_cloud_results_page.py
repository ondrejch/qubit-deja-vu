#!/usr/bin/env python3
"""Generate notes/cloud_run_results.html, the shareable summary of the 15 October 2026
ibm_kingston runs, from the run reports in data/cloudrun/ (numbers are read, not retyped).

  python scripts/make_cloud_results_page.py
"""
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "data", "cloudrun") + os.sep
main = json.load(open(ROOT + "2026-10-15_main/report.json"))
fol = json.load(open(ROOT + "2026-10-15_followup/report.json"))
jobs = {name: json.load(open(ROOT + f"2026-10-15_{name}/job.json")) for name in ("pilot", "main", "followup")}
layout = fol["layout"]


def fmt(x, digits=1):
    """1.23e-3 style with a real minus and superscript exponent, for HTML."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    sign = "−" if m < 0 else ""
    exp = str(e).replace("-", "\u2212")
    return f"{sign}{abs(m):.{digits}f}\u00d710<sup>{exp}</sup>"


def num(x, d=4):
    s = f"{x:+.{d}f}"
    return s.replace("-", "−")


qlim = {g: main["conditions"][l]["pooled"]["Q_fix_upper_5sigma"] for g, l in (("1.44", "A0"), ("5.44", "A1"), ("41.4", "A2"))}
b0 = main["conditions"]["B0"]["pooled"]
wq2 = b0["order2"]["w_Q"][0] + 5 * b0["order2"]["w_Q"][1]
wd4 = b0["order4"]["w_D"][0] + 5 * b0["order4"]["w_D"][1]
t0 = fol["conditions"]["T0"]["per_qubit"]
s2 = [(pq["D_fix_clean"][0], pq["D_fix_clean"][1]) for pq in t0]
lim21 = s2[0][0] + 5 * s2[0][1]
lim89 = s2[5][0] + 5 * s2[5][1]
cross_all = [pq["all"]["cross_block"] for pq in t0]
cross_clean = [pq["clean"]["cross_block"] for pq in t0]
charged_main = sum(j["metrics"]["usage"]["qpu_charge_time_seconds"] for j in jobs["main"]["jobs"])
charged_fol = sum(j["metrics"]["usage"]["qpu_charge_time_seconds"] for j in jobs["followup"]["jobs"])


# ------------------------------------------------------------------ charts
def bar_chart_memory():
    W, H, L, R, T, B = 640, 300, 64, 16, 16, 44
    lo, hi = -0.011, 0.009
    def y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)
    n = len(layout)
    slot = (W - L - R) / n
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="fig1t">',
           '<title id="fig1t">Clean neighbor covariance per qubit at a 1.44 microsecond spacing</title>']
    for t in (-0.010, -0.005, 0.0, 0.005):
        yy = y(t)
        out.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        out.append(f'<text class="tick" x="{L - 8}" y="{yy + 4:.1f}" text-anchor="end">{num(t, 3)}</text>')
    for i, (v, se) in enumerate(s2):
        cx = L + slot * (i + 0.5)
        flagged = abs(v) > 5 * se
        top, bot = y(max(v, 0)), y(min(v, 0))
        out.append(f'<rect class="{"bar-flag" if flagged else "bar-ok"}" x="{cx - slot * 0.28:.1f}" y="{top:.1f}" '
                   f'width="{slot * 0.56:.1f}" height="{max(bot - top, 1):.1f}"/>')
        y1, y2 = y(v + 5 * se), y(v - 5 * se)
        out.append(f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>')
        for yy in (y1, y2):
            out.append(f'<line class="whisker" x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        label_y = y(v + 5 * se) - 6 if v >= 0 else y(v - 5 * se) + 14
        out.append(f'<text class="val" x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle">{num(v, 4)}</text>')
        out.append(f'<text class="cat" x="{cx:.1f}" y="{H - B + 20:.1f}" text-anchor="middle">{layout[i]}</text>')
    out.append(f'<text class="axis-title" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">physical qubit</text>')
    out.append("</svg>")
    return "\n".join(out)


def bar_chart_cross():
    W, H, L, R, T, B = 640, 280, 64, 16, 16, 44
    lo, hi = -0.001, 0.007
    def y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)
    n = len(layout)
    slot = (W - L - R) / n
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="fig2t">',
           '<title id="fig2t">Shot-wide covariance before and after removing leak-tagged blocks</title>']
    for t in (0.0, 0.002, 0.004, 0.006):
        yy = y(t)
        out.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        out.append(f'<text class="tick" x="{L - 8}" y="{yy + 4:.1f}" text-anchor="end">{num(t, 3) if t else "0"}</text>')
    for i in range(n):
        cx = L + slot * (i + 0.5)
        for j, (series, cls) in enumerate(((cross_all, "bar-raw"), (cross_clean, "bar-ok"))):
            v, se = series[i]
            x0 = cx - slot * 0.32 + j * slot * 0.32
            top, bot = y(max(v, 0)), y(min(v, 0))
            out.append(f'<rect class="{cls}" x="{x0:.1f}" y="{top:.1f}" width="{slot * 0.3:.1f}" height="{max(bot - top, 1):.1f}"/>')
            mid = x0 + slot * 0.15
            out.append(f'<line class="whisker" x1="{mid:.1f}" x2="{mid:.1f}" y1="{y(v + se):.1f}" y2="{y(v - se):.1f}"/>')
        out.append(f'<text class="cat" x="{cx:.1f}" y="{H - B + 20:.1f}" text-anchor="middle">{layout[i]}</text>')
    out.append(f'<text class="axis-title" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">physical qubit</text>')
    out.append("</svg>")
    return "\n".join(out)


rows_s2 = "\n".join(
    f"<tr><td>{layout[i]}</td><td>{num(v, 4)} ± {se:.4f}</td><td>"
    + ("readout memory" if abs(v) > 5 * se else f"w<sub>D</sub> &lt; {fmt(v + 5 * se)}")
    + "</td></tr>"
    for i, (v, se) in enumerate(s2))

html = f"""<title>ibm_kingston Persistence Tests</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; three result cards, two to-scale figures, compact tables. */
:root {{
  --bg: #f4f6f7; --surface: #ffffff; --ink: #18232c; --muted: #55636e; --rule: #d6dde2;
  --accent: #0b6772; --flag: #a65d00; --raw: #9aa8b2;
  --display: "Newsreader", Georgia, "Times New Roman", serif;
  --body: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #11171c; --surface: #172028; --ink: #e3e9ed; --muted: #9aa9b4; --rule: #2c3944;
  --accent: #4cc0c9; --flag: #e7a14a; --raw: #5d6d79; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #11171c; --surface: #172028; --ink: #e3e9ed; --muted: #9aa9b4; --rule: #2c3944;
  --accent: #4cc0c9; --flag: #e7a14a; --raw: #5d6d79; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.6 var(--body); padding: 40px 16px 64px; }}
main {{ max-width: 46rem; margin: 0 auto; display: grid; gap: 2.2rem; }}
header {{ display: grid; gap: .6rem; }}
.eyebrow {{ font: 500 .75rem/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 clamp(2rem, 5vw, 2.8rem)/1.1 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 1.45rem/1.25 var(--display); margin: 0 0 .5rem; text-wrap: balance; }}
p {{ margin: 0; max-width: 65ch; }}
.lede {{ color: var(--muted); font-size: 1.05rem; }}
section {{ display: grid; gap: .8rem; min-width: 0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 1rem; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1rem 1.1rem; display: grid; gap: .4rem; align-content: start; min-width: 0; }}
.card .k {{ font: 500 .72rem/1.2 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
.card .v {{ font: 600 1.35rem/1.25 var(--display); }}
.card p {{ font-size: .92rem; color: var(--muted); }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1rem; display: grid; gap: .6rem; min-width: 0; }}
figure .scroll {{ overflow-x: auto; }}
svg {{ width: 100%; min-width: 30rem; height: auto; display: block; }}
figcaption {{ font-size: .88rem; color: var(--muted); max-width: 65ch; }}
.grid {{ stroke: var(--rule); stroke-width: 1; }}
.axis {{ stroke: var(--muted); stroke-width: 1.2; }}
.tick, .cat, .axis-title, .val {{ font-family: var(--mono); font-size: 11px; fill: var(--muted); }}
.cat {{ fill: var(--ink); font-size: 12px; }}
.val {{ fill: var(--ink); }}
.bar-ok {{ fill: var(--accent); }}
.bar-flag {{ fill: var(--flag); }}
.bar-raw {{ fill: var(--raw); }}
.whisker {{ stroke: var(--ink); stroke-width: 1.2; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1rem; font: .8rem var(--mono); color: var(--muted); }}
.sw {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 2px; vertical-align: -1px; margin-right: .35rem; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: .92rem; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--rule); }}
th {{ font: 500 .72rem/1.2 var(--mono); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }}
td:first-child {{ font-family: var(--mono); }}
ul {{ margin: 0; padding-left: 1.2rem; display: grid; gap: .4rem; max-width: 65ch; }}
code {{ font: .85em var(--mono); background: var(--bg); padding: .1em .3em; border-radius: 3px; overflow-wrap: anywhere; }}
.note {{ font-size: .88rem; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 1rem; }}
</style>
<main>
<header>
  <div class="eyebrow">IBM Quantum cloud runs · 15 October 2026</div>
  <h1>ibm_kingston Persistence Tests</h1>
  <p class="lede">Six qubits of IBM's ibm_kingston processor were read out thousands of times in a row to look for a hidden variable that persists between measurements, in two models built for this project. Neither model shows a signal. The ordinary effects the runs uncovered, leakage and readout memory, matter for error correction.</p>
</header>

<section aria-label="Main results">
  <div class="cards">
    <div class="card"><div class="k">Qubit-anchored model</div><div class="v">w<sub>1</sub>w<sub>2</sub> &lt; {fmt(qlim['1.44'])}</div>
      <p>Five-sigma limit at a 1.44 µs readout spacing; {fmt(qlim['5.44'])} at 5.44 µs and {fmt(qlim['41.4'])} at 41.4 µs. The shape fit agrees: w<sub>Q</sub> &lt; {fmt(wq2)}.</p></div>
    <div class="card"><div class="k">Detector-anchored model</div><div class="v">w<sub>D</sub> &lt; {fmt(wd4)}</div>
      <p>No cusp in the angle scan. With leaked blocks removed, qubits 21 and 89 give {fmt(lim21)} and {fmt(lim89)}, assuming ordinary effects never push the correlation negative.</p></div>
    <div class="card"><div class="k">What the hardware showed</div><div class="v">Leakage and readout memory</div>
      <p>Leakage episodes lasting tens of microseconds, driven by elapsed time rather than readouts, plus readout memory of either sign on four qubits.</p></div>
  </div>
</section>

<section>
  <h2>What was tested</h2>
  <p>Donadi and Hossenfelder's 2022 toy model picks each measurement outcome by comparing a hidden random number with the Born probability, but it says nothing about how that number changes between measurements. A draft theory note in this project adds two minimal completions: the hidden variable belongs either to the detector or to the qubit, and survives from one readout to the next with probability w. Both keep every single readout at its Born value, so the only signature is a correlation between successive outcomes.</p>
  <p>These completions are constructions made for this project, not proposals by Donadi or Hossenfelder. The runs bound these two models on this processor and readout. They do not test superdeterminism in general or Hossenfelder's 2025 collapse model.</p>
</section>

<section>
  <h2>How the runs worked</h2>
  <ul>
    <li>Each qubit ran a chain of 16 to 66 readouts per shot with no resets in between. Every setting was a frame phase on the same two π/2 pulses, so ordinary physics can only produce smooth, low-order dependence on the preparation angle.</li>
    <li>Random relabeling of each readout separates detector-carried effects from qubit-carried ones, and a scan of the preparation angle looks for the cusp that a threshold hidden variable produces.</li>
    <li>The design and the analysis code were committed before any data existed, for the main run and again for the follow-up.</li>
    <li>QPU time charged: 12 s for the pilot, 2 s for a first main-run attempt that IBM rejected before running, {charged_main} s for the main run and {charged_fol} s for the leak-tagged follow-up. That is 580 s of the free plan's 600 s per 28 days.</li>
  </ul>
</section>

<section>
  <h2>Leakage explains the shot-wide correlation</h2>
  <p>In the main run, neighboring readouts were correlated on every qubit, and the correlation stayed nearly as strong two to eight readouts apart. A hidden variable that persists from readout to readout cannot produce that pattern. The follow-up placed two flip pulses after every four readouts: a qubit that has leaked out of its two levels reads "1" three times in a row instead of alternating, which tags the block.</p>
  <figure>
    <div class="legend"><span><span class="sw" style="background:var(--raw)"></span>all blocks</span><span><span class="sw" style="background:var(--accent)"></span>leak-tagged blocks removed</span></div>
    <div class="scroll">{bar_chart_cross()}</div>
    <figcaption><b>Figure 1.</b> Covariance between readouts in consecutive six-readout blocks, about 8.7 µs apart, per qubit, at a 1.44 µs readout spacing (follow-up condition T0, 173,600 shots per qubit). Grey: every block; teal: only blocks whose own check and the previous block's check alternate normally. Whiskers show ±1 standard error. Removing tagged blocks takes the pooled value from 0.0011 to 0.0065 per qubit to {num(fol['conditions']['T0']['pooled']['cross_block_clean'][0], 4)} ± {fol['conditions']['T0']['pooled']['cross_block_clean'][1]:.4f}.</figcaption>
  </figure>
  <p>The leaked fraction barely changed when the readout rate dropped by a factor of 3.8, so the leakage builds up with elapsed time rather than being caused by the readouts. Lifetimes from the tags are 9 to 124 µs; these are lower limits.</p>
</section>

<section>
  <h2>What remains: readout memory</h2>
  <figure>
    <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>within five standard errors of zero</span><span><span class="sw" style="background:var(--flag)"></span>beyond five standard errors</span></div>
    <div class="scroll">{bar_chart_memory()}</div>
    <figcaption><b>Figure 2.</b> Covariance of neighboring readouts after leak-tagged blocks are removed, per qubit, at a 1.44 µs spacing (follow-up condition T0), with the pulse-amplitude term subtracted. Whiskers span ±5 standard errors. A persisting detector variable can only add a positive value. The negative values on qubits 154, 11 and 149 show that ordinary readout memory of either sign is present on this device.</figcaption>
  </figure>
  <div class="table-wrap"><table>
    <thead><tr><th>Qubit</th><th>Clean neighbor covariance</th><th>Reading</th></tr></thead>
    <tbody>
{rows_s2}
    </tbody>
  </table></div>
  <p>On qubit 109 the memory grows to {num(fol['conditions']['T2']['per_qubit'][4]['D_fix_clean'][0], 4)} at a 41.4 µs spacing. A hidden variable that decays between readouts would weaken with spacing, so this, too, is conventional. Because negative contributions occur on this device, the limits from single qubits rest on an assumption. The shape-fit limit in the second card does not.</p>
</section>

<section>
  <h2>Why it matters beyond the models</h2>
  <p>Leakage that lasts for many readout rounds and readout memory at the 10<sup>−3</sup> level are the kinds of correlated measurement errors that degrade surface-code memory in simulations by Kam and colleagues (Quantum Sci. Technol., 2025). Seeing them took about 10<sup>7</sup> readout pairs per qubit, well within a free cloud allowance.</p>
</section>

<p class="note">Records, run reports and code are in the project repository: run data under <code>data/cloudrun/</code>, the design notes <code>notes/CLOUD_RUN_DESIGN.md</code> and <code>notes/CLOUD_RUN_FOLLOWUP_DESIGN.md</code>, the results <code>notes/CLOUD_RUN_RESULTS.md</code>, and the analysis in <code>analysis/cloud_run_*.py</code>. The leakage model fitted after the main run is exploratory. This is independent work using a personal IBM Quantum account; no lab or institution has endorsed it.</p>
</main>
"""
out = os.path.join(REPO, "notes", "cloud_run_results.html")
open(out, "w", encoding="utf-8").write(html)
print(out, len(html), "bytes; main", charged_main, "followup", charged_fol)
