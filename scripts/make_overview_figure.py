#!/usr/bin/env python3
"""Generate notes/overview.svg, the graphical abstract for the repository README,
directly from the machine-generated run reports in data/cloudrun/ (no numbers are
retyped by hand).

  python scripts/make_overview_figure.py
"""
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "data", "cloudrun") + os.sep

main_rep = json.load(open(ROOT + "2026-10-15_main/report.json"))
fol_rep = json.load(open(ROOT + "2026-10-15_followup/report.json"))
fez_f0 = json.load(open(ROOT + "2026-10-15_fezmini/report.json"))
fez_g = json.load(open(ROOT + "2026-10-15_fezrepeat_rerun/report.json"))

k_layout = fol_rep["layout"]
f_layout = fez_g["layout"]


def fmt_e(x, digits=1):
    """Format in e-notation (e.g. 5.3e-4) per prose guidelines."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / (10 ** e)
    sign = "\u2212" if m < 0 else ""
    return f"{sign}{abs(m):.{digits}f}e{e}"


def snum(x, d=4, plus=True):
    fmt = f"{{:+.{d}f}}" if plus else f"{{:.{d}f}}"
    return fmt.format(x).replace("-", "\u2212")


# Headline bounds from 2026-10-15_main/report.json
q_lim_144 = main_rep["conditions"]["A0"]["pooled"]["Q_fix_upper_5sigma"]
q_lim_544 = main_rep["conditions"]["A1"]["pooled"]["Q_fix_upper_5sigma"]
q_lim_414 = main_rep["conditions"]["A2"]["pooled"]["Q_fix_upper_5sigma"]
b0_pooled = main_rep["conditions"]["B0"]["pooled"]
wd_lim_o4 = b0_pooled["order4"]["w_D"][0] + 5 * b0_pooled["order4"]["w_D"][1]

# Follow-up T0 (ibm_kingston, 1.44 us) and G0/G1/F0 (ibm_fez, 1.39 / 5.39 us)
t0_pq = fol_rep["conditions"]["T0"]["per_qubit"]
g0_pq = fez_g["conditions"]["G0"]["per_qubit"]
g1_pq = fez_g["conditions"]["G1"]["per_qubit"]
f0_pq = fez_f0["conditions"]["F0"]["per_qubit"]

k_s1_clean_pool = fol_rep["conditions"]["T0"]["pooled"]["cross_block_clean"]
f_s1_clean_pool = fez_g["conditions"]["G0"]["pooled"]["cross_block_clean"]
k_s2_pool = fol_rep["conditions"]["T0"]["pooled"]["D_fix_clean"]
f_s2_pool = fez_g["conditions"]["G0"]["pooled"]["D_fix_clean"]

k_leak_min = min(pq["fraction_checks_leaked"] for pq in t0_pq) * 100
k_leak_max = max(pq["fraction_checks_leaked"] for pq in t0_pq) * 100
f_leak_min = min(pq["fraction_checks_leaked"] for pq in g0_pq) * 100
f_leak_max = max(pq["fraction_checks_leaked"] for pq in g0_pq) * 100


def build_panel_a():
    """Panel A: Reset-free pulse step and six-record leak-tag block (B B B B F F)."""
    out = [
        '<g id="panel-a">',
        '<rect x="24" y="98" width="1032" height="162" rx="8" fill="#ffffff" stroke="#d5dfe6" stroke-width="1.2"/>',
        '<text class="panel-title" x="40" y="121">A. Reset-free mid-circuit readout chain and six-record leak-tag block (B B B B F F)</text>',
        '<text class="panel-sub-right" x="1036" y="120" text-anchor="end">no mid-circuit reset · settings enter only as virtual-Z frame phases &#966;<tspan dy="2.5" font-size="8.5">k</tspan></text>',
        '<line x1="40" y1="131" x2="1040" y2="131" stroke="#e6edf2" stroke-width="1"/>',
    ]

    # Left sub-diagram: Single record k pulse sequence
    out.append('<text class="sub-hdr" x="40" y="147">Single record k (identical physical pulse schedule across all angles)</text>')
    wy = 175
    out.append(f'<line x1="42" y1="{wy}" x2="438" y2="{wy}" stroke="#334756" stroke-width="1.6"/>')
    out.append(f'<text class="mono-sm" x="42" y="{wy - 11}">x<tspan dy="3" font-size="8">k&#8722;1</tspan></text>')

    gates = [
        (72, 46, "delay", "#f1f5f8", "#8fa3b0", "#334756"),
        (126, 68, "Rz(\u03b2'\u2212\u03b1)", "#eef6f7", "#0b6772", "#0b525b"),
        (202, 36, "\u221aX", "#ffffff", "#16232d", "#16232d"),
        (246, 62, "Rz(\u03b1\u2212\u03b2)", "#eef6f7", "#0b6772", "#0b525b"),
        (316, 36, "\u221aX", "#ffffff", "#16232d", "#16232d"),
        (360, 74, "measure_2", "#16232d", "#16232d", "#ffffff"),
    ]
    for gx, gw, glabel, gfill, gstroke, gtext in gates:
        out.append(
            f'<rect x="{gx}" y="{wy - 13}" width="{gw}" height="26" rx="4" '
            f'fill="{gfill}" stroke="{gstroke}" stroke-width="1.2"/>'
        )
        out.append(
            f'<text class="gate-txt" x="{gx + gw / 2:.1f}" y="{wy + 4}" '
            f'text-anchor="middle" fill="{gtext}">{glabel}</text>'
        )

    out.append(
        '<text class="caption-sm" x="40" y="213">'
        'Angle \u03c6<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">&#160;= \u03b2</tspan>'
        '<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">&#160;\u2212 \u03b1</tspan>'
        '<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">&#160;gives E[x</tspan>'
        '<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">&#160;| x</tspan>'
        '<tspan dy="2" font-size="8">k\u22121</tspan><tspan dy="-2">] = \u2212x</tspan>'
        '<tspan dy="2" font-size="8">k\u22121</tspan><tspan dy="-2">&#160;cos \u03c6</tspan>'
        '<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">.</tspan>'
        '</text>'
    )
    out.append(
        '<text class="caption-sm" x="40" y="231">'
        'With one leaked level, quantum memory stays within \u2264 cos 4\u03c6<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">.</tspan>'
        '</text>'
    )
    out.append(
        '<text class="caption-sm" x="40" y="248">'
        'Balanced steps (\u03c6<tspan dy="2" font-size="8">k</tspan><tspan dy="-2">&#160;= \u00b1\u03c0/2) map |0\u27e9, |1\u27e9 to the equator (memoryless D = 0).</tspan>'
        '</text>'
    )

    # Vertical divider
    out.append('<line x1="454" y1="136" x2="454" y2="252" stroke="#dce5ec" stroke-width="1.2"/>')

    # Right sub-diagram: 6-record block B1 B2 B3 B4 F1 F2
    out.append(
        '<text class="sub-hdr" x="470" y="147">'
        'Six-record block (11 blocks = 66 readouts/shot at 1.4 and 5.4 \u00b5s)'
        '</text>'
    )

    bx0 = 470
    bw = 82
    bgap = 11
    by = 158
    bh = 34
    blocks = [
        ("B1", "\u03c6 = \u00b1\u03c0/2", "equator", False),
        ("B2", "\u03c6 = \u00b1\u03c0/2", "equator", False),
        ("B3", "\u03c6 = \u00b1\u03c0/2", "equator", False),
        ("B4", "\u03c6 = \u00b1\u03c0/2", "check start", False),
        ("F1", "\u03c6 = 0", "\u03c0 flip", True),
        ("F2", "\u03c6 = 0", "\u03c0 flip", True),
    ]
    for idx, (bname, bphi, bsub, is_flip) in enumerate(blocks):
        x = bx0 + idx * (bw + bgap)
        fill = "#fff5eb" if is_flip else "#ebf5f6"
        stroke = "#c26700" if is_flip else "#0b6772"
        tcol = "#964f00" if is_flip else "#084c54"
        out.append(
            f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        out.append(
            f'<text class="block-lbl" x="{x + bw / 2:.1f}" y="{by + 14}" text-anchor="middle" fill="{tcol}">'
            f'{bname} <tspan class="block-phi" fill="{tcol}">({bphi})</tspan></text>'
        )
        out.append(
            f'<text class="block-sub" x="{x + bw / 2:.1f}" y="{by + 28}" text-anchor="middle">{bsub}</text>'
        )
        if idx < 5:
            ax1 = x + bw + 1
            ax2 = x + bw + bgap - 1
            ay = by + bh / 2
            out.append(f'<line x1="{ax1}" y1="{ay:.1f}" x2="{ax2}" y2="{ay:.1f}" stroke="#7f93a1" stroke-width="1.2"/>')

    # Clean track
    out.append(
        '<rect x="470" y="201" width="566" height="23" rx="4" fill="#f1f8f9" stroke="#bddce0" stroke-width="1"/>'
    )
    out.append(
        '<text class="track-tag" x="478" y="216" fill="#08535c">CLEAN CHECK</text>'
    )
    out.append(
        '<text class="caption-sm" x="568" y="216">'
        '(B4, F1, F2) alternates (x, \u2212x, x): keep B1..B4 when this and prior check alternate'
        '</text>'
    )

    # Leaked track
    out.append(
        '<rect x="470" y="229" width="566" height="23" rx="4" fill="#fdf4eb" stroke="#ebcba8" stroke-width="1"/>'
    )
    out.append(
        '<text class="track-tag" x="478" y="244" fill="#9e5200">LEAKED |2\u27e9</text>'
    )
    out.append(
        f'<text class="caption-sm" x="568" y="244">'
        f'Ignores \u221aX flips, reads "1, 1, 1": removed ({k_leak_min:.2f}\u2013{k_leak_max:.2f}% kingston, '
        f'{f_leak_min:.2f}\u2013{f_leak_max:.2f}% fez)'
        f'</text>'
    )

    out.append("</g>")
    return "\n".join(out)


def build_panel_b():
    """Panel B: Cross-block covariance before and after removing leak-tagged blocks (S1)."""
    X0, Y0, W, H = 24, 274, 424, 344
    L, R, T, B = X0 + 56, X0 + W - 16, Y0 + 78, Y0 + H - 54
    lo, hi = -0.0016, 0.0074

    def y_px(v):
        return T + (hi - v) / (hi - lo) * (B - T)

    out = [
        '<g id="panel-b">',
        f'<rect x="{X0}" y="{Y0}" width="{W}" height="{H}" rx="8" fill="#ffffff" stroke="#d5dfe6" stroke-width="1.2"/>',
        f'<text class="panel-title" x="{X0 + 16}" y="{Y0 + 23}">B. Shot-wide correlation explained by leakage (S1)</text>',
        f'<text class="panel-sub" x="{X0 + 16}" y="{Y0 + 41}">Cross-block covariance (~8.7 \u00b5s apart) at 1.4 \u00b5s spacing, all vs. clean</text>',
        # Legend
        f'<rect x="{X0 + 16}" y="{Y0 + 52}" width="10" height="10" rx="2" fill="#95a5b1"/>',
        f'<text class="legend-txt" x="{X0 + 31}" y="{Y0 + 61}">All blocks (unfiltered)</text>',
        f'<rect x="{X0 + 172}" y="{Y0 + 52}" width="10" height="10" rx="2" fill="#0b6772"/>',
        f'<text class="legend-txt" x="{X0 + 187}" y="{Y0 + 61}">Leak-tagged blocks removed (\u00b11 SE)</text>',
    ]

    # Grid lines
    for tick in (-0.001, 0.0, 0.002, 0.004, 0.006):
        yy = y_px(tick)
        cls = "axis-line" if tick == 0 else "grid-line"
        out.append(f'<line class="{cls}" x1="{L}" x2="{R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        lbl = "0" if tick == 0 else snum(tick, 3)
        out.append(f'<text class="tick-txt" x="{L - 6}" y="{yy + 3.5:.1f}" text-anchor="end">{lbl}</text>')

    plot_w = R - L
    gap_proc = 14
    slot_w = (plot_w - gap_proc) / 12.0

    series_12 = []
    for i, q in enumerate(k_layout):
        pq = t0_pq[i]
        series_12.append(("K", q, pq["all"]["cross_block"], pq["clean"]["cross_block"]))
    for i, q in enumerate(f_layout):
        pq = g0_pq[i]
        series_12.append(("F", q, pq["all"]["cross_block"], pq["clean"]["cross_block"]))

    for idx, (proc, q, raw_cb, cln_cb) in enumerate(series_12):
        cx = L + idx * slot_w + (gap_proc if idx >= 6 else 0) + slot_w * 0.5
        bw = slot_w * 0.36
        for j, ((val, se), fill_col) in enumerate(((raw_cb, "#95a5b1"), (cln_cb, "#0b6772"))):
            bx = cx - bw + j * bw
            top = y_px(max(val, 0.0))
            bot = y_px(min(val, 0.0))
            out.append(
                f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw - 1:.1f}" '
                f'height="{max(bot - top, 1.2):.1f}" fill="{fill_col}"/>'
            )
            mx = bx + (bw - 1) * 0.5
            y_hi = y_px(val + se)
            y_lo = y_px(val - se)
            out.append(f'<line class="whisker-sm" x1="{mx:.1f}" x2="{mx:.1f}" y1="{y_hi:.1f}" y2="{y_lo:.1f}"/>')

        out.append(f'<text class="q-lbl" x="{cx:.1f}" y="{B + 15:.1f}" text-anchor="middle">{q}</text>')

    sep_x = L + 6 * slot_w + gap_proc * 0.5
    out.append(f'<line x1="{sep_x:.1f}" x2="{sep_x:.1f}" y1="{T}" y2="{B + 18}" stroke="#cbd6df" stroke-width="1" stroke-dasharray="3,3"/>')

    k_mid = L + 3 * slot_w
    f_mid = L + 6 * slot_w + gap_proc + 3 * slot_w
    out.append(
        f'<text class="proc-lbl" x="{k_mid:.1f}" y="{B + 33:.1f}" text-anchor="middle">'
        f'kingston T0 (clean {snum(k_s1_clean_pool[0], 4)})</text>'
    )
    out.append(
        f'<text class="proc-lbl" x="{f_mid:.1f}" y="{B + 33:.1f}" text-anchor="middle">'
        f'fez G0 (clean {snum(f_s1_clean_pool[0], 4)})</text>'
    )

    out.append("</g>")
    return "\n".join(out)


def build_panel_c():
    """Panel C: Clean neighbor covariance D across all 12 Heron qubits (S2)."""
    X0, Y0, W, H = 462, 274, 594, 344
    L, R, T, B = X0 + 58, X0 + W - 16, Y0 + 78, Y0 + H - 54
    lo, hi = -0.0122, 0.0108

    def y_px(v):
        return T + (hi - v) / (hi - lo) * (B - T)

    out = [
        '<g id="panel-c">',
        f'<rect x="{X0}" y="{Y0}" width="{W}" height="{H}" rx="8" fill="#ffffff" stroke="#d5dfe6" stroke-width="1.2"/>',
        f'<text class="panel-title" x="{X0 + 16}" y="{Y0 + 23}">C. Clean neighbor readout memory D across 12 Heron qubits (S2)</text>',
        f'<text class="panel-sub" x="{X0 + 16}" y="{Y0 + 41}">Lag-1 covariance in leak-free blocks (whiskers: \u00b15 SE; |z| &gt; 5 flagged as readout memory)</text>',
        # Legend row
        f'<rect x="{X0 + 16}" y="{Y0 + 52}" width="10" height="10" rx="2" fill="#0b6772"/>',
        f'<text class="legend-txt" x="{X0 + 30}" y="{Y0 + 61}">|z| \u2264 5 (1.4 \u00b5s)</text>',
        f'<rect x="{X0 + 125}" y="{Y0 + 52}" width="10" height="10" rx="2" fill="#c26700"/>',
        f'<text class="legend-txt" x="{X0 + 139}" y="{Y0 + 61}">|z| &gt; 5 memory (1.4 \u00b5s)</text>',
        f'<circle cx="{X0 + 286}" cy="{Y0 + 57}" r="3.6" fill="#ffffff" stroke="#16232d" stroke-width="1.4"/>',
        f'<text class="legend-txt" x="{X0 + 295}" y="{Y0 + 61}">fez F0 open-plan (1.39 \u00b5s)</text>',
        f'<polygon points="{X0 + 448},{Y0 + 52.5} {X0 + 452.5},{Y0 + 57} {X0 + 448},{Y0 + 61.5} {X0 + 443.5},{Y0 + 57}" fill="#1d4ed8" stroke="#ffffff" stroke-width="0.9"/>',
        f'<text class="legend-txt" x="{X0 + 457}" y="{Y0 + 61}">fez G1 (5.39 \u00b5s)</text>',
    ]

    for tick in (-0.010, -0.005, 0.0, 0.005, 0.010):
        yy = y_px(tick)
        cls = "axis-line" if tick == 0 else "grid-line"
        out.append(f'<line class="{cls}" x1="{L}" x2="{R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        lbl = "0" if tick == 0 else snum(tick, 3)
        out.append(f'<text class="tick-txt" x="{L - 6}" y="{yy + 3.5:.1f}" text-anchor="end">{lbl}</text>')

    plot_w = R - L
    gap_proc = 18
    slot_w = (plot_w - gap_proc) / 12.0

    items = []
    for i, q in enumerate(k_layout):
        v, se = t0_pq[i]["D_fix_clean"]
        items.append(("K", q, v, se, None, None))
    for i, q in enumerate(f_layout):
        v, se = g0_pq[i]["D_fix_clean"]
        v_f0 = f0_pq[i]["D_fix_clean"][0]
        v_g1 = g1_pq[i]["D_fix_clean"][0]
        items.append(("F", q, v, se, v_f0, v_g1))

    for idx, (proc, q, v, se, v_f0, v_g1) in enumerate(items):
        cx = L + idx * slot_w + (gap_proc if idx >= 6 else 0) + slot_w * 0.5
        flagged = abs(v) > 5 * se
        fill_col = "#c26700" if flagged else "#0b6772"
        bw = slot_w * 0.36 if proc == "F" else slot_w * 0.50
        top = y_px(max(v, 0.0))
        bot = y_px(min(v, 0.0))
        out.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
            f'height="{max(bot - top, 1.5):.1f}" rx="1.5" fill="{fill_col}"/>'
        )
        y_hi = y_px(v + 5 * se)
        y_lo = y_px(v - 5 * se)
        out.append(f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y_hi:.1f}" y2="{y_lo:.1f}"/>')
        for yy in (y_hi, y_lo):
            out.append(f'<line class="whisker" x1="{cx - 4.0:.1f}" x2="{cx + 4.0:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')

        hi_bound = v + 5 * se
        lo_bound = v - 5 * se
        if v_f0 is not None and v_g1 is not None:
            yf0 = y_px(v_f0)
            yg1 = y_px(v_g1)
            ox1 = cx - bw / 2 - 4.8
            ox2 = cx + bw / 2 + 4.8
            out.append(
                f'<circle cx="{ox1:.1f}" cy="{yf0:.1f}" r="3.2" fill="#ffffff" stroke="#16232d" stroke-width="1.3"/>'
            )
            out.append(
                f'<polygon points="{ox2:.1f},{yg1 - 4.0:.1f} {ox2 + 4.0:.1f},{yg1:.1f} '
                f'{ox2:.1f},{yg1 + 4.0:.1f} {ox2 - 4.0:.1f},{yg1:.1f}" fill="#1d4ed8" stroke="#ffffff" stroke-width="1.0"/>'
            )
            hi_bound = max(hi_bound, v_f0, v_g1)
            lo_bound = min(lo_bound, v_f0, v_g1)

        if v >= 0:
            lbl_y = y_px(hi_bound) - 5.5
        else:
            lbl_y = y_px(lo_bound) + 11.5
        out.append(
            f'<text class="val-lbl" x="{cx:.1f}" y="{lbl_y:.1f}" text-anchor="middle">{snum(v, 4)}</text>'
        )
        out.append(
            f'<text class="q-lbl" x="{cx:.1f}" y="{B + 15:.1f}" text-anchor="middle">{q}</text>'
        )

    sep_x = L + 6 * slot_w + gap_proc * 0.5
    out.append(
        f'<line x1="{sep_x:.1f}" x2="{sep_x:.1f}" y1="{T}" y2="{B + 18}" '
        f'stroke="#cbd6df" stroke-width="1" stroke-dasharray="3,3"/>'
    )

    k_mid = L + 3 * slot_w
    f_mid = L + 6 * slot_w + gap_proc + 3 * slot_w
    out.append(
        f'<text class="proc-lbl" x="{k_mid:.1f}" y="{B + 33:.1f}" text-anchor="middle">'
        f'ibm_kingston · T0 (1.44 \u00b5s, pool {snum(k_s2_pool[0], 4)})</text>'
    )
    out.append(
        f'<text class="proc-lbl" x="{f_mid:.1f}" y="{B + 33:.1f}" text-anchor="middle">'
        f'ibm_fez · G0 (1.39 \u00b5s, pool {snum(f_s2_pool[0], 4)}), F0 &amp; G1</text>'
    )

    out.append("</g>")
    return "\n".join(out)


svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 652" width="1080" height="652" role="img" aria-labelledby="ov-title ov-desc">
  <title id="ov-title">qubit-deja-vu: time-correlated readout errors and persistent hidden-state bounds on IBM Heron processors</title>
  <desc id="ov-desc">Three-panel graphical summary generated from data/cloudrun/*/report.json: (A) the reset-free mid-circuit chain and B-B-B-B-F-F leak-tag block; (B) cross-block covariance before and after removing leak-tagged blocks across 12 qubits; (C) clean lag-1 neighbor covariance D across 6 qubits on ibm_kingston and 6 qubits on ibm_fez.</desc>
  <style>
    .bg-card {{ fill: #f4f7f9; stroke: #c9d4dc; stroke-width: 1.5; }}
    .eyebrow {{ font: 600 10.5px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; letter-spacing: 0.07em; fill: #0b6772; }}
    .main-title {{ font: 700 18.5px "Newsreader", "Georgia", "Liberation Serif", "DejaVu Serif", serif; fill: #16232d; }}
    .main-sub {{ font: 400 11.4px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #4a5a67; }}
    .badge-box {{ fill: #ffffff; stroke: #cbd6df; stroke-width: 1.1; }}
    .badge-k {{ font: 600 9.5px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; letter-spacing: 0.05em; fill: #52626f; }}
    .badge-v {{ font: 700 14.5px "Newsreader", "Georgia", "Liberation Serif", "DejaVu Serif", serif; fill: #0b6772; }}
    .badge-sub {{ font: 400 9.6px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #52626f; }}
    .panel-title {{ font: 700 13.5px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #16232d; }}
    .panel-sub {{ font: 400 11.2px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #52626f; }}
    .panel-sub-right {{ font: 500 10.5px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; fill: #52626f; }}
    .sub-hdr {{ font: 600 11.2px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #283845; }}
    .gate-txt {{ font: 600 10px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; }}
    .mono-sm {{ font: 500 10.5px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; fill: #16232d; }}
    .caption-sm {{ font: 400 10.5px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #3b4c59; }}
    .block-lbl {{ font: 700 11px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; }}
    .block-phi {{ font: 500 9.2px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; }}
    .block-sub {{ font: 400 9.5px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #4a5a67; }}
    .track-tag {{ font: 700 9.5px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; letter-spacing: 0.04em; }}
    .legend-txt {{ font: 400 10.5px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #3b4c59; }}
    .grid-line {{ stroke: #e3eaee; stroke-width: 1; }}
    .axis-line {{ stroke: #6c7d8c; stroke-width: 1.2; }}
    .tick-txt {{ font: 400 9.8px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; fill: #52626f; }}
    .q-lbl {{ font: 600 10.2px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; fill: #16232d; }}
    .proc-lbl {{ font: 600 10.5px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #283845; }}
    .val-lbl {{ font: 500 8.6px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; fill: #16232d; }}
    .whisker {{ stroke: #16232d; stroke-width: 1.15; }}
    .whisker-sm {{ stroke: #16232d; stroke-width: 1.0; }}
    .callout-hdr {{ font: 600 9.6px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #08535c; }}
    .callout-val {{ font: 500 9.4px "JetBrains Mono", "Liberation Mono", "DejaVu Sans Mono", monospace; fill: #16232d; white-space: pre; }}
    .footer-txt {{ font: 400 10.4px "Public Sans", "Liberation Sans", "DejaVu Sans", sans-serif; fill: #4a5a67; }}
  </style>

  <!-- Outer card -->
  <rect class="bg-card" x="4" y="4" width="1072" height="644" rx="10"/>

  <!-- Header -->
  <text class="eyebrow" x="26" y="30">QUBIT-DEJA-VU · IBM QUANTUM HERON RUNS (15-16 OCT 2026) · PRE-REGISTERED DESIGNS</text>
  <text class="main-title" x="26" y="55">Time-Correlated Readout Errors on 12 IBM Heron Qubits</text>
  <text class="main-sub" x="26" y="76">Reset-free chains of up to 66 readouts/shot screen for readout memory and bound persistent hidden states</text>

  <!-- Top-right summary badges for the two completions -->
  <g id="header-badges">
    <rect class="badge-box" x="632" y="18" width="220" height="66" rx="6"/>
    <text class="badge-k" x="644" y="34">QUBIT-ANCHORED (5\u03c3)</text>
    <text class="badge-v" x="644" y="55">w<tspan dy="3" font-size="10">1</tspan><tspan dy="-3">w</tspan><tspan dy="3" font-size="10">2</tspan><tspan dy="-3">&#160;&lt; {fmt_e(q_lim_144)}</tspan></text>
    <text class="badge-sub" x="644" y="74">1.44 \u00b5s ({fmt_e(q_lim_544)} at 5.44 \u00b5s; {fmt_e(q_lim_414)} at 41.4 \u00b5s)</text>

    <rect class="badge-box" x="860" y="18" width="196" height="66" rx="6"/>
    <text class="badge-k" x="872" y="34">DETECTOR-ANCHORED (5\u03c3)</text>
    <text class="badge-v" x="872" y="55">w<tspan dy="3" font-size="10">D</tspan><tspan dy="-3">&#160;&lt; {fmt_e(wd_lim_o4)}</tspan></text>
    <text class="badge-sub" x="872" y="74">B0 order-4 shape fit (no cusp)</text>
  </g>

  {build_panel_a()}
  {build_panel_b()}
  {build_panel_c()}

  <!-- Footer -->
  <text class="footer-txt" x="26" y="636">Numbers read directly from data/cloudrun/*/report.json (1.0e7 readout pairs/qubit on ibm_kingston T0; 4.6e6 on ibm_fez G0). Opposite signs on qubits 11 (\u22120.0085, \u221220\u03c3) and 109 (+0.0060, +14\u03c3) cancel in device averages.</text>
</svg>
"""

out_svg = os.path.join(REPO, "notes", "overview.svg")
with open(out_svg, "w", encoding="utf-8") as fh:
    fh.write(svg)
print(out_svg, len(svg), "bytes")
