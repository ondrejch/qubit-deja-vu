#!/usr/bin/env python3
"""Build the circuits of the IBM cloud run (notes/CLOUD_RUN_DESIGN.md).

Each qubit runs a reset-free chain of K records. Record k is

  [delay] rz(beta_(k-1) - alpha_k) sx rz(alpha_k - beta_k) sx barrier measure_2

with beta_(-1) = 0. It applies step(alpha_k, beta_k) = P(beta_k) P(alpha_k) of
analysis/cloud_run_design.py (P(g): pi/2 rotation about (cos g, sin g, 0)),
up to a z rotation that commutes with the readout, and then reads out. The
trailing frame phase of each step is carried into the next step's leading rz,
so no diagonal gate sits directly before a readout where a compiler could
drop it. Every setting is an rz angle, bound per parameter set, so all
records use the same physical pulses.

The script builds the circuits and parameter arrays, checks them offline
against an IBM fake backend (transpiled at optimization level 0: gate counts,
the rz before each readout, parameters, the step convention, and the
estimated duration), and saves them with the drawn settings. It never
contacts IBM and submits nothing. Requires qiskit and qiskit-ibm-runtime.
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
import cloud_run_design as design  # noqa: E402
import cloud_run_followup as followup  # noqa: E402

from qiskit import QuantumCircuit, qpy, transpile  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.quantum_info import DensityMatrix  # noqa: E402

try:
    from qiskit_ibm_runtime.circuit import MidCircuitMeasure
except ImportError:  # pragma: no cover
    MidCircuitMeasure = None

AMPLITUDE_PULSES = (1, 9, 25, 49, 97)  # sx counts = 1 mod 4: linear gain in amplitude error


def chain_circuit(K: int, n_qubits: int, delay_dt: int, use_measure_2: bool) -> QuantumCircuit:
    """Parameterized chain on n_qubits logical qubits; parameters are ordered
    qubit-major, step-major, (lead, mid) = (beta_(k-1) - alpha_k, alpha_k - beta_k)."""
    qc = QuantumCircuit(n_qubits, n_qubits * K, name=f"chain_K{K}_d{delay_dt}")
    params = []
    for q in range(n_qubits):
        for k in range(K):
            params.append(tuple(Parameter(f"{x}_{q}_{k}") for x in ("lead", "mid")))
    for k in range(K):
        for q in range(n_qubits):
            lead, mid = params[q * K + k]
            if delay_dt > 0:
                qc.delay(delay_dt, q, unit="dt")
            qc.rz(lead, q)
            qc.sx(q)
            qc.rz(mid, q)
            qc.sx(q)
        qc.barrier()
        for q in range(n_qubits):
            if use_measure_2:
                qc.append(MidCircuitMeasure(), [q], [q * K + k])
            else:
                qc.measure(q, q * K + k)
    return qc


def generation_names(K: int, n_qubits: int) -> list:
    """Parameter names in the column order produced by draw_settings."""
    return [f"{x}_{q}_{k}" for q in range(n_qubits) for k in range(K) for x in ("lead", "mid")]


def order_for(circuit, K: int, n_qubits: int) -> np.ndarray:
    """Column permutation that puts draw_settings values in circuit.parameters
    order, which Qiskit sorts by name (a PUB binds columns in that order)."""
    col = {name: i for i, name in enumerate(generation_names(K, n_qubits))}
    return np.array([col[p.name] for p in circuit.parameters])


def amplitude_circuits(n_qubits: int, use_measure_2: bool) -> list:
    circuits = []
    for m in AMPLITUDE_PULSES:
        qc = QuantumCircuit(n_qubits, n_qubits, name=f"amplitude_{m}")
        for _ in range(m):
            qc.sx(range(n_qubits))
        qc.barrier()
        for q in range(n_qubits):
            if use_measure_2:
                qc.append(MidCircuitMeasure(), [q], [q])
            else:
                qc.measure(q, q)
        circuits.append(qc)
    return circuits


def draw_settings(kind: str, K: int, n_qubits: int, n_sets: int, seed: int):
    """Phases for n_sets parameter sets; returns (values (n_sets, 2 K n_qubits), meta)."""
    rng = np.random.default_rng(seed)
    n = n_sets * n_qubits
    if kind == "balanced":
        alpha, beta, relabel = design.balanced_settings(n, K, rng)
        extra = {"relabel": relabel.reshape(n_sets, n_qubits, K)}
    elif kind == "probe":
        alpha, beta, phi = design.probe_settings(n, K, rng)
        extra = {"phi": phi.reshape(n_sets, n_qubits, K)}
    elif kind == "tagged":
        alpha, beta, phi = followup.tagged_settings(n, K, rng)
        extra = {"phi": phi.reshape(n_sets, n_qubits, K)}
    elif kind == "phase":
        alpha = rng.uniform(0.0, 2 * math.pi, (n, K))
        phi = rng.uniform(0.0, 2 * math.pi, (n, K))
        beta = alpha + phi
        extra = {"phi": phi.reshape(n_sets, n_qubits, K)}
    else:
        raise ValueError(kind)
    alpha = alpha.reshape(n_sets, n_qubits, K)
    beta = beta.reshape(n_sets, n_qubits, K)
    prev_beta = np.concatenate([np.zeros(alpha.shape[:-1] + (1,)), beta[..., :-1]], axis=-1)
    vals = np.stack([prev_beta - alpha, alpha - beta], axis=-1)  # (sets, qubits, K, 2)
    vals = np.mod(vals + math.pi, 2 * math.pi) - math.pi
    meta = {"alpha": alpha, "beta": beta, **extra}
    return vals.reshape(n_sets, -1), meta


def check_step_convention(steps: int = 3) -> float:
    """Largest Bloch-vector mismatch between a run of circuit steps (with the
    carried frame phase) and design.apply_step, after removing the final
    pending z rotation rz(beta_last), which commutes with the readout."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20):
        alpha = rng.uniform(0, 2 * math.pi, steps)
        beta = rng.uniform(0, 2 * math.pi, steps)
        theta, ph = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.rz(ph, 0)
        prev = 0.0
        for a, b in zip(alpha, beta):
            qc.rz(prev - a, 0)
            qc.sx(0)
            qc.rz(a - b, 0)
            qc.sx(0)
            prev = b
        qc.rz(prev, 0)
        rho = DensityMatrix(qc).data
        bloch = np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])
        v = np.array([[math.sin(theta) * math.cos(ph), math.sin(theta) * math.sin(ph), math.cos(theta)]])
        for a, b in zip(alpha, beta):
            v = design.apply_step(v, np.array([a]), np.array([b]))
        worst = max(worst, float(np.abs(bloch - v[0]).max()))
    return worst


def pick_qubits(backend, n_qubits: int, use_measure_2: bool) -> list:
    """Lowest readout error, pairwise non-adjacent, measure_2 available if used."""
    props = backend.properties()
    target = backend.target
    edges = set()
    for a, b in backend.coupling_map.get_edges():
        edges.add((a, b))
        edges.add((b, a))
    ranked = sorted(range(backend.num_qubits), key=lambda q: props.readout_error(q))
    chosen = []
    for q in ranked:
        if use_measure_2 and (q,) not in target["measure_2"]:
            continue
        if any((q, c) in edges for c in chosen):
            continue
        chosen.append(q)
        if len(chosen) == n_qubits:
            break
    return chosen


# Main job: condition label, chain kind, added delay before each step (s),
# records per shot, QPU seconds (analysis/cloud_run_design.py PLAN).
# C2 reads one probe step from the initial |0>: both completions keep that
# marginal at its Born value -cos(phi), so its harmonics measure phase generation.
MAIN_CONDITIONS = [("A0", "balanced", 0.0, 64, 60.0), ("A1", "balanced", 4e-6, 64, 60.0),
                   ("A2", "balanced", 40e-6, 16, 60.0), ("B0", "probe", 0.0, 64, 180.0),
                   ("C2", "phase", 0.0, 1, 3.2)]
PILOT_CONDITIONS = [("P0", "probe", 0.0, 64, None), ("C2", "phase", 0.0, 1, 2.4)]


def build_all(backend, n_qubits: int = 6, shots: int = 400, seed: int = 20261013, layout=None,
              pilot: bool = False, pilot_sets: int = 64, followup_job: bool = False,
              conditions: list | None = None, amplitude: bool = True) -> dict:
    """Transpile every condition for `backend` (a live or fake BackendV2).

    Returns {"pubs": [(isa, values, shots)], "labels", "meta", "report"}. The
    number of parameter sets per condition follows from its QPU-second share
    and the backend's rep_delay and readout duration. Nothing is submitted.
    """
    target = backend.target
    use_m2 = "measure_2" in target.operation_names and MidCircuitMeasure is not None
    meas_name = "measure_2" if use_m2 else "measure"
    layout = layout or pick_qubits(backend, n_qubits, use_m2)
    dt = target.dt
    rep_delay = getattr(backend, "default_rep_delay", None) or design.HARDWARE["rep_delay_s"]
    t_meas = target[meas_name][(layout[0],)].duration
    t_sx = target["sx"][(layout[0],)].duration
    report = {"backend": backend.name, "measure": meas_name, "layout": list(layout), "dt_s": dt,
              "rep_delay_s": rep_delay, "readout_s": t_meas, "sx_s": t_sx, "shots": shots, "checks": []}

    def ok(label: str, cond: bool) -> None:
        report["checks"].append(label)
        if not cond:
            raise AssertionError(label)

    mismatch = check_step_convention()
    ok(f"step convention matches the design script (max Bloch error {mismatch:.1e})", mismatch < 1e-9)
    pubs, labels, meta_all = [], [], {}
    if conditions is None:
        conditions = PILOT_CONDITIONS if pilot else (followup.FOLLOWUP_CONDITIONS if followup_job else MAIN_CONDITIONS)
    for i, (label, kind, delay, K, seconds) in enumerate(conditions):
        delay_dt = int(round(delay / dt))
        t_exec = rep_delay + K * (2 * t_sx + t_meas + delay_dt * dt)
        sets = pilot_sets if seconds is None else max(1, int(round(seconds / (t_exec * shots))))
        isa = transpile(chain_circuit(K, n_qubits, delay_dt, use_m2), backend=backend,
                        initial_layout=layout, optimization_level=0)
        ops = isa.count_ops()
        ok(f"{label}: {2 * K * n_qubits} sx", ops.get("sx", 0) == 2 * K * n_qubits)
        ok(f"{label}: {K * n_qubits} {meas_name}", ops.get(meas_name, 0) == K * n_qubits)
        ok(f"{label}: all {2 * K * n_qubits} rz kept", ops.get("rz", 0) == 2 * K * n_qubits)
        ok(f"{label}: parameters preserved", isa.num_parameters == 2 * K * n_qubits)
        ok(f"{label}: no reset or extra gates", set(ops) <= {"sx", "rz", meas_name, "barrier", "delay"})
        vals, meta = draw_settings(kind, K, n_qubits, sets, seed + i)
        vals = vals[:, order_for(isa, K, n_qubits)]
        ok(f"{label}: binds", isa.assign_parameters(vals[0]).num_parameters == 0)
        duration = isa.estimate_duration(target, unit="s")
        report[label] = {"kind": kind, "records": K, "delay_dt": delay_dt, "sets": sets,
                         "executions": sets * shots, "parameters": isa.num_parameters,
                         "estimated_duration_us": round(duration * 1e6, 2),
                         "estimated_qpu_s": round(sets * shots * (rep_delay + duration), 1), "ops": dict(ops)}
        pubs.append((isa, vals, shots))
        labels.append(label)
        meta_all[label] = {"kind": kind, "records": K, **meta}
    for c in (amplitude_circuits(n_qubits, use_m2) if amplitude else []):
        isa = transpile(c, backend=backend, initial_layout=layout, optimization_level=0)
        m = int(c.name.split("_")[1])
        ok(f"{c.name}: {m * n_qubits} sx", isa.count_ops().get("sx", 0) == m * n_qubits)
        pubs.append((isa, None, shots))
        labels.append(c.name)
    report["estimated_qpu_s_total"] = round(sum(report[l]["estimated_qpu_s"] for l in labels if l in report)
                                            + amplitude * len(AMPLITUDE_PULSES) * shots * (rep_delay + 100 * t_sx + t_meas), 1)
    return {"pubs": pubs, "labels": labels, "meta": meta_all, "report": report}


def outcomes(bit_array, n_qubits: int, K: int) -> np.ndarray:
    """BitArray of one chain PUB -> outcomes (sets, shots, qubits, K), +1 = |0>.

    Classical bit q * K + k holds record k of qubit q.
    """
    bits = bit_array.to_bool_array(order="little")
    return (1 - 2 * bits.astype(np.int8)).reshape(bits.shape[:-1] + (n_qubits, K))


def flatten(r: np.ndarray, meta: dict):
    """(sets, shots, qubits, K) outcomes and their settings -> rows (N, K)
    of outcomes, alpha, beta (and phi for probe chains) for the analysis
    functions of analysis/cloud_run_design.py. Rows are (set, shot, qubit)."""
    sets, shots, nq, K = r.shape

    def rows(x):
        return np.broadcast_to(x[:, None], (sets, shots, nq, K)).reshape(-1, K)

    out = {"r": r.reshape(-1, K), "alpha": rows(meta["alpha"]), "beta": rows(meta["beta"])}
    if "phi" in meta:
        out["phi"] = rows(meta["phi"])
    return out


def aer_end_to_end(seed: int = 3) -> dict:
    """Run small logical chains on an ideal simulator and analyze them with the
    design script's functions: checks bit order, signs and the step convention."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    sim = AerSimulator(seed_simulator=seed)
    sampler = AerSampler.from_backend(sim, seed=seed)
    n_qubits, K = 2, 8
    res = {}
    for kind, sets, shots in (("balanced", 64, 250), ("probe", 128, 125)):
        qc = transpile(chain_circuit(K, n_qubits, 0, False), sim, optimization_level=0)
        vals, meta = draw_settings(kind, K, n_qubits, sets, seed)
        vals = vals[:, order_for(qc, K, n_qubits)]
        bits = sampler.run([(qc, vals, shots)]).result()[0].data.c
        data = flatten(outcomes(bits, n_qubits, K), meta)
        if kind == "balanced":
            st = design.balanced_statistics(data["r"], data["alpha"], data["beta"])
            res[kind] = {k: st[k] for k in ("d_lag1", "q_lag2")}
        else:
            r, phi = data["r"], data["phi"]
            y = np.concatenate([r[:, k - 1] * r[:, k] for k in range(1, K, 2)])
            x = np.concatenate([-np.cos(phi[:, k]) for k in range(1, K, 2)])
            slope = float(np.sum(x * y) / np.sum(x * x))
            resid = y - slope * x
            se = float(math.sqrt(np.sum(resid ** 2) / (len(y) - 1) / np.sum(x * x)))
            res[kind] = {"slope_on_minus_cos_phi": (slope, se)}
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="FakeKingston", help="fake backend class for the offline check")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--shots", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--pilot", action="store_true", help="build the pilot job instead of the main job")
    parser.add_argument("--aer", action="store_true", help="also run the end-to-end check on Qiskit Aer")
    parser.add_argument("--out", help="directory for circuits (QPY), parameter values and settings (NPZ)")
    args = parser.parse_args()

    import qiskit_ibm_runtime.fake_provider as fake_provider

    backend = getattr(fake_provider, args.backend)()
    built = build_all(backend, args.qubits, args.shots, args.seed, pilot=args.pilot)
    report = built["report"]
    if args.aer:
        e2e = aer_end_to_end()
        report["aer_end_to_end"] = e2e
        for key, (val, se) in list(e2e["balanced"].items()):
            report["checks"].append(f"Aer balanced chain: {key} = 0 within 5 SE")
            if abs(val) > 5 * se:
                raise AssertionError(f"Aer {key} = {val:.4f} +- {se:.4f}")
        slope, se = e2e["probe"]["slope_on_minus_cos_phi"]
        report["checks"].append("Aer probe pairs follow -cos(phi) (slope 1 within 5 SE)")
        if abs(slope - 1.0) > 5 * se:
            raise AssertionError(f"Aer probe slope {slope:.4f} +- {se:.4f}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "circuits.qpy"), "wb") as fh:
            qpy.dump([pub[0] for pub in built["pubs"]], fh)
        for label, meta in built["meta"].items():
            vals = built["pubs"][built["labels"].index(label)][1]
            np.savez_compressed(os.path.join(args.out, f"{label}_settings.npz"), values=vals,
                                **{k: v for k, v in meta.items() if isinstance(v, np.ndarray)})
        with open(os.path.join(args.out, "build_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=1, default=str))
    print(f"PASS: {len(report['checks'])} checks")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
