#!/usr/bin/env python3
"""Submit a cloud-run job to IBM Quantum and save its records (notes/CLOUD_RUN_DESIGN.md).

Without --submit the script only builds the job against the live backend and
prints the qubit layout and the estimated QPU time; nothing is sent. With
--submit it sends the job, writes the drawn settings and the job identifier
to the run directory before waiting, then saves the results with
cloud_run_analyze.save_job() and writes the analysis report. --retrieve JOB_ID
saves jobs submitted earlier from the same run directory (the identifiers are
read from its job.json when JOB_ID is "auto").

The main job is sent as a batch of jobs: the small calibration PUBs together,
then each chain condition alone, with any PUB above MAX_SETS_PER_PUB
parameter sets split into equal parts. One job holding PUBs of very different
sizes fails with IBM error 1520 (no fit into a shot loop of at most 1000 while
preserving the PUB size ratios), as the first main-job attempt on 15 October
2026 did. Split parts are rejoined into one condition when saved.

  python analysis/cloud_run_submit.py --job pilot                       # build and estimate only
  python analysis/cloud_run_submit.py --job main --submit --out data/cloudrun/2026-10-15_main
  python analysis/cloud_run_submit.py --retrieve JOB_ID --out data/cloudrun/2026-10-15_main
  python analysis/cloud_run_submit.py --job fezmini --backend ibm_fez --submit --out data/cloudrun/2026-10-15_fezmini
  python analysis/cloud_run_submit.py --job fezrepeat --backend ibm_fez --instance-of payg --max-qpu-s 60 --submit --out data/cloudrun/DATE_fezrepeat

Submit only with the author's explicit go-ahead for that specific job
(the author's rule for this project). Needs qiskit-ibm-runtime and a saved IBM Quantum account.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHOTS = {"pilot": 300, "main": 400, "followup": 400, "fezmini": 400, "fezrepeat": 400}
JOB_OVERHEAD_S = 3.0  # charged per job beyond the estimate, from the 15 October jobs
USAGE_MARGIN = 0.75  # the estimate may use at most this fraction of the instance's remaining usage
MAX_SETS_PER_PUB = 700


class _Joined:
    """A PUB result whose bits are several PUB results joined along the set axis."""

    def __init__(self, parts):
        self.data = self
        self.c = self
        self._parts = parts

    def to_bool_array(self, order="little"):
        import numpy as np

        return np.concatenate([p.data.c.to_bool_array(order=order) for p in self._parts], axis=0)


class _JobIds:
    def __init__(self, ids):
        self._ids = ids

    def job_id(self):
        return ",".join(self._ids)


def plan_jobs(built: dict, single_job: bool) -> list:
    """Jobs as lists of (label, pub, (first_set, end_set) or None)."""
    import numpy as np

    entries = []
    for label, pub in zip(built["labels"], built["pubs"]):
        circuit, values, shots = pub
        if values is not None and len(values) > MAX_SETS_PER_PUB:
            k = -(-len(values) // MAX_SETS_PER_PUB)
            edges = np.linspace(0, len(values), k + 1).astype(int)
            entries.append([(label, (circuit, values[a:b], shots), (int(a), int(b)))
                            for a, b in zip(edges[:-1], edges[1:])])
        else:
            entries.append([(label, pub, None)])
    if single_job:
        return [[e for parts in entries for e in parts]]
    big = [i for i, parts in enumerate(entries)
           if parts[0][1][1] is not None and sum(len(e[1][1]) for e in parts) >= 100]
    small = [e for i, parts in enumerate(entries) if i not in big for e in parts]
    return ([small] if small else []) + [entries[i] for i in big]


def assemble(built: dict, plan: list, results: list) -> list:
    """PUB results in built["labels"] order, joining split parts."""
    by_label = {}
    for job_entries, result in zip(plan, results):
        for j, (label, _, span) in enumerate(job_entries):
            by_label.setdefault(label, []).append((span[0] if span else 0, result[j]))
    out = []
    for label in built["labels"]:
        parts = [r for _, r in sorted(by_label[label], key=lambda t: t[0])]
        out.append(parts[0] if len(parts) == 1 else _Joined(parts))
    return out


def _save_report(out: str) -> dict:
    import cloud_run_analyze as cra

    report = cra.analyze(out)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
        fh.write("\n")
    return report


def _usage(job) -> dict:
    out = {}
    for attr in ("usage", "metrics"):
        try:
            out[attr] = getattr(job, attr)()
        except Exception as exc:  # not every runtime version provides both
            out[attr] = f"not available: {type(exc).__name__}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", choices=sorted(SHOTS), default="pilot")
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--layout", help="comma-separated physical qubits (default: the follow-up uses the "
                                         "qubits of the 15 October 2026 run; other jobs pick from calibration)")
    parser.add_argument("--shots", type=int, help="shots per PUB (default: 300 pilot, 400 main)")
    parser.add_argument("--out", help="run directory, e.g. data/cloudrun/2026-10-15_main")
    parser.add_argument("--submit", action="store_true", help="send the job (needs the author's go-ahead)")
    parser.add_argument("--account", help="saved account name (default: the default saved account)")
    parser.add_argument("--instance-of", help="use the default saved key with the instance saved under this name")
    parser.add_argument("--max-qpu-s", type=float,
                        help="refuse to submit if the estimate plus job overhead exceeds this many seconds")
    parser.add_argument("--retrieve", metavar="JOB_ID", help="save a job submitted earlier from --out")
    args = parser.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    import cloud_run_analyze as cra

    if args.instance_of:
        instance = QiskitRuntimeService.saved_accounts()[args.instance_of]["instance"]
        service = QiskitRuntimeService(name=args.account, instance=instance) if args.account else \
            QiskitRuntimeService(instance=instance)
    else:
        service = QiskitRuntimeService(name=args.account) if args.account else QiskitRuntimeService()
    if args.retrieve:
        if not args.out:
            parser.error("--retrieve needs --out")
        with open(os.path.join(args.out, "built.pkl"), "rb") as fh:
            built = pickle.load(fh)
        with open(os.path.join(args.out, "job.json"), encoding="utf-8") as fh:
            record = json.load(fh)
        jobs_meta = record["jobs"]
        if args.retrieve != "auto":
            ids = args.retrieve.split(",")
            jobs_meta = [{**m, "job_id": i} for m, i in zip(jobs_meta, ids)]
        plan = [[(lab, None, tuple(span) if span else None) for lab, span in m["entries"]] for m in jobs_meta]
        jobs = [service.job(m["job_id"]) for m in jobs_meta]
        results = [j.result() for j in jobs]
        cra.save_job(args.out, built, assemble(built, plan, results), _JobIds([j.job_id() for j in jobs]))
        report = _save_report(args.out)
        print(json.dumps({"saved": args.out, "flags": report["flags"]}, indent=1, default=str))
        return 0

    from qiskit_ibm_runtime.executor_sampler import Sampler

    from cloud_run_circuits import build_all

    shots = args.shots or SHOTS[args.job]
    backend = service.backend(args.backend)
    import cloud_run_followup as followup

    layout = [int(x) for x in args.layout.split(",")] if args.layout else (
        followup.LAYOUT if args.job == "followup" else (followup.FEZ_LAYOUT if args.job == "fezrepeat" else None))
    built = build_all(backend, n_qubits=args.qubits, shots=shots, pilot=args.job == "pilot",
                      followup_job=args.job == "followup", layout=layout,
                      conditions={"fezmini": followup.FEZ_MINI_CONDITIONS,
                                  "fezrepeat": followup.FEZ_REPEAT_CONDITIONS}.get(args.job),
                      amplitude=args.job != "fezrepeat")
    rep = built["report"]
    summary = {"backend": backend.name, "job": args.job, "shots": shots, "layout": rep["layout"],
               "readout_s": rep["readout_s"], "estimated_qpu_s": rep["estimated_qpu_s_total"],
               "conditions": {k: {x: rep[k][x] for x in ("sets", "executions", "estimated_qpu_s")}
                              for k in built["labels"] if k in rep}}
    print(json.dumps(summary, indent=1), flush=True)
    n_jobs = len(plan_jobs(built, single_job=args.job in ("pilot", "fezmini", "fezrepeat")))
    total = rep["estimated_qpu_s_total"] + JOB_OVERHEAD_S * n_jobs
    print(f"estimate with overhead: {total:.1f} s in {n_jobs} job(s)", flush=True)
    if args.max_qpu_s is not None and total > args.max_qpu_s:
        print(f"Not submitted: {total:.1f} s exceeds --max-qpu-s {args.max_qpu_s}.")
        return 3
    # IBM cancels running jobs when an instance reaches its usage or cost limit,
    # and on 15 October 2026 a job estimated at 57 s was cancelled after 58 s
    # charged against a 63 s limit. Require the estimate to stay under
    # USAGE_MARGIN of what the instance has left.
    try:
        remaining = service.usage().get("usage_remaining_seconds")
    except Exception:  # some instances do not report usage
        remaining = None
    print(f"instance usage remaining: {remaining} s", flush=True)
    if remaining is not None and total > USAGE_MARGIN * remaining:
        print(f"Not submitted: {total:.1f} s exceeds {USAGE_MARGIN:.0%} of the {remaining} s the instance has left.")
        return 4
    if not args.submit:
        print("Not submitted (no --submit).")
        return 0
    if not args.out:
        parser.error("--submit needs --out")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "built.pkl"), "wb") as fh:
        pickle.dump({k: built[k] for k in ("labels", "meta", "report")}, fh)

    from qiskit_ibm_runtime import Batch

    plan = plan_jobs(built, single_job=args.job in ("pilot", "fezmini", "fezrepeat"))
    record = {"backend": backend.name, "job": args.job, "shots": shots, "account": args.account or "default",
              "instance_of": args.instance_of,
              "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "estimate": summary, "jobs": []}
    prior = os.path.join(args.out, "job.json")
    if os.path.exists(prior):
        with open(prior, encoding="utf-8") as fh:
            old = json.load(fh)
        record["earlier_attempts"] = old.get("earlier_attempts", []) + [
            {k: old[k] for k in ("job_id", "jobs", "submitted", "outcome") if k in old}]
    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        jobs = []
        for job_entries in plan:
            job = sampler.run([e[1] for e in job_entries], shots=shots)
            jobs.append(job)
            record["jobs"].append({"job_id": job.job_id(),
                                   "entries": [[e[0], list(e[2]) if e[2] else None] for e in job_entries]})
            print("submitted job", job.job_id(), [e[0] for e in job_entries], flush=True)
    with open(prior, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    t0 = time.time()
    results = []
    for job, meta in zip(jobs, record["jobs"]):
        try:
            results.append(job.result())
        except Exception as exc:  # report the service's message (size limits, payload, and so on)
            meta["outcome"] = f"failed: {type(exc).__name__}: {str(exc)[:1500]}"
            print("JOB FAILED:", job.job_id(), meta["outcome"], flush=True)
        else:
            meta["outcome"] = "done"
        meta.update(_usage(job))
    record["finished_after_s"] = round(time.time() - t0)
    with open(prior, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    if len(results) != len(jobs):
        return 2
    print(f"results after {record['finished_after_s']} s", flush=True)
    cra.save_job(args.out, built, assemble(built, plan, results), _JobIds([j.job_id() for j in jobs]))
    report = _save_report(args.out)
    print(json.dumps({"saved": args.out, "flags": report["flags"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
