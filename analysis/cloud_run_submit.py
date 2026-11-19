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

With --max-job-s, a long condition is split across jobs instead of into parts of
one job: each part of a split condition runs alone, so no job estimates over the
cap. That is needed for any condition whose QPU share exceeds the cap, since IBM
cancels a running job when its instance hits its usage limit (a 58 s job was
cancelled at a 63 s limit on 15 October). plan_jobs refuses a plan whose job
estimates exceed the cap, and job_estimates prints them.

  python analysis/cloud_run_submit.py --job pilot                       # build and estimate only
  python analysis/cloud_run_submit.py --job main --submit --out data/cloudrun/2026-10-15_main
  python analysis/cloud_run_submit.py --retrieve JOB_ID --out data/cloudrun/2026-10-15_main
  python analysis/cloud_run_submit.py --job fezmini --backend ibm_fez --submit --out data/cloudrun/2026-10-15_fezmini
  python analysis/cloud_run_submit.py --job fezrepeat --backend ibm_fez --instance-of payg --max-qpu-s 60 --submit --out data/cloudrun/DATE_fezrepeat
  python analysis/cloud_run_submit.py --job eig --backend ibm_fez --max-job-s --max-qpu-s 400    # estimate only

Submit only with the author's explicit go-ahead for that specific job
(AGENTS.md). Needs qiskit-ibm-runtime and a saved IBM Quantum account.
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

SHOTS = {"pilot": 300, "main": 400, "followup": 400, "fezmini": 400, "fezrepeat": 400, "eig": 400, "eigrepeat": 400}
JOB_OVERHEAD_S = 3.0  # charged per job beyond the estimate, from the 15 October jobs
USAGE_MARGIN = 0.75  # the estimate may use at most this fraction of the instance's remaining usage
MAX_SETS_PER_PUB = 700
MAX_JOB_S = 60.0  # keep every job's estimate under this (Section 1, item 7 of the next-window note)
BIG_PUB_SETS = 100  # an unsplit PUB this large runs in a job of its own


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


def pub_sets(built: dict, label: str) -> int:
    """Parameter sets in one built PUB (1 for a parameter-free amplitude PUB)."""
    for lab, (_, values, _) in zip(built["labels"], built["pubs"]):
        if lab == label:
            return 1 if values is None else len(values)
    raise KeyError(label)


def per_set_seconds(built: dict, label: str) -> float:
    """Estimated QPU seconds per parameter set of a built PUB, shots included.

    Chain PUBs use the builder's own estimate. Amplitude PUBs have no report
    entry; they follow the builder's duration formula for m pulses.
    """
    info = built["report"].get(label)
    if info and info.get("sets"):
        return info["estimated_qpu_s"] / info["sets"]
    if label.startswith("amplitude_"):
        rep = built["report"]
        m = int(label.split("_")[1])
        return rep["shots"] * (rep["rep_delay_s"] + m * rep["sx_s"] + rep["readout_s"])
    return 0.0


def set_spans(n_sets: int, per_set_s: float, max_job_s: float | None = None,
              max_sets_per_pub: int = MAX_SETS_PER_PUB) -> list:
    """(first_set, end_set) ranges for one PUB.

    Every part stays under max_sets_per_pub (IBM error 1520), and when
    max_job_s is given every part's duration estimate stays under it too. With
    max_job_s unset this is the pre-existing equal-part split of large PUBs.
    """
    import numpy as np

    cap = max_sets_per_pub
    if max_job_s and per_set_s > 0:
        cap = min(cap, max(1, int(max_job_s // per_set_s)))
    if n_sets <= cap:
        return [(0, n_sets)]
    edges = np.linspace(0, n_sets, -(-n_sets // cap) + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def job_plan(entries: list, max_job_s: float | None = None) -> list:
    """Jobs as lists of (label, (first_set, end_set) or None).

    entries is [(label, n_sets, per_set_s)]. A condition the duration cap splits
    contributes one job per part: parts of a duration split never share a job,
    which is what the cap is for. A condition split only by MAX_SETS_PER_PUB
    keeps its parts in one job, as the 15 October main run did. An unsplit PUB of
    at least BIG_PUB_SETS sets gets a job of its own (error 1520), and the rest
    are packed together in order while their estimates stay under max_job_s.
    Small job first, as in the 15 October jobs.
    """
    jobs, big, small = [], [], []
    for label, n_sets, per_set in entries:
        size_spans = set_spans(n_sets, per_set)  # the MAX_SETS_PER_PUB split alone
        spans = size_spans if max_job_s is None else set_spans(n_sets, per_set, max_job_s)
        if len(spans) > len(size_spans):
            jobs.extend([[[label, span]] for span in spans])  # a duration split runs one part per job
        elif n_sets >= BIG_PUB_SETS:
            big.append([[label, span] for span in spans])     # the condition's own job, parts together
        else:
            small.append((label, spans[0], n_sets * per_set))
    packed, group, group_s = [], [], 0.0
    for label, span, seconds in small:
        if group and max_job_s and group_s + seconds > max_job_s:
            packed.append(group)
            group, group_s = [], 0.0
        group.append([label, span])
        group_s += seconds
    if group:
        packed.append(group)
    return packed + big + jobs


def job_estimates(plan: list, built: dict) -> list:
    """Estimated QPU seconds per job of a plan from plan_jobs."""
    out = []
    for job_entries in plan:
        total = 0.0
        for label, (_, values, _), _ in job_entries:
            # values are already sliced to the job's span; a parameter-free PUB is one set
            total += (1 if values is None else len(values)) * per_set_seconds(built, label)
        out.append(round(total, 2))
    return out


def plan_jobs(built: dict, single_job: bool, max_job_s: float | None = None) -> list:
    """Jobs as lists of (label, pub, (first_set, end_set) or None)."""
    entries = [(label, pub_sets(built, label), per_set_seconds(built, label))
               for label in built["labels"]]
    plan = job_plan(entries, max_job_s)
    if single_job:
        plan = [[e for job in plan for e in job]]
    by_label = dict(zip(built["labels"], built["pubs"]))
    out = []
    for job in plan:
        job_entries = []
        for label, span in job:
            circuit, values, shots = by_label[label]
            if values is None or span is None or span == (0, len(values)):
                job_entries.append((label, (circuit, values, shots), None))
            else:
                a, b = span
                job_entries.append((label, (circuit, values[a:b], shots), (a, b)))
        out.append(job_entries)
    return out


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


def _write_record(path: str, record: dict) -> None:
    """Write job.json atomically, so an interrupted write never leaves it truncated."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    os.replace(tmp, path)


def complete_labels(built: dict, plan: list, results: list) -> tuple:
    """Labels whose every part came back, and the indices of the jobs that did not.

    results[i] is job i's result, or None if the job failed or was not
    retrieved. A label with any part missing is left out entirely: its settings
    are the full arrays, so a joined result with a hole would misalign every
    later set.
    """
    failed = [i for i, r in enumerate(results) if r is None]
    spans = {}
    for job, r in zip(plan, results):
        if r is not None:
            for e in job:
                spans.setdefault(e[0], []).append(e[2])
    return [lab for lab in built["labels"] if _covers(spans.get(lab, []), _n_sets(built, lab))], failed


def _n_sets(built: dict, label: str) -> int | None:
    """Parameter sets of a chain condition from its saved settings (None for amplitude PUBs)."""
    for key in ("phi", "alpha"):
        if key in built["meta"].get(label, {}):
            return len(built["meta"][label][key])
    return None


def _covers(spans: list, n_sets: int | None) -> bool:
    """True when the returned parts tile sets 0..n_sets exactly once.

    Checked against the settings rather than the plan, because a plan read back
    from job.json after a stopped submission lists only the jobs queued.
    """
    if not spans:
        return False
    if any(s is None for s in spans):
        return len(spans) == 1
    edge = 0
    for a, b in sorted(spans):
        if a != edge:
            return False
        edge = b
    return n_sets is not None and edge == n_sets


def save_results(out: str, built: dict, plan: list, results: list, job_ids: list) -> tuple:
    """Save every condition whose parts all succeeded, then write the report.

    Returns (report, complete). With a failed job the run is saved without the
    conditions that job held, and the manifest lists them and the job ids.
    """
    import cloud_run_analyze as cra

    labels, failed = complete_labels(built, plan, results)
    if not labels:
        print("Nothing saved: no condition has all its parts.", flush=True)
        return None, False
    keep = [(job, r) for job, r in zip(plan, results) if r is not None]
    sub = dict(built, labels=labels)
    pubs = assemble(sub, [[e for e in job if e[0] in labels] for job, _ in keep],
                    [_Subset(r, [j for j, e in enumerate(job) if e[0] in labels]) for job, r in keep])
    left_out = [lab for lab in built["labels"] if lab not in labels]
    missing = {"conditions": left_out, "jobs": [job_ids[i] for i in failed]} if left_out else None
    cra.save_job(out, sub, pubs, _JobIds([i for i, r in zip(job_ids, results) if r is not None]), missing=missing)
    return _save_report(out), not left_out


class _Subset:
    """The PUB results of one job at the given entry indices, in order."""

    def __init__(self, result, idx):
        self._items = [result[j] for j in idx]

    def __getitem__(self, j):
        return self._items[j]


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
    parser.add_argument("--qubits", type=int,
                        help="qubits in parallel (default 6; eig: cloud_run_eigenstate.EIG_QUBITS)")
    parser.add_argument("--layout", help="comma-separated physical qubits (default: the follow-up uses the "
                                         "qubits of the 15 October 2026 run; other jobs pick from calibration)")
    parser.add_argument("--shots", type=int, help="shots per PUB (default: 300 pilot, 400 main)")
    parser.add_argument("--out", help="run directory, e.g. data/cloudrun/2026-10-15_main")
    parser.add_argument("--submit", action="store_true", help="send the job (needs the author's go-ahead)")
    parser.add_argument("--account", help="saved account name (default: the default saved account)")
    parser.add_argument("--instance-of", help="use the default saved key with the instance saved under this name")
    parser.add_argument("--max-qpu-s", type=float,
                        help="refuse to submit if the estimate plus job overhead exceeds this many seconds")
    parser.add_argument("--max-job-s", type=float, nargs="?", const=MAX_JOB_S, default=None,
                        help=f"split conditions so every job estimates under this many seconds "
                             f"(default when passed: {MAX_JOB_S})")
    parser.add_argument("--retrieve", metavar="JOB_ID", help="save a job submitted earlier from --out")
    parser.add_argument("--refresh-usage", action="store_true",
                        help="record each job's final charged usage in --out's job.json (IBM finalizes it after the "
                             "job ends, so the value saved at completion can be 0); sends nothing")
    args = parser.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    import cloud_run_analyze as cra

    if args.instance_of:
        instance = QiskitRuntimeService.saved_accounts()[args.instance_of]["instance"]
        service = QiskitRuntimeService(name=args.account, instance=instance) if args.account else \
            QiskitRuntimeService(instance=instance)
    else:
        service = QiskitRuntimeService(name=args.account) if args.account else QiskitRuntimeService()
    if args.refresh_usage:
        if not args.out:
            parser.error("--refresh-usage needs --out")
        path = os.path.join(args.out, "job.json")
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        for meta in record["jobs"]:
            meta.update(_usage(service.job(meta["job_id"])))
        record["usage_refreshed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record["charged_total_s"] = sum(m["usage"] for m in record["jobs"] if isinstance(m.get("usage"), (int, float)))
        _write_record(path, record)
        print("charged per job:", [m.get("usage") for m in record["jobs"]], "total", record["charged_total_s"])
        return 0
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
        results = []
        for m in jobs_meta:
            try:
                results.append(service.job(m["job_id"]).result())
                m["retrieved"] = "done"
            except Exception as exc:  # keep going: save every condition whose parts all came back
                m["retrieved"] = f"failed: {type(exc).__name__}: {str(exc)[:1500]}"
                print("JOB NOT RETRIEVED:", m["job_id"], m["retrieved"], flush=True)
                results.append(None)
        if args.retrieve == "auto":
            _write_record(os.path.join(args.out, "job.json"), record)
        report, complete = save_results(args.out, built, plan, results, [m["job_id"] for m in jobs_meta])
        if report is not None:
            print(json.dumps({"saved": args.out, "flags": report["flags"]}, indent=1, default=str))
        return 0 if complete else 2

    from qiskit_ibm_runtime.executor_sampler import Sampler

    from cloud_run_circuits import build_all
    import cloud_run_eigenstate as eigenstate

    shots = args.shots or SHOTS[args.job]
    backend = service.backend(args.backend)
    import cloud_run_followup as followup

    layout = [int(x) for x in args.layout.split(",")] if args.layout else (
        followup.LAYOUT if args.job == "followup" else (followup.FEZ_LAYOUT if args.job == "fezrepeat" else (
            list(eigenstate.EIG_REPEAT_LAYOUT) if args.job == "eigrepeat" else None)))
    eig = args.job == "eig"
    # eig keeps the October fez positives as controls and excludes the other
    # October fez qubits, so the picker cannot return them as fresh.
    built = build_all(backend, n_qubits=args.qubits or (eigenstate.EIG_QUBITS if eig or args.job == "eigrepeat" else 6),
                      shots=shots,
                      include=eigenstate.EIG_CONTROLS if eig and not layout else (),
                      exclude=eigenstate.EIG_EXCLUDE if eig and not layout else (),
                      pilot=args.job == "pilot",
                      followup_job=args.job == "followup", layout=layout,
                      conditions={"fezmini": followup.FEZ_MINI_CONDITIONS,
                                  "fezrepeat": followup.FEZ_REPEAT_CONDITIONS,
                                  "eig": eigenstate.EIG_CONDITIONS,
                                  "eigrepeat": eigenstate.EIG_REPEAT_CONDITIONS}.get(args.job),
                      amplitude=args.job != "fezrepeat")
    rep = built["report"]
    summary = {"backend": backend.name, "job": args.job, "shots": shots, "layout": rep["layout"],
               "readout_s": rep["readout_s"], "estimated_qpu_s": rep["estimated_qpu_s_total"],
               "calibration": rep.get("calibration"),
               "conditions": {k: {x: rep[k][x] for x in ("sets", "executions", "estimated_qpu_s")}
                              for k in built["labels"] if k in rep}}
    single_job = args.job in ("pilot", "fezmini", "fezrepeat")
    plan = plan_jobs(built, single_job, args.max_job_s)
    estimates = job_estimates(plan, built)
    n_jobs = len(plan)
    if args.max_job_s is not None:
        over = [i for i, e in enumerate(estimates) if e > args.max_job_s]
        if over:
            print(f"Not submitted: jobs {over} estimate "
                  f"{[estimates[i] for i in over]} s, over the --max-job-s {args.max_job_s} s cap.")
            return 5
    summary["jobs"] = [{"entries": [e[0] for e in job], "estimated_qpu_s": e}
                       for job, e in zip(plan, estimates)]
    print(json.dumps(summary, indent=1), flush=True)
    total = rep["estimated_qpu_s_total"] + JOB_OVERHEAD_S * n_jobs
    print(f"estimate with overhead: {total:.1f} s in {n_jobs} job(s); "
          f"per-job estimates {estimates}", flush=True)

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

    record = {"backend": backend.name, "job": args.job, "shots": shots, "account": args.account or "default",
              "instance_of": args.instance_of,
              "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "estimate": summary, "jobs": []}
    prior = os.path.join(args.out, "job.json")
    if os.path.exists(prior):
        with open(prior, encoding="utf-8") as fh:
            old = json.load(fh)
        record["earlier_attempts"] = old.get("earlier_attempts", []) + [
            {k: old[k] for k in ("job_id", "jobs", "submitted", "outcome") if k in old}]
    # job.json is rewritten after every queued job, so an interrupted submission
    # still leaves the ids of the jobs already queued for --retrieve auto.
    _write_record(prior, record)
    jobs = []
    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        for k, job_entries in enumerate(plan):
            try:
                job = sampler.run([e[1] for e in job_entries], shots=shots)
            except Exception as exc:  # stop queueing; the jobs already queued are still waited on and saved
                record["submit_error"] = {"job_index": k, "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
                _write_record(prior, record)
                print("SUBMISSION STOPPED at job", k, record["submit_error"]["error"], flush=True)
                break
            jobs.append(job)
            record["jobs"].append({"job_id": job.job_id(),
                                   "entries": [[e[0], list(e[2]) if e[2] else None] for e in job_entries]})
            _write_record(prior, record)
            print("submitted job", job.job_id(), [e[0] for e in job_entries], flush=True)

    t0 = time.time()
    results = []
    for job, meta in zip(jobs, record["jobs"]):
        try:
            results.append(job.result())
        except Exception as exc:  # report the service's message (size limits, payload, and so on)
            meta["outcome"] = f"failed: {type(exc).__name__}: {str(exc)[:1500]}"
            print("JOB FAILED:", job.job_id(), meta["outcome"], flush=True)
            results.append(None)
        else:
            meta["outcome"] = "done"
        meta.update(_usage(job))
        _write_record(prior, record)
    record["finished_after_s"] = round(time.time() - t0)
    _write_record(prior, record)
    print(f"results after {record['finished_after_s']} s", flush=True)
    # Jobs never queued count as failed, so their conditions are left out and listed.
    results += [None] * (len(plan) - len(jobs))
    ids = [j.job_id() for j in jobs] + ["not submitted"] * (len(plan) - len(jobs))
    report, complete = save_results(args.out, built, plan, results, ids)
    if report is not None:
        print(json.dumps({"saved": args.out, "flags": report["flags"]}, indent=1, default=str))
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
