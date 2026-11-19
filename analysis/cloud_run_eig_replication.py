#!/usr/bin/env python3
"""Apply the E0R replication rules (notes/CLOUD_RUN_NEXT_DESIGN.md Section 2H) to the saved readings.

Inputs are the outputs of analysis/cloud_run_eig_drift.py on the 19 November
2026 E0 run (exploratory there) and on its repeat E0R (the pre-registered
primary reading). The rules were fixed in Section 2H before E0R's data; this
script, written after the data, only applies them:

  3. per qubit, against E0's in-chain reading: replicated when E0R reads memory
     with the same sign; contradicted when it reads the opposite sign, or reads
     consistent while its 3-sigma interval excludes the E0 value; otherwise
     inconclusive.
  4. the first-readout effect: de1 (in-chain minus AS e1) beyond 3 standard
     errors below zero replicates it, beyond 3 above zero contradicts it.
  6. drift between runs: the in-chain contrast deviations of E0 and E0R
     differing by more than 3 combined standard errors.

All on the clean mask, as in the readings themselves.

  python3 analysis/cloud_run_eig_replication.py \\
      data/cloudrun/2026-11-19_eig/drift_exploratory.json \\
      data/cloudrun/2026-11-19_eigrepeat/inchain_reading.json \\
      --output data/cloudrun/2026-11-19_eigrepeat/replication.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

K = 3.0


def sign(reading: str) -> int:
    return -1 if "positive" in reading else (1 if "negative" in reading else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("e0")
    ap.add_argument("e0r")
    ap.add_argument("--output")
    args = ap.parse_args()
    first = json.load(open(args.e0, encoding="utf-8"))["qubits"]
    rep = json.load(open(args.e0r, encoding="utf-8"))["qubits"]
    out = {"note": "Rules of notes/CLOUD_RUN_NEXT_DESIGN.md Section 2H, fixed before the E0R data and applied "
                   "here to the saved in-chain readings (clean mask). E0's reading is exploratory; E0R's is the "
                   "pre-registered primary.", "qubits": {}}
    for q in rep:
        a, b = first[q], rep[q]
        sa, sb = sign(a["reading_in_chain_null"]), sign(b["reading_in_chain_null"])
        va, ea = a["clean"]["contrast_deviation_in_chain_null"]
        vb, eb = b["clean"]["contrast_deviation_in_chain_null"]
        if sb != 0:
            rule3 = "replicated" if sb == sa else "contradicted"
        elif abs(va - vb) > K * eb:
            rule3 = "contradicted"
        else:
            rule3 = "inconclusive"
        d1, s1 = b["clean"]["de1"]
        rule4 = "replicated" if d1 < -K * s1 else ("contradicted" if d1 > K * s1 else "inconclusive")
        dz = (vb - va) / math.hypot(ea, eb)
        out["qubits"][q] = {
            "e0_in_chain": {"reading": a["reading_in_chain_null"], "deviation": [va, ea]},
            "e0r_in_chain": {"reading": b["reading_in_chain_null"], "deviation": [vb, eb],
                             "z_clean": b["clean"]["z_in_chain_null"], "z_clean_next": b["clean_next"]["z_in_chain_null"]},
            "rule3_reading": rule3,
            "rule4_first_readout": {"de1": [d1, s1], "verdict": rule4},
            "rule6_drift": {"z_difference": dz, "drifted": abs(dz) > K},
            "e0r_first_readout_reading": b["pre_registered_reading"],
        }
        print(q, rule3, f"| de1 {d1 * 100:+.2f}% {rule4} | drift z {dz:+.2f}")
    counts = {}
    for v in out["qubits"].values():
        counts[v["rule3_reading"]] = counts.get(v["rule3_reading"], 0) + 1
    out["summary"] = {"rule3": counts,
                      "rule4_replicated": sum(v["rule4_first_readout"]["verdict"] == "replicated" for v in out["qubits"].values()),
                      "rule6_drifted": [q for q, v in out["qubits"].items() if v["rule6_drift"]["drifted"]]}
    print(out["summary"])
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
