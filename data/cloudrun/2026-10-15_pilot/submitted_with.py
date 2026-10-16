"""Submit the cloud-run pilot to ibm_kingston (author's go-ahead, 15 October 2026) and save it."""
import json
import os
import pickle
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/o/AustinUT/Research/TestingSuperDet/analysis")
from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: E402
from qiskit_ibm_runtime.executor_sampler import Sampler  # noqa: E402
from cloud_run_circuits import build_all  # noqa: E402
from cloud_run_analyze import save_job  # noqa: E402

OUT = os.path.expanduser("~/cloudrun/pilot")
SHOTS = 300
os.makedirs(OUT, exist_ok=True)

service = QiskitRuntimeService()
backend = service.backend("ibm_kingston")
built = build_all(backend, n_qubits=6, shots=SHOTS, pilot=True)
with open(os.path.join(OUT, "built.pkl"), "wb") as fh:
    pickle.dump({k: built[k] for k in ("labels", "meta", "report")}, fh)
print("layout", built["report"]["layout"], "estimated QPU s", built["report"]["estimated_qpu_s_total"], flush=True)

sampler = Sampler(mode=backend)
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False
job = sampler.run(built["pubs"], shots=SHOTS)
job_id = job.job_id()
with open(os.path.join(OUT, "job.json"), "w", encoding="utf-8") as fh:
    json.dump({"job_id": job_id, "backend": backend.name, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh)
print("submitted job", job_id, flush=True)

t0 = time.time()
try:
    result = job.result()
except Exception as exc:  # report the service's message (size limits, etc.)
    print("JOB FAILED:", type(exc).__name__, str(exc)[:2000], flush=True)
    try:
        print("status", job.status(), flush=True)
    except Exception:
        pass
    sys.exit(2)
print(f"result after {time.time() - t0:.0f} s; status {job.status()}", flush=True)
manifest = save_job(OUT, built, result, job)
print("saved", json.dumps(manifest["chains"]), flush=True)
for attr in ("usage", "metrics"):
    try:
        val = getattr(job, attr)()
        print(attr, json.dumps(val, default=str)[:1500], flush=True)
    except Exception as exc:
        print(attr, "not available:", type(exc).__name__, flush=True)
