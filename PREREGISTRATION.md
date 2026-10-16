# Pre-registration record

The designs and analysis code were developed in the author's private repository and committed there before each run's data existed. The commit hashes below refer to that private repository, and the author can show them on request. Job creation times come from IBM's job metadata in each run's `job.json`.

| Design fixed | Private commit, time (UTC) | First IBM job created (UTC) |
|---|---|---|
| Pilot and main run (`notes/CLOUD_RUN_DESIGN.md`, analysis code) | 85a36c6, 2026-10-15 18:43:44 | pilot 18:44:32; main 19:28:34 (rejected by IBM), rerun as a batch 19:55:28 |
| Leak-tagged follow-up (`notes/CLOUD_RUN_FOLLOWUP_DESIGN.md`, Sections 1 to 8) | 992b41a, 2026-10-15 22:07:52 | 22:11:45 |
| ibm_fez add-on (Section 9) | d501525, 2026-10-16 03:08:22 | 03:08:35 |
| Paid ibm_fez repeat (Section 10) | 54ddcf2, 2026-10-16 03:14:07; adjusted before data in b5bd8ec, 03:31:48 | 03:31:55 (cancelled at the cost limit), rerun 03:37:29 |

Analyses chosen after seeing the data are labeled exploratory: `analysis/cloud_run_leakage.py`, Sections 4 and 6 of `notes/CLOUD_RUN_RESULTS.md`, and the run-length counts in Section 4.
