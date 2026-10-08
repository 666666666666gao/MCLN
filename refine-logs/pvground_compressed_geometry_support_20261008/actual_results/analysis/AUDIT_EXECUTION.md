# Actual independent audit execution

All commands ran locally with cached offline NumPy. No source/evaluator was imported into the CPU recount, no torch was imported, and no neural/remote operation ran.

1. `uv run --offline --with numpy python -B -X utf8 C:/Users/gb/.codex/tmp/pvground_compressed_geometry_support_20261008/analysis/AUDIT_independent_cpu.py`
   - Actual process session: 79049; completion output chunk `b08837`; exit code **0**.
   - Script elapsed time: **23.732454299984965 seconds**.
   - Runtime: Python 3.13.12, NumPy 2.5.3.
   - Produced `AUDIT_independent_cpu.json` and `AUDIT_machine_reviewed_files.json`.
   - Earlier two executions stopped on auditor-imposed IoU upper-range assertions. The observed excursions and revisions are recorded in `AUDIT_initial_execution.json`; no experiment data were modified. Final checks retain raw values and explicitly count range excursions.
2. `uv run --offline --with numpy python -B -X utf8 C:/Users/gb/.codex/tmp/pvground_compressed_geometry_support_20261008/analysis/AUDIT_source_checks.py`
   - Actual completion output chunk `3a9b2c`; exit code **0**; tool wall time **2.3492979 seconds**.
   - Produced `AUDIT_source_checks.json` and `AUDIT_source_reviewed_files.json`.

An exploratory AST comparison also stopped when it attempted to find `_scene_graph_parse` as a class method in the older unused model-tree dataset. The corrected inspection explicitly used the actual imported dataset's method and compared only methods present in both files. No result was inferred from the failed lookup.

Deterministic scope: exact archived inventory; every initial/final NPZ array; 14,604,288 float64 candidate Box IoUs (and a separate float32 comparison); saved top-16/32/64/256 oracle labels; every saved training record; complete row identities; source/hash binding and archived runtime markers. Not performed: raw dataset or raw Mask rebuild, checkpoint deserialization, fresh NN inference, CPU/GPU model construction, remote inspection, retention, deletion or publication.
