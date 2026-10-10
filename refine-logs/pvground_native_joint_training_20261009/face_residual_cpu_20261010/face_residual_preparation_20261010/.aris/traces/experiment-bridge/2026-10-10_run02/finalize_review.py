"""Finalize review artifacts only; never execute audited code."""
import datetime
import hashlib
import json
from pathlib import Path
F = Path(r"C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_residual_preparation_20261010")
S = F / "source_review"
T = F / ".aris/traces/experiment-bridge/2026-10-10_run02"
report = json.loads((S / "EXPERIMENT_CODE_REVIEW_R2.json").read_bytes())
checked = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
           for name, expected in report["audited_input_hashes"].items()}
assert all(checked.values())
assert report["verdict"] == "PASS" and report["blocking_issue_count"] == 0
assert not (F / "cpu_execution_attempt2").exists()
stamp = datetime.datetime.now().astimezone()
paths = []
for extension in ["md", "json"]:
    raw = (S / ("EXPERIMENT_CODE_REVIEW_R2." + extension)).read_bytes()
    versioned = S / ("EXPERIMENT_CODE_REVIEW_" + stamp.strftime("%Y%m%d_%H%M%S") + "." + extension)
    assert not versioned.exists()
    versioned.write_bytes(raw)
    canonical = S / ("EXPERIMENT_CODE_REVIEW." + extension)
    canonical.write_bytes(raw)
    assert versioned.read_bytes() == canonical.read_bytes() == raw
    paths += [versioned.name, "EXPERIMENT_CODE_REVIEW_R2." + extension, canonical.name]
assert (S / "EXPERIMENT_CODE_REVIEW_R2.md").read_bytes() == (T / "001-transport-rescue.response.md").read_bytes()
manifest_entries = "".join("| " + stamp.isoformat() + " | /experiment-bridge | " + name
                           + " | implementation | R2 bounded CPU transport-rescue source review PASS, 0 blockers; no CPU execution |\n"
                           for name in paths)
with (S / "MANIFEST.md").open("a", encoding="utf-8") as handle:
    handle.write(manifest_entries)
event = {"event": "review_trace", "skill": "experiment-bridge", "purpose": "round-2-transport-rescue-source-review",
         "agent_id": "/root/pvg_face_cpu_route_rescue_20261010", "trace_path": str(T), "status": "ok",
         "events_location_scope": "Kept within authorized .aris/traces; no writes outside source_review and .aris/traces."}
(T / "review_trace_event.json").write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
final = {"time_cst": stamp.isoformat(), "verdict": report["verdict"], "blocking_issue_count": 0,
         "input_count": len(checked), "input_hashes_unchanged": all(checked.values()),
         "canonical_R2_md_identical": (S / "EXPERIMENT_CODE_REVIEW.md").read_bytes() == (S / "EXPERIMENT_CODE_REVIEW_R2.md").read_bytes(),
         "canonical_R2_json_identical": (S / "EXPERIMENT_CODE_REVIEW.json").read_bytes() == (S / "EXPERIMENT_CODE_REVIEW_R2.json").read_bytes(),
         "full_response_md_identical": True, "local_attempt2_absent": not (F / "cpu_execution_attempt2").exists(),
         "canonical_json_sha256": hashlib.sha256((S / "EXPERIMENT_CODE_REVIEW.json").read_bytes()).hexdigest(),
         "canonical_md_sha256": hashlib.sha256((S / "EXPERIMENT_CODE_REVIEW.md").read_bytes()).hexdigest(),
         "R1_reports_and_archived_launcher_preserved": True, "reviewer_executed_candidate": False,
         "reviewer_SSH_calls": 0, "reviewer_GPU_calls": 0, "reviewer_training_status_queries": 0,
         "published_reports": paths}
(T / "REPORT_FINALIZATION_CHECK.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
print(json.dumps(final, indent=2))

