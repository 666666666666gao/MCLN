"""Reviewer-only stdlib inspection. Never imports or executes candidate sources."""
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path

F = Path(r"C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_residual_preparation_20261010")
V = F.parent
T = F / ".aris/traces/experiment-bridge/2026-10-10_run02"
P = Path(r"C:\Users\gb\.codex\skills")
whole = Path(r"C:\Users\gb\.codex\tmp\pvground_whole_mask_range_20261003\whole_mask_range.py")
env = Path(r"C:\Users\gb\.codex\tmp\pvground_cs_restart_20261002\env_spec.json")
inputs = [F / name for name in [
    "CPU_CHECK_PLAN.md", "run_face_residual_cpu_authorized.py",
    "face_residual_span_mixer.py", "check_face_residual_modules_cpu.py",
    "TRANSPORT_FIX_PREPARATION.json", "STATIC_CPU_ROOT_READ.json",
    "STATIC_CPU_ROOT_TRANSPORT_EXIT.json", "source_conditioned_init.json",
    "transport_attempt1/EXPERIMENT_CODE_REVIEW_R1.md",
    "transport_attempt1/EXPERIMENT_CODE_REVIEW_R1.json",
    "transport_attempt1/REVIEWED_LAUNCHER_ATTEMPT1.py",
    "transport_attempt1/RAW_STDERR.txt", "transport_attempt1/RAW_STDOUT.json",
    "transport_attempt1/TRANSPORT_EXIT.json",
    "cpu_execution/RAW_STDERR.txt", "cpu_execution/RAW_STDOUT.json",
    "cpu_execution/TRANSPORT_EXIT.json",
]]
inputs += [V / "NATIVE_SOURCE_PORT.json", V.parent / "SCP_TRANSPORT_WITNESS.json"]
inputs += [V / "source" / name for name in
           ["mask_support_corrector.py", "extremal_span_mixer.py", "native_mask_geometry.py"]]
inputs += [whole, env, P / "experiment-bridge/SKILL.md"]
inputs += [P / "shared-references" / name for name in
           ["local-codex-policy.md", "review-tracing.md", "output-versioning.md",
            "output-manifest.md", "output-language.md"]]
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read_json(path):
    return json.loads(path.read_bytes())

r1 = read_json(F / "transport_attempt1/EXPERIMENT_CODE_REVIEW_R1.json")
prep = read_json(F / "TRANSPORT_FIX_PREPARATION.json")
port = read_json(V / "NATIVE_SOURCE_PORT.json")
static = read_json(F / "STATIC_CPU_ROOT_READ.json")
old_path = F / "transport_attempt1/REVIEWED_LAUNCHER_ATTEMPT1.py"
current_path = F / "run_face_residual_cpu_authorized.py"
old = old_path.read_text(encoding="utf-8")
current = current_path.read_text(encoding="utf-8")
expected = old.replace(
    "attempt = root/'cpu_execution'",
    "static_root = json.loads((root/'STATIC_CPU_ROOT_READ.json').read_bytes())\n"
    "assert static_root['root'] == '/root/autodl-tmp/pvground_face_residual_cpu_20261010'\n"
    "assert static_root['exists'] is False and static_root['CPU_execution_receipt_exists'] is False\n"
    "attempt = root/'cpu_execution_attempt2'"
).replace(
    "SSH_ASKPASS=str(v.parent/'NativeSshAskPass.exe')",
    "SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe'"
).replace(
    "'UserKnownHostsFile='+str(v.parent/'native_scp_known_hosts')",
    "'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts'"
)
diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), current.splitlines(keepends=True),
                                   fromfile="ARCHIVED_ATTEMPT1", tofile="CURRENT_ATTEMPT2"))
(T / "LAUNCHER_DELTA.diff").write_text(diff, encoding="utf-8")
sources = [path for path in inputs if path.suffix == ".py"]
parsed = []
for path in sources:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parsed.append(str(path))
current_tree = ast.parse(current)
embedded = next(node.value.value for node in current_tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "code" for target in node.targets))
ast.parse(embedded, feature_version=(3, 7))
old_tree = ast.parse(old)
old_embedded = next(node.value.value for node in old_tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "code" for target in node.targets))
env_spec = read_json(env)
canonical_env = hashlib.sha256(json.dumps(env_spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
source_matches = {name: digest(V / "source" / name) == port["files"][name]["sha256"]
                  for name in ["mask_support_corrector.py", "extremal_span_mixer.py", "native_mask_geometry.py"]}
r1_unchanged = {name: digest(F / name) == r1["audited_input_hashes"][str(F / name)]
                for name in ["CPU_CHECK_PLAN.md", "face_residual_span_mixer.py",
                             "check_face_residual_modules_cpu.py", "source_conditioned_init.json"]}
archives = {name: (F / "transport_attempt1" / name).read_bytes() == (F / "cpu_execution" / name).read_bytes()
            for name in ["RAW_STDERR.txt", "RAW_STDOUT.json", "TRANSPORT_EXIT.json"]}
stderr = (F / "cpu_execution/RAW_STDERR.txt").read_bytes()
unchanged_r1_reports = {
    ext: (F / ("transport_attempt1/EXPERIMENT_CODE_REVIEW_R1." + ext)).read_bytes()
    == (F / ("source_review/EXPERIMENT_CODE_REVIEW_20261010_074450." + ext)).read_bytes()
    for ext in ["md", "json"]
}
new_params = 332 * 64 + 64 + 64 * 32 + 32 + 32 + 1
prior_params = 2 * (288 * 32 + 32) + (41 * 32 + 32) + (32 * 32 + 32) + (105 * 64 + 64) + (64 * 32 + 32) + 32 + 1
checks = {
    "only_requested_launcher_changes": current == expected,
    "embedded_remote_program_unchanged": embedded == old_embedded,
    "old_launcher_matches_R1_historical_digest": digest(old_path) == r1["audited_input_hashes"][str(current_path)],
    "current_launcher_differs_from_R1_historical_digest": digest(current_path) != r1["audited_input_hashes"][str(current_path)],
    "old_launcher_matches_fix_preparation": digest(old_path) == prep["old_launcher_sha256"],
    "current_launcher_matches_fix_preparation": digest(current_path) == prep["new_launcher_sha256"],
    "unchanged_reviewed_inputs_match_R1": all(r1_unchanged.values()),
    "archived_failure_bytes_match_original": all(archives.values()),
    "R1_archived_reports_match_timestamped_original": all(unchanged_r1_reports.values()),
    "failed_transport_exit_255": read_json(F / "cpu_execution/TRANSPORT_EXIT.json")["exit_code"] == 255,
    "failed_transport_stdout_empty": len((F / "cpu_execution/RAW_STDOUT.json").read_bytes()) == 0,
    "failed_transport_stderr_133_bytes": len(stderr) == 133,
    "failed_transport_has_host_key_verification_error": b"Host key verification failed" in stderr,
    "static_transport_exit_0": read_json(F / "STATIC_CPU_ROOT_TRANSPORT_EXIT.json")["exit_code"] == 0,
    "static_root_target_exact": static["root"] == "/root/autodl-tmp/pvground_face_residual_cpu_20261010",
    "static_root_and_receipt_absent": static["exists"] is False and static["CPU_execution_receipt_exists"] is False,
    "static_observation_no_training_queries": static["training_status_queries"] == 0,
    "static_record_marks_host_key_route_verified": static["unchanged_host_key_route_verified"] is True,
    "local_attempt2_absent": not (F / "cpu_execution_attempt2").exists(),
    "source_sha_matches_port": all(source_matches.values()),
    "whole_mask_range_sha_matches_port": digest(whole) == port["files"]["whole_mask_range.py"]["sha256"],
    "env_canonical_hash_matches_port": canonical_env == port["env_spec_sha256"],
    "env_has_PYTHONPATH": "PYTHONPATH" in env_spec["env"],
    "new_parameter_count_23425": new_params == 23425,
    "prior_parameter_count_29793": prior_params == 29793,
    "combined_parameter_count_53218": new_params + prior_params == 53218,
    "canonical_review_outputs_excluded_from_inputs": str(F / "source_review/EXPERIMENT_CODE_REVIEW.json") not in hashes
       and str(F / "source_review/EXPERIMENT_CODE_REVIEW.md") not in hashes,
}
out = {
    "time_cst": datetime.datetime.now().astimezone().isoformat(),
    "checks": checks, "all_static_checks_pass": all(checks.values()),
    "audited_input_hashes": hashes, "ast_parsed": parsed,
    "embedded_remote_python37_ast": "PASS",
    "unchanged_inputs_R1_detail": r1_unchanged,
    "source_matches_port_detail": source_matches,
    "original_and_archived_failure_same_bytes": archives,
    "R1_archived_reports_unchanged": unchanged_r1_reports,
    "old_launcher_sha256": digest(old_path),
    "current_launcher_sha256": digest(current_path),
    "historical_mutable_launcher_path_not_revalidated_as_unchanged": str(current_path),
    "source_root_resolves_locally_to": str(F.resolve()),
    "static_CPU_root_observation": static,
    "stderr_review": {"bytes": len(stderr), "host_key_error_present": b"Host key verification failed" in stderr,
                      "raw_stderr_quoted_or_copied_to_trace": False},
    "env_spec_review": {"canonical_sha256": canonical_env, "PYTHONPATH_present": "PYTHONPATH" in env_spec["env"]},
    "reviewer_executed_candidate": False, "reviewer_SSH_calls": 0,
    "reviewer_GPU_calls": 0, "reviewer_training_status_queries": 0,
}
(T / "STATIC_REVIEW_CHECKS_R2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in out.items() if key != "audited_input_hashes"}, indent=2, ensure_ascii=False))
print(diff)
assert all(checks.values())

