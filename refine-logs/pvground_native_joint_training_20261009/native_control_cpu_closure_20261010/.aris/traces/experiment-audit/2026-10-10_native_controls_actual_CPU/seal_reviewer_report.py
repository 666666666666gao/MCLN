"""Finalize local audit metadata and unsigned SHA256 seal; no project code execution."""
import datetime
import hashlib
import json
from pathlib import Path

trace = Path(__file__).resolve().parent
root = trace.parents[3]
review = root / 'native_direct_controls_20261010/actual_CPU_review'

def digest(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()

report_path = review / 'EXPERIMENT_AUDIT.json'
report = json.loads(report_path.read_bytes())
assert report['verdict'] == 'WARN'
assert report['blocking_issue_count'] == 0
assert report['execution_scope'] == 'CLOSED_CPU_SYNTHETIC_MODULE_ENGINEERING_NOT_PV_OR_ACCURACY'
assert report['actual_reviewer_model'] == report['actual_reviewer_reasoning'] == 'UNATTESTED'
assert report['review_independence'] == 'same-family' and report['acceptance_status'] == 'provisional'
assert report['gpu_admission'] is False and report['formal_accuracy_approved'] is False
assert report['audited_input_count'] == len(report['audited_input_hashes']) == 31
assert report['hash_only_prior_source_input_count'] == len(report['hash_only_prior_source_inputs']) == 37
assert (review / 'EXPERIMENT_AUDIT.md').read_bytes() == (trace / '001-native-controls-CPU-review.response.md').read_bytes()

all_hashes = dict(report['audited_input_hashes'])
all_hashes.update(report['hash_only_prior_source_inputs'])
for path, expected in all_hashes.items():
    assert digest(path) == expected, path
instruction_hashes = dict(report['instruction_input_hashes'])
for path, expected in instruction_hashes.items():
    assert digest(path) == expected, path
executor_metadata_path = trace / 'executor_request_metadata.json'
instruction_hashes[str(executor_metadata_path)] = digest(executor_metadata_path)
executor_metadata = json.loads(executor_metadata_path.read_bytes())
assert executor_metadata['actual_model_identity'] == 'UNATTESTED'
assert executor_metadata['requested_reviewer'] == 'gpt-6-astra'
assert executor_metadata['requested_reasoning'] == 'max'

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
inventory = dict(
    generated_at=now,
    semantically_reviewed_input_count=31,
    semantically_reviewed_input_hashes=report['audited_input_hashes'],
    hash_only_continuity_input_count=37,
    hash_only_continuity_input_hashes=report['hash_only_prior_source_inputs'],
    instruction_and_request_hashes=instruction_hashes,
    all_input_hashes_match_initial_or_bound_hashes=True,
    private_bootstrap_contents_or_snapshots_included=False,
)
(trace / 'input_hashes.final.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
artifacts = {}
for path in sorted(trace.iterdir()):
    if path.is_file():
        artifacts[str(path)] = digest(path)
for path in (review / 'EXPERIMENT_AUDIT.md', report_path):
    artifacts[str(path)] = digest(path)
seal = dict(
    seal_type='SHA256_CLOSED_CPU_REVIEW_SNAPSHOT_NOT_SIGNED',
    generated_at=now,
    verdict=report['verdict'],
    blocking_issue_count=0,
    execution_scope=report['execution_scope'],
    required_input_count=30,
    supplemental_semantic_input_count=1,
    audited_input_count=31,
    hash_only_continuity_input_count=37,
    inputs_unchanged_on_final_read=True,
    audited_input_hashes=report['audited_input_hashes'],
    hash_only_continuity_input_hashes=report['hash_only_prior_source_inputs'],
    instruction_input_hashes=instruction_hashes,
    artifact_hashes=artifacts,
    bounded_cpu_engineering_evidence_supported=True,
    formal_result_approval=False,
    gpu_admission=False,
    control_training_admission=False,
    requested_reviewer_model='gpt-6-astra',
    requested_reviewer_reasoning='max',
    actual_reviewer_model='UNATTESTED',
    actual_reviewer_reasoning='UNATTESTED',
    review_independence='same-family',
    acceptance_status='provisional',
    task_name=report['task_name'],
    agent_id='UNATTESTED',
    context_disclosure=report['additional_context_disclosure'],
    reviewer_neural_or_remote_execution=False,
    current_training_status_queried=False,
    report_and_raw_response_byte_identical=True,
)
seal_path = review / 'SEAL.json'
assert not seal_path.exists()
seal_path.write_text(json.dumps(seal, indent=2) + '\n', encoding='utf-8')
loaded_seal = json.loads(seal_path.read_bytes())
for path, expected in loaded_seal['artifact_hashes'].items():
    assert digest(path) == expected, path
print(json.dumps(dict(
    status='AUDIT_REPORT_AND_UNSIGNED_SEAL_VERIFIED',
    verdict=loaded_seal['verdict'],
    blocking_issue_count=0,
    audited_inputs=31,
    hash_only_inputs=37,
    artifacts=len(loaded_seal['artifact_hashes']),
    report_json_sha256=digest(report_path),
    report_md_sha256=digest(review / 'EXPERIMENT_AUDIT.md'),
    seal_sha256=digest(seal_path),
    seal_path=str(seal_path),
)))
