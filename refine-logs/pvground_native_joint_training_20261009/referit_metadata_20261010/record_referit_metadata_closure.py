"""Close the bounded metadata check without promoting loader or model claims."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'referit_metadata_20261010'
audit = json.loads((data / 'actual_review/EXPERIMENT_AUDIT.json').read_bytes())
seal = json.loads((data / 'actual_review/ACTUAL_REVIEW_SEAL.json').read_bytes())
assert audit['verdict'] == 'WARN' and audit['bounded_check_verdict'] == 'PASS'
assert audit['blocking_issue_count'] == 0 and seal['audited_inputs_unchanged']
for name, digest in seal['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
for row in seal['output_artifacts']:
    raw = Path(row['path']).read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
result = json.loads((data / 'METADATA_RESULT.json').read_bytes())
transport = json.loads((data / 'TRANSPORT_EXIT.json').read_bytes())
assert transport['exit_code'] == 0 and result == json.loads((data / 'RAW_STDOUT.json').read_bytes())
assert [row['metadata_eligible_expressions'] for row in result['records']] == [32919, 7899, 65846, 17726]
assert all(not row['missing_superpoint_paths'] and not row['missing_groupfree_proposal_paths'] for row in result['records'])
clock_delta = (datetime.datetime.fromisoformat(result['observed_cst']) -
               datetime.datetime.fromisoformat(transport['finished_cst'])).total_seconds()
closure = dict(status='BOUNDED_REAL_REFERIT_METADATA_CHECK_CLOSED',
    closed_cst=datetime.datetime.now().astimezone().isoformat(),
    audit_verdict=audit['verdict'], bounded_check_verdict='PASS', blocking_issue_count=0,
    result_sha256=hashlib.sha256((data / 'METADATA_RESULT.json').read_bytes()).hexdigest(),
    audit_sha256=hashlib.sha256((data / 'actual_review/EXPERIMENT_AUDIT.json').read_bytes()).hexdigest(),
    seal_sha256=hashlib.sha256((data / 'actual_review/ACTUAL_REVIEW_SEAL.json').read_bytes()).hexdigest(),
    metadata_counts=[32919, 7899, 65846, 17726], all_required_selected_scene_input_files_exist=True,
    actual_loader_length_verified=False, scans_pickle_membership_verified=False,
    proposal_array_contents_verified=False, model_forward_verified=False,
    remote_timestamp=result['observed_cst'], local_transport_finished=transport['finished_cst'],
    remote_minus_local_finished_seconds=clock_delta, clocks_synchronized=False,
    normal_training_queries=0, active_source_changed=False, REC_accuracy=None,
    GPU_or_training_admission=False, reviewer_identity='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', full_goal_complete=False)
target = root / 'ACTUAL_REFERIT_METADATA_CLOSURE.json'
assert not target.exists()
target.write_text(json.dumps(closure, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
notes = root / 'HANDOFF_REFERIT_METADATA_20261010.md'
text = notes.read_text(encoding='utf-8')
before = '源代码复核已完成，无阻断；真实输出复核待单独关闭。'
assert before in text
text = text.replace(before,
    '源代码复核PASS，无阻断；真实输出复核已关闭，总结为WARN、限定元数据检查PASS、0阻断。'
    '唯一提醒是服务器观察时间比本地传输结束时间晚5.239782秒，两端时钟同步未获确认；原始时间各自保留，不据此推断跨主机精确先后关系。')
notes.write_text(text, encoding='utf-8')
print(json.dumps(dict(status=closure['status'], metadata_counts=closure['metadata_counts'],
    bounded_check_verdict='PASS', current_training_queries=0, full_goal_complete=False)))
