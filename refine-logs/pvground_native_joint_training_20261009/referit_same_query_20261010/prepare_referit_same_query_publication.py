"""Prepare section143 only after the bounded native evaluator result is audited."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'referit_same_query_20261010'
result = json.loads((data / 'cpu_execution/SAME_QUERY_CPU_RESULT.json').read_bytes())
receipt = json.loads((data / 'cpu_execution/CPU_EXECUTION.json').read_bytes())
source = json.loads((data / 'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes())
actual = json.loads((data / 'actual_review/EXPERIMENT_AUDIT.json').read_bytes())
assert result['status'] == 'SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU_PASS'
assert result['formal_accuracy'] is None and not result['GPU_training_admission']
assert source['execution_scope'] == 'SOURCE_ONLY' and not source['blocking_findings']
assert actual['execution_scope'] == 'SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU'
assert actual['verdict'] in ('PASS', 'WARN') and not actual['blocking_findings']
for review in (source, actual):
    for path, digest in review['audited_input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
summary = dict(status='SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU_REVIEWED',
    prepared_cst=datetime.datetime.now().astimezone().isoformat(), result=result,
    execution_receipt=receipt,
    review=dict(source_verdict=source['verdict'], actual_verdict=actual['verdict'],
        execution_scope=actual['execution_scope'], blocking_findings=actual['blocking_findings'],
        non_blocking_findings=actual['non_blocking_findings'],
        actual_runtime_model='UNATTESTED', independence='same-family', acceptance='provisional'),
    best_retained_counts=[5677, 4920], formal_accuracy=None,
    normal_training_status_queries=0, active_training_source_changed=False,
    full_benchmark_evaluation_completed=False, GPU_training_admission=False,
    C_training_vs_filtered_deployment_Query_equivalence_verified=False,
    final_method_selection_pending=True, full_goal_complete=False)
(data / 'SAME_QUERY_CPU_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
names = ['referit_same_query_20261010/' + name for name in (
    'PLAN.md', 'HANDOFF_SAME_QUERY_CPU_20261010.md', 'SAME_QUERY_CPU_SUMMARY.json',
    'SOURCE_PREPARATION.json', 'EVALUATOR_CHANGE.diff', 'CHECK_SPEC.json',
    'BASE_WARM_SOURCE_HASHES.json', 'WARM_SOURCE_HASHES.json',
    'nr3d/init.json', 'sr3d/init.json', 'check_native_same_query_cpu.py',
    'prepare_cpu_check.py', 'run_authorized.py',
    'source_review/EXPERIMENT_CODE_REVIEW.md', 'source_review/EXPERIMENT_CODE_REVIEW.json',
    'actual_review/EXPERIMENT_AUDIT.md', 'actual_review/EXPERIMENT_AUDIT.json',
    'cpu_execution/CPU_EXECUTION.json', 'cpu_execution/CPU_STDOUT.txt',
    'cpu_execution/SAME_QUERY_CPU_RESULT.json', 'cpu_execution/TRANSPORT_EXIT.json')]
names += ['referit_same_query_20261010/' + path.relative_to(data).as_posix()
    for path in (data / 'source').rglob('*.py')]
names += ['referit_author_entry_20261010/' + name for name in (
    'PLAN.md', 'SOURCE_PREPARATION.json', 'ENTRY_CHANGE.diff',
    'source_review/EXPERIMENT_CODE_REVIEW.md', 'source_review/EXPERIMENT_CODE_REVIEW.json')]
names += ['prepare_referit_author_entry.py', 'prepare_referit_same_query.py',
    'prepare_referit_same_query_publication.py', 'prepare_referit_same_query_handoff.py',
    'record_referit_same_query_publication.py']
assert all((root / name).is_file() for name in names)
assert not any('STDERR' in name or '.aris/' in name or 'RAW_' in name for name in names)
(root / 'REFERIT_SAME_QUERY_PUBLIC_FILE_LIST.json').write_text(json.dumps(names, indent=2)+'\n')
content = (root / 'publish_face_native_cpu_authorized.py').read_text(encoding='utf-8')
replacements = [
    ("'c_off_normal_publication.json'", "'referit_data_cpu_publication.json'"),
    ("prior['section'] == '20.376.140'", "prior['section'] == '20.376.142'"),
    ("'face_native_cpu_publication.json').exists()", "'referit_same_query_publication.json').exists()"),
    ('face_native_factory_cpu_20261010/FACE_NATIVE_CPU_SUMMARY.json', 'referit_same_query_20261010/SAME_QUERY_CPU_SUMMARY.json'),
    ("summary['review']['actual_verdict'] == 'PASS'", "summary['review']['actual_verdict'] in ('PASS','WARN')"),
    ("b'20.376.141' not in old", "b'20.376.143' not in old"),
    ('face_native_factory_cpu_20261010/HANDOFF_FACE_NATIVE_CPU_20261010.md', 'referit_same_query_20261010/HANDOFF_SAME_QUERY_CPU_20261010.md'),
    ('face_native_factory_cpu_20261010/', 'referit_same_query_20261010/'),
    ("/face_native_factory_cpu_20261010'", "/referit_same_query_20261010'"),
    ('FACE_NATIVE_CPU_PUBLIC', 'REFERIT_SAME_QUERY_PUBLIC'),
    ('publish_face_native_cpu_authorized.py', 'publish_referit_same_query_authorized.py'),
    ("doc.name+'.tmp_face_native_cpu'", "doc.name+'.tmp_referit_same_query'"),
    ('Record actual native face-factory CPU initialization evidence', 'Align native ReferIt box-mask reporting on the same selected Query'),
    ('FACE_NATIVE_CPU_ALL_', 'REFERIT_SAME_QUERY_ALL_'),
    ("section='20.376.141'", "section='20.376.143'"),
    ("integrity_verdict='PASS_ENGINEERING_CPU_SCOPE_ONLY'", "integrity_verdict=summary['review']['actual_verdict']"),
    ('face_native_cpu_local_commit.json', 'referit_same_query_local_commit.json'),
    ('face_training_launched=False', 'referit_training_launched=False'),
    ("(root / 'face_native_cpu_publication.json').write_text", "(root / 'referit_same_query_publication.json').write_text"),
]
for old, new in replacements:
    assert old in content, old
    content = content.replace(old, new)
ast.parse(content, feature_version=(3, 7))
(root / 'publish_referit_same_query_authorized.py').write_text(content, encoding='utf-8')
print(json.dumps(dict(status=summary['status'], review=actual['verdict'],
    public_files=len(names), full_goal_complete=False)))
