"""Prepare bounded section141 from completed factory receipts; no model or job query."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
face = root / 'face_native_factory_cpu_20261010'
result = json.loads((face / 'cpu_execution/NATIVE_FACE_FACTORY_CPU_RESULT.json').read_bytes())
receipt = json.loads((face / 'cpu_execution/CPU_EXECUTION.json').read_bytes())
source = json.loads((face / 'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes())
actual = json.loads((face / 'actual_review/EXPERIMENT_AUDIT.json').read_bytes())
assert result['status'] == 'ACTUAL_NATIVE_FACE_FACTORY_CPU_INITIALIZATION_PASS'
assert result['initial_complete1301_states_equal_between_modes']
assert result['formal_accuracy'] is None and result['GPU_calls'] == 0
assert source['execution_scope'] == 'SOURCE_ONLY' and not source['blocking_findings']
assert actual['execution_scope'] == 'ACTUAL_NATIVE_FACE_FACTORY_CPU_INITIALIZATION'
assert actual['verdict'] == 'PASS' and not actual['blocking_findings']
summary = dict(
    status='ACTUAL_NATIVE_FACE_FACTORY_CPU_INITIALIZATION_REVIEWED_PASS',
    prepared_cst=datetime.datetime.now().astimezone().isoformat(),
    result=result, execution_receipt=receipt,
    review=dict(source_verdict=source['verdict'], actual_verdict=actual['verdict'],
        execution_scope=actual['execution_scope'], blocking_findings=[],
        actual_runtime_model='UNATTESTED', independence='same-family', acceptance='provisional'),
    best_retained_counts=[5677,4920], formal_accuracy=None,
    normal_training_status_queries=0, active_training_source_changed=False,
    GPU_training_admission=False, full_goal_complete=False)
(face / 'FACE_NATIVE_CPU_SUMMARY.json').write_text(
    json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
names = ['face_native_factory_cpu_20261010/'+name for name in (
    'HANDOFF_FACE_NATIVE_CPU_20261010.md','FACE_NATIVE_CPU_SUMMARY.json','PLAN.md',
    'check_native_face_factory_cpu.py','prepare_source.py','run_native_face_factory_cpu_authorized.py',
    'source/train_dist_mod.py','source/face_residual_span_mixer.py','source/native_face_model_initialization.py',
    'source_conditioned.json','without_additional_source.json','NATIVE_SOURCE_PORT.json',
    'NORMAL_NATIVE_RUN_PROTOCOL.json','NORMAL_E0_IDENTITY.json',
    'source_review/EXPERIMENT_CODE_REVIEW.md','source_review/EXPERIMENT_CODE_REVIEW.json',
    'actual_review/EXPERIMENT_AUDIT.md','actual_review/EXPERIMENT_AUDIT.json',
    'cpu_execution/CPU_EXECUTION.json','cpu_execution/CPU_STDOUT.txt',
    'cpu_execution/NATIVE_FACE_FACTORY_CPU_RESULT.json','cpu_execution/TRANSPORT_EXIT.json')]
names += ['prepare_face_native_cpu_publication.py']
assert all((root / name).is_file() for name in names)
assert not any('STDERR' in name or 'private/' in name or 'RAW_' in name for name in names)
(root / 'FACE_NATIVE_CPU_PUBLIC_FILE_LIST.json').write_text(json.dumps(names,indent=2)+'\n')
content = (root / 'publish_c_off_normal_authorized.py').read_text(encoding='utf-8')
replacements = [
    ("'c_off_runtime_publication.json'", "'c_off_normal_publication.json'"),
    ("prior['section'] == '20.376.139'", "prior['section'] == '20.376.140'"),
    ("'c_off_normal_publication.json').exists()", "'face_native_cpu_publication.json').exists()"),
    ('normal_controls_20261010/C_OFF_NORMAL_SUMMARY.json','face_native_factory_cpu_20261010/FACE_NATIVE_CPU_SUMMARY.json'),
    ("assert not summary['normal_training_net_gain'] and summary['prescribed_best_epoch'] == 0\nassert summary['fixed_terminal_delta_from_E0'] == [-101, -432]",
     "assert summary['formal_accuracy'] is None and not summary['GPU_training_admission']\nassert summary['review']['actual_verdict'] == 'PASS' and not summary['review']['blocking_findings']"),
    ("b'20.376.140' not in old", "b'20.376.141' not in old"),
    ('normal_controls_20261010/HANDOFF_C_OFF_NORMAL_20261010.md','face_native_factory_cpu_20261010/HANDOFF_FACE_NATIVE_CPU_20261010.md'),
    ('c_off_normal_20261010/','face_native_factory_cpu_20261010/'),
    ("/c_off_normal_20261010'", "/face_native_factory_cpu_20261010'"),
    ('C_OFF_NORMAL_PUBLIC','FACE_NATIVE_CPU_PUBLIC'),
    ('publish_c_off_normal_authorized.py','publish_face_native_cpu_authorized.py'),
    ("doc.name+'.tmp_normal_terminal'", "doc.name+'.tmp_face_native_cpu'"),
    ('Record completed C-off preflight and ordinary joint training launch','Record actual native face-factory CPU initialization evidence'),
    ('C_OFF_NORMAL_ALL_','FACE_NATIVE_CPU_ALL_'),
    ("section='20.376.140'", "section='20.376.141'"),
    ("observed_normal_metrics=summary['metrics']", "reviewed_result=summary['result']"),
    ('terminal_result=summary','engineering_result=summary'),
    ('control_training_launched=True','c_off_control_training_launched=True, face_training_launched=False'),
    ('selected_epoch=0,','selected_epoch=0, selected_epoch_scope=\'retained_best_parent_not_new_training\','),
    ("integrity_verdict='WARN'", "integrity_verdict='PASS_ENGINEERING_CPU_SCOPE_ONLY'"),
    ('c_off_normal_local_commit.json','face_native_cpu_local_commit.json'),
    ("(root / 'c_off_normal_publication.json').write_text", "(root / 'face_native_cpu_publication.json').write_text"),
]
for old,new in replacements:
    assert old in content,old
    content=content.replace(old,new)
ast.parse(content,feature_version=(3,7))
(root / 'publish_face_native_cpu_authorized.py').write_text(content,encoding='utf-8')
print(json.dumps(dict(status=summary['status'],public_files=len(names),GPU_training_admission=False)))
