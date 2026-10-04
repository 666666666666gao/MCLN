"""Record actual completed source review and collected V2 runtime receipts."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
revision = local.parent / 'revision2'
review = json.loads((local / 'READBACK_FORMAL_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and review['execution_scope'] == 'SOURCE_ONLY'
assert not review['blocking_findings'] and len(review['reviewed_files']) == 42
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
wait = json.loads((revision / 'readback_preflight_wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0
assert not wait['terminal']['controller_alive']
intake = json.loads((revision / 'complete_preflight/INTAKE.json').read_bytes())
assert intake['downloaded_weight_files'] == 0 and len(intake['files']) == 28
proofs = {}
for arm in ('evidence_hidden', 'evidence_visible'):
    proof = json.loads((revision / 'complete_preflight' / arm / 'preflight.json').read_bytes())
    assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2
    assert proof['geometry_provider_and_g_states_exact'] and proof['zero_residual_native_semantic_exact']
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    proofs[arm] = dict(status='pass', optimizer_steps=2, batch_size=proof['batch_size'],
        use_geometry_evidence=proof['use_geometry_evidence'], serialization_bytes=proof['serialization_bytes'],
        parent_states_exact=True, zero_residual_native_semantic_exact=True, optimizer_restore_exact=True)
stamp = datetime.datetime.now().astimezone().isoformat()
runtime = dict(time_cst=stamp, actual_terminal_cst=wait['terminal']['status']['finished_cst'],
    status='ACTUAL_BOTH_ARMS_TWO_UPDATE_PASS', proofs=proofs, collected_text_files=28,
    downloaded_weight_files=0, formal_training_started=False, accuracy_result=False,
    closed_native_observer_session=24708, closed_collector_session=50701)
assert not (revision / 'ACTUAL_PREFLIGHT_SUMMARY.json').exists()
(revision / 'ACTUAL_PREFLIGHT_SUMMARY.json').write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
response = '''PASS — SOURCE_ONLY，无阻断项。

已保存 [审查报告](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/formal_draft/READBACK_FORMAL_SOURCE_REVIEW.md>) 和 [JSON](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/formal_draft/READBACK_FORMAL_SOURCE_REVIEW.json>)，包含 42 个文件的精确路径与 SHA256。

26 个 Python 文件语法检查通过；实际 CPU 复算历史 9508 行得到 5616/4506，阈值差异 0。没有远端操作、模型执行或源码修改。

结论为 same-family provisional；Astra/max 后端未获证实。正式启动仍须实际 V2 两臂预检通过，本审查不代表运行或精度成功。
'''
assert not (local / 'ACTUAL_FORMAL_SOURCE_REVIEW_RESPONSE.txt').exists()
(local / 'ACTUAL_FORMAL_SOURCE_REVIEW_RESPONSE.txt').write_text(response, encoding='utf-8')
record = dict(status='ACTUALLY_COMPLETED_SOURCE_ONLY', actual_task='/root/pvg_readback_formal_source_review',
    requested_model='gpt-6-astra', requested_reasoning_effort='max', model_effort_independently_attested=False,
    review_independence='same-family', acceptance_status='provisional', fork_turns='none',
    verdict='PASS', blocking_findings=0, reviewed_files=42, AST37_files=26, time_cst=stamp,
    response_sha256=hashlib.sha256(response.encode('utf-8')).hexdigest(),
    reviewer_runtime_executed=False, reviewer_accuracy_result=False,
    separate_executor_v2_preflight_pass=True, formal_training_started=False)
(local / 'FORMAL_SOURCE_REVIEW_CALL.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
trace = local / '.aris/traces/experiment-bridge/2026-10-04_readback_formal_source_run01'
assert trace.exists()
(trace / '001-formal-source.response.md').write_text(response, encoding='utf-8')
(trace / 'run.meta.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with (trace / 'events.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(time_cst=stamp, event='actual_final_response_recorded', verdict='PASS_SOURCE_ONLY')) + '\n')
face = local.parent.parent / 'pvground_face_conditioned_20261004'
state_path = face / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=stamp, status='READBACK_FORMAL_SOURCE_AND_V2_RUNTIME_PASS_NOT_YET_LAUNCHED',
    active_readback_formal_source_reviewer=None, readback_formal_source_review='PASS_SOURCE_ONLY',
    readback_revision2_runtime_pass=True, readback_runtime_checked=True, readback_gpu_preflight_pass=True,
    readback_optimizer_steps=4, readback_formal_training_started=False,
    native_observer_closed=True, native_observer_session_id=None, sole_live_native_session_id=None,
    controller_alive=False, remote_controller_pid=None, readback_preflight_collector_closed=50701,
    last_remote_observation_cst=wait['terminal']['time_cst'],
    readback_revision2_terminal_cst=runtime['actual_terminal_cst'],
    readback_source_review_scope='Fresh42-file formal SOURCE_ONLY PASS; actual V2 both2-updatePASS separately observed.',
    observer_resume_rule='24708 and50701 closed exit0; never resume old observers/collectors.',
    next_actions=['Launch exact reviewed formal pair once; all parent identities and runtime gates are checked by launcher.',
                  'Start one ETA-based formal observer only after actual launch receipt; no early remote polling.',
                  'Publish actual V1 failure, corrected V2 success and actual formal launch without accuracy claim.'])
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
cursor = '\nActual V2 CPU+two-update GPU PASS both arms;4updates,28text receipts,0weights,0accuracy. Observer24708 and collector50701 closed0. Fresh formal42-file SOURCE_ONLY PASS/same-family provisional; formal not yet launched. Never rerun completed preflights/collectors.\n'
for path in (face / 'NEXT_CONTINUATION.md', local.parent / 'NEXT_CONTINUATION.md'):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(cursor)
with (Path('C:/Users/gb/memory') / '2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ' actual correctedV2 terminal23:02:43 bothCPU/twoGPUupdatesPASS, collected28text0weights;24708/50701closed0. Freshformal42fileSOURCE_ONLY PASS samefamily/provisional, no reviewerGPU. Formal not yetlaunched; launch exactreviewedpair next, protected4506parents. GoalACTIVE_UNMET.\n')
print(json.dumps(dict(runtime_status=runtime['status'], formal_source_review='PASS_SOURCE_ONLY', formal_training_started=False)))
