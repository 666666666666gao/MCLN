"""Record bounded CPU evidence; keep the normal training observer unchanged."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_20261011'
publication = json.loads((root / 'query_mask_assignment_cpu_publication.json').read_bytes())
assert publication['status'] == 'QUERY_MASK_ASSIGNMENT_CPU_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section'] == '20.376.149' and publication['github_main_verified']
summary = json.loads((data / 'CPU_CHECK_SUMMARY.json').read_bytes())
assert summary['status'] == 'ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_CPU_CHECK_AUDITED'
assert summary['formal_accuracy'] is None and summary['gpu_calls'] == 0
owner = json.loads((root / 'normal_controls_20261010/NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid'] == 51540 and owner['first_due_cst'] == '2026-10-11T05:00:40.968644+08:00'
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ('.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928'):
    assert hashlib.sha256((Path('C:/Users/gb') / name / doc).read_bytes()).hexdigest() == publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == publication['doc_sha256']
now = datetime.datetime.now().astimezone().isoformat()
classification = 'ACTUAL_PROGRESS_QUERY_MASK_MATCHING_CPU_GRADIENT_PATH_VERIFIED_SECTION149_SYNCED'
for path in (Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
        Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')):
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.148'
    value.update(query_mask_assignment_CPU=summary, query_mask_assignment_CPU_publication=publication,
        latest_handoff_section=publication['section'], handoff_sha256=publication['doc_sha256'],
        published_heads=publication['heads'], remote_handoff_sync_pending=False,
        remote_handoff_last_confirmed_section=publication['section'], remote_handoff_last_confirmed_sha256=publication['doc_sha256'],
        updated_cst=now, current_turn_classification=classification, full_goal_complete=False)
    value['normal_training_requirement'].update(next_normal_training_observation_not_before=owner['first_due_cst'],
        current_normal_observer_native_session=52851, current_normal_observer_pid=51540)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
path = root / 'NORMAL_CONTINUATION_STATE.json'
value = json.loads(path.read_bytes())
value.update(query_mask_assignment_CPU=summary, query_mask_assignment_CPU_publication=publication,
    current_turn_classification=classification,
    next_action='Wait for sole normal observer52851/PID51540 dueOct11 05:00:40; current C-off closure precedes native GPU preflight or a new normal matcher training control.',
    full_goal_complete=False)
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + now + ': section20.376.149 synchronized MAIN ' + publication['heads'][0]
    + ', docSHA ' + publication['doc_sha256']
    + '. Actual isolated QueryMask matcher CPU33calls/44assignmentproblems/66GTpairs;6native SetCriterion boxes-only forward/backward. Default/text equalswarmoriginal; queryownMask changescost/matches, noMask pathunchanged; directL1/GIoU gradientsfollownewmatches. Synthetic tiedgeometryonly,0PV/data/fullcriterion/Maskloss/optimizer/GPU/mainquery/newaccuracy; notGPU/normaltrainingadmission. Freshsource/actualWARN0blocks samefamily/provisional/UNATTESTED. CurrentCoff untouched; soleobserver52851/PID51540 first05:00:40 preserved, best5677/4920/parents/V99protected. NeedCoffterminaldecisionbeforenativeGPU ornewtraining;threeeffective/fullNrSr unproved. GoalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-11.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='QUERY_MASK_ASSIGNMENT_ACTUAL_CPU_SECTION149_RECORDED', section=publication['section'],
    next_main_observation_cst=owner['first_due_cst'], full_goal_complete=False)))
