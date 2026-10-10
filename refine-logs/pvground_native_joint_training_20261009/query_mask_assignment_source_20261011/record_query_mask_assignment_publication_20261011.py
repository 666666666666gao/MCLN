"""Record reviewed isolated source while preserving the active normal job."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_20261011'
publication = json.loads((root / 'query_mask_assignment_source_publication.json').read_bytes())
assert publication['status'] == 'QUERY_MASK_ASSIGNMENT_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section'] == '20.376.148' and publication['github_main_verified']
summary = json.loads((data / 'SOURCE_READINESS_SUMMARY.json').read_bytes())
assert summary['status'] == 'ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_SOURCE_REVIEWED_NOT_ADMITTED'
assert not summary['training_started'] and not summary['actual_CPU_check_completed']
assert summary['active_training_source_mutations'] == summary['current_training_queries'] == 0
owner = json.loads((root / 'normal_controls_20261010/NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid'] == 51540 and owner['first_due_cst'] == '2026-10-11T05:00:40.968644+08:00'
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ('.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928'):
    assert hashlib.sha256((Path('C:/Users/gb') / name / doc).read_bytes()).hexdigest() == publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == publication['doc_sha256']
now = datetime.datetime.now().astimezone().isoformat()
classification = 'ACTUAL_PROGRESS_NATIVE_QUERY_MASK_ASSIGNMENT_SOURCE_IMPLEMENTED_REVIEWED_SECTION148_SYNCED'
for path in (
        Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
        Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')):
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.147'
    value.update(query_mask_assignment_source=summary, query_mask_assignment_source_publication=publication,
        latest_handoff_section=publication['section'], handoff_sha256=publication['doc_sha256'],
        published_heads=publication['heads'], remote_handoff_sync_pending=False,
        remote_handoff_last_confirmed_section=publication['section'],
        remote_handoff_last_confirmed_sha256=publication['doc_sha256'],
        updated_cst=now, current_turn_classification=classification, full_goal_complete=False)
    value['normal_training_requirement'].update(
        next_normal_training_observation_not_before=owner['first_due_cst'],
        current_normal_observer_native_session=52851, current_normal_observer_pid=51540)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
path = root / 'NORMAL_CONTINUATION_STATE.json'
value = json.loads(path.read_bytes())
value.update(query_mask_assignment_source=summary, query_mask_assignment_source_publication=publication,
    current_turn_classification=classification,
    next_action='Prepare and source-review limited CPU checks for isolated Query Mask assignment; no GPU or new training before current C-off closure. Preserve sole normal observer52851/PID51540 dueOct11 05:00:40.',
    full_goal_complete=False)
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + now + ': section20.376.148 synchronized MAIN ' + publication['heads'][0]
    + ', docSHA ' + publication['doc_sha256']
    + '. Isolated QueryMask-assignment source implemented: exactlymain_utils.py/models.losses2changed,12originalsourcesbyteequal. CLI text(default)/query passednativefactory; queryuses ownsp_pred_masks onlywhenlastMaskavailable, retains threshold>0/rawpointmapping/L1/.0002/class1/bbox0/giou2 andallvalidGTone-to-one. Sixearlierprefixes/model.forward/256candidates/last-bbs unchanged;0newmodelparams. SOURCE_ONLY '
    + summary['source_review_verdict']
    + '/0blocks samefamily/provisional/UNATTESTED, notCPU/GPU/trainingadmission. NoCPUstrategycheck/GPUpreflight/model/data/criterion/updates/newaccuracy/activeedit/mainquery/queuedjob. MaskDINOalreadyqueryMaskmatching, notnewthirdcontribution; no trainingcauseclaim. CurrentCoff untouched, originalobserver52851/PID51540 first05:00:40 maintained; preservebest5677/4920 andparents/V99. NextlimitedCPUstrategyengineering thenCoffterminaldecisionbeforeGPU. Threeeffective/fullNrSr unproved, goalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-11.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='QUERY_MASK_ASSIGNMENT_SECTION148_CANONICAL_AND_MEMORY_RECORDED',
    section=publication['section'], next_main_observation_cst=owner['first_due_cst'], full_goal_complete=False)))
