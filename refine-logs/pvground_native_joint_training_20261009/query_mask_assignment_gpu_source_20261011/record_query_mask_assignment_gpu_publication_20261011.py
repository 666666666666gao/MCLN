"""Record reviewed preflight source without changing current GPU scheduling."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_gpu_20261011'
publication = json.loads((root / 'query_mask_assignment_gpu_source_publication.json').read_bytes())
assert publication['status'] == 'QUERY_MASK_ASSIGNMENT_GPU_SOURCE_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section'] == '20.376.150' and publication['github_main_verified']
summary = json.loads((data / 'SOURCE_READINESS_SUMMARY.json').read_bytes())
assert summary['status'] == 'NATIVE_QUERY_MASK_MATCHER_GPU_PREFLIGHT_SOURCE_REVIEWED_NOT_ADMITTED'
assert not summary['serial_gpu_preflight_admitted'] and not summary['native_GPU_preflight_completed']
owner = json.loads((root / 'normal_controls_20261010/NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid'] == 51540 and owner['first_due_cst'] == '2026-10-11T05:00:40.968644+08:00'
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ('.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928'):
    assert hashlib.sha256((Path('C:/Users/gb') / name / doc).read_bytes()).hexdigest() == publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == publication['doc_sha256']
now = datetime.datetime.now().astimezone().isoformat()
classification = 'ACTUAL_PROGRESS_NATIVE_QUERY_MATCHER_REAL_BATCH_PREFLIGHT_SOURCE_REVIEWED_SECTION150_SYNCED'
for path in (Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
        Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')):
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.149'
    value.update(query_mask_assignment_GPU_preparation=summary, query_mask_assignment_GPU_source_publication=publication,
        latest_handoff_section=publication['section'], handoff_sha256=publication['doc_sha256'], published_heads=publication['heads'],
        remote_handoff_sync_pending=False, remote_handoff_last_confirmed_section=publication['section'],
        remote_handoff_last_confirmed_sha256=publication['doc_sha256'], updated_cst=now,
        current_turn_classification=classification, full_goal_complete=False)
    value['normal_training_requirement'].update(next_normal_training_observation_not_before=owner['first_due_cst'],
        current_normal_observer_native_session=52851, current_normal_observer_pid=51540)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
path = root / 'NORMAL_CONTINUATION_STATE.json'
value = json.loads(path.read_bytes())
value.update(query_mask_assignment_GPU_preparation=summary, query_mask_assignment_GPU_source_publication=publication,
    current_turn_classification=classification,
    next_action='Sole normal observer52851/PID51540 dueOct11 05:00:40; C-off full terminal review must precede isolated matcher real-batch GPU preflight. No new controller or queue.',
    full_goal_complete=False)
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + now + ': section20.376.150 synchronized MAIN ' + publication['heads'][0]
    + ', docSHA ' + publication['doc_sha256']
    + '. QueryMask native full-batch GPUpreflight SOURCE_ONLY prepared/reviewed,14sourcesexactQvariant; text/queryprotocolonlyMasksource differs,bothGPUadmissionFalse gatebeforeTorch. ReusesactualCoff native twoB8/16rows/2updates/fullcriterion/1295/E0exactstate/Adam/scheduler/allRNGcoldrestore; plansbothmatchingcomparisonon sameoutputs/targets, returnsconfiguredonly; sourceCLI saved/recreatedexplicit. NoactualPV/data/loss/GPU/update/mainquery/sourceedit/newcontroller/queue/newaccuracy; reviewerWARN0blocks samefamily/provisional/UNATTESTED. Oldwholepreflight647.08s/twoUpdate25.90s supports~11min estimatepermode, notnewmeasurement. C-on/E0 protectedinitialbested8455dd; currentCoffterminalstillneededbeforeGPUornewnormal, considerreuseoldtextcontrolonlyafteractualevidence. Soleobserver52851/PID51540 first05:00:40 unchanged, best5677/4920/parents/V99protected;3effective/fullNrSr stillunproved, goalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-11.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='QUERY_MASK_MATCHER_GPU_PREFLIGHT_SOURCE_SECTION150_RECORDED',
    section=publication['section'], next_main_observation_cst=owner['first_due_cst'], full_goal_complete=False)))
