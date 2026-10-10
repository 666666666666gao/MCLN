"""Record the closed CPU investigation without changing the one normal observer."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'matcher_mask_role_cpu_20261011'
publication = json.loads((root / 'matcher_mask_role_publication.json').read_bytes())
assert publication['status'] == 'MATCHER_MASK_ROLE_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section'] == '20.376.147' and publication['github_main_verified']
summary = json.loads((data / 'ACTUAL_CHECK_SUMMARY.json').read_bytes())
assert summary['status'] == 'ACTUAL_NATIVE_MATCHER_CPU_CHECK_AUDITED'
assert summary['fixture_count'] == summary['matcher_calls'] == 9
assert summary['formal_accuracy'] is None and summary['gpu_calls'] == 0
owner = json.loads((root / 'normal_controls_20261010/NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid'] == 51540 and owner['first_due_cst'] == '2026-10-11T05:00:40.968644+08:00'
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ('.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928'):
    assert hashlib.sha256((Path('C:/Users/gb') / name / doc).read_bytes()).hexdigest() == publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == publication['doc_sha256']
now = datetime.datetime.now().astimezone().isoformat()
classification = 'ACTUAL_PROGRESS_NATIVE_MATCHER_MASK_ROLE_CPU_CHECK_SECTION147_SYNCED'
for path in (
        Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
        Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')):
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.146'
    value.update(native_matcher_mask_role_cpu=summary, native_matcher_mask_role_publication=publication,
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
value.update(native_matcher_mask_role_cpu=summary, native_matcher_mask_role_publication=publication,
    current_turn_classification=classification,
    next_action='Keep the sole authorized normal observer52851/PID51540 dueOct11 05:00:40; finish the active C-off budget before any new matcher training. CPU check completed once, not benchmark accuracy.',
    full_goal_complete=False)
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + now + ': section20.376.147 synchronized MAIN ' + publication['heads'][0]
    + ', docSHA ' + publication['doc_sha256']
    + '. Actual unchanged native HungarianMatcher CPU check completed9fixtures/9calls; B1/B2 Q256 T256 four syntheticpoints/root1or2GT. Actualwarm threecore sources andutils/envSHA verified; swapping ownQueryMask with boxes/logits heldfixed leaves fullnativecost identical; sharedTextMask cost changes onlyGTcolumnconstants withinrecordedfloatrounding. Fullmatrixcapturedonlyinmemory, summaryandassignmentretained. QueryMask stillchanges finalboxes via native_mask_geometry/spanmixer andthusindirectmatching; NOT dropcausality ornewaccuracy. Source/actual reviews freshsamefamily/provisional/UNATTESTED '
    + summary['source_review_verdict'] + '/' + summary['actual_review_verdict']
    + '/0blocks. ZeroPVforward/dataset/criterion/optimizer/GPU/maintrainingqueries, no newmodelpolicyimplemented. Originalnormalobserver52851/PID51540 first05:00:40 maintained, source/recipe/best5677/4920 untouched; latestactualE1 remains5644/4604interim. Threeeffective/fullNrSr unproved, goalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-11.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='MATCHER_MASK_ROLE_SECTION147_CANONICAL_AND_MEMORY_RECORDED',
    section=publication['section'], next_main_observation_cst=owner['first_due_cst'], full_goal_complete=False)))
