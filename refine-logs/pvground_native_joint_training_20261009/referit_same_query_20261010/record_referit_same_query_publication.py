"""Record section143 without changing the live C-off job or its observer."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
publication = json.loads((root / 'referit_same_query_publication.json').read_bytes())
assert publication['status'] == 'REFERIT_SAME_QUERY_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section'] == '20.376.143' and publication['github_main_verified']
data = root / 'referit_same_query_20261010'
summary = json.loads((data / 'SAME_QUERY_CPU_SUMMARY.json').read_bytes())
assert not summary['review']['blocking_findings']
assert not summary['GPU_training_admission'] and summary['formal_accuracy'] is None
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ('.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928'):
    assert hashlib.sha256((Path('C:/Users/gb') / name / doc).read_bytes()).hexdigest() == publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == publication['doc_sha256']
now = datetime.datetime.now().astimezone().isoformat()
interface = dict(status=summary['status'],
    result=str(data / 'cpu_execution/SAME_QUERY_CPU_RESULT.json'),
    actual_review=str(data / 'actual_review/EXPERIMENT_AUDIT.json'),
    source_review_verdict=summary['review']['source_verdict'],
    actual_review_verdict=summary['review']['actual_verdict'],
    review_scope='SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU',
    source_warm_sha256='39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677',
    author_bbs_ranking_preserved=True, mask_protocol='Same Query as native last/bbs winner',
    fixture_cases=5, evaluator_instances=10, native_evaluate_calls=10,
    real_dataset_evaluation=False, GPU_training_admission=False, formal_accuracy=None,
    C_training_vs_filtered_deployment_Query_equivalence_verified=False,
    final_method_selection_pending=True)
classification = 'ACTUAL_PROGRESS_NATIVE_SAME_QUERY_CPU_AND_SECTION143_SYNCHRONIZATION'
for path in [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
             Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')]:
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.142'
    value.update(latest_handoff_section='20.376.143', handoff_sha256=publication['doc_sha256'],
        published_heads=publication['heads'], remote_handoff_sync_pending=False,
        remote_handoff_last_confirmed_section='20.376.143',
        remote_handoff_last_confirmed_sha256=publication['doc_sha256'], updated_cst=now,
        latest_referit_same_query_publication=publication, referit_same_query_cpu=interface,
        referit_author_entry_source_review_complete=True,
        referit_author_entry_execution_completed=False,
        current_turn_classification=classification, full_goal_complete=False)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
state_path = root / 'NORMAL_CONTINUATION_STATE.json'
state = json.loads(state_path.read_bytes())
state.update(referit_same_query_cpu=interface, referit_same_query_publication=publication,
    current_turn_classification=classification,
    next_action='Original C-off observer21631/PID48772 first dueOct11 01:03:05; same-Query CPU check61578 completed and consumed, do not restart; finalize architecture after C-off result before Nr/Sr GPU checks.',
    full_goal_complete=False)
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
note = '\nPV-Ground '+now+': section20.376.143 allcopies/GitHub synchronized MAIN '+publication['heads'][0]+', docSHA '+publication['doc_sha256']+'; real native evaluator CPU engineering check22:07:50--22:08:00 completedexit0, fiveexplicit B1/Q256/T256/P4 syntheticfixtures/10evaluatecalls. Original author bbs preserved; bothMaskreports now use bbswinner, positivefilter fixture correctlylowersMaskIoU1to0. Actualboundwarm evaluator39c8de92, notdataset-source b77376d. Freshsource '+summary['review']['source_verdict']+'/actual '+summary['review']['actual_verdict']+'/0blocks, identityUNATTESTED same-family/provisional. No realdata/PVforward/loss/optimizer/GPU/weights/accuracy/mainquery; noGPUadmission. Normalentry authorflag assertions prepared only; finalC andtraining/deploymentselectionequivalence stillpending. OriginalC-off source/observer21631/PID48772 unchanged, firstdueOct11 01:03:05. Best5677/4920 preserved; normalefficacy/threeeffective/fullNrSr remainunproved; goalACTIVE_UNMET.\n'
for path in [Path('C:/Users/gb/memory/2026-10-10.md'), Path('C:/Users/gb/MEMORY.md')]:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='REFERIT_SAME_QUERY_SECTION143_CANONICAL_AND_MEMORY_RECORDED',
    section=publication['section'], main=publication['heads'][0], full_goal_complete=False)))
