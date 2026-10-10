"""Record reviewed source preparation and preserve the sole C-off observer."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
publication = json.loads((root / 'referit_gpu_source_publication.json').read_bytes())
assert publication['status'] == 'REFERIT_GPU_SOURCE_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section'] == '20.376.144' and publication['github_main_verified']
data = root / 'referit_native_gpu_preparation_20261010'
summary = json.loads((data / 'SOURCE_READINESS_SUMMARY.json').read_bytes())
assert not summary['review']['blocking_findings']
assert not summary['GPU_training_admission'] and summary['formal_accuracy'] is None
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ('.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928'):
    assert hashlib.sha256((Path('C:/Users/gb') / name / doc).read_bytes()).hexdigest() == publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == publication['doc_sha256']
now = datetime.datetime.now().astimezone().isoformat()
preparation = dict(status=summary['status'], source_review_verdict=summary['review']['source_verdict'],
    source_review=str(data / 'source_review/EXPERIMENT_CODE_REVIEW.json'), execution_scope='SOURCE_ONLY',
    prepared_check=str(data / 'native_referit_preflight.py'), provisional_C_off=True,
    final_method_selected=False, source_deployed=False, GPU_training_admission=False,
    real_execution_completed=False, formal_accuracy=None,
    next_action='Wait actual ScanRefer C-off result/final method, then actual single-card resource intake and isolated GPU check; no automatic launch.')
classification = 'ACTUAL_PROGRESS_REFERIT_NATIVE_GPU_CHECK_SOURCE_PREPARED_AND_SECTION144_SYNCHRONIZED'
for path in [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
             Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')]:
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.143'
    value.update(latest_handoff_section='20.376.144', handoff_sha256=publication['doc_sha256'],
        published_heads=publication['heads'], remote_handoff_sync_pending=False,
        remote_handoff_last_confirmed_section='20.376.144',
        remote_handoff_last_confirmed_sha256=publication['doc_sha256'], updated_cst=now,
        referit_native_gpu_check_preparation=preparation,
        latest_referit_gpu_source_publication=publication,
        current_turn_classification=classification, full_goal_complete=False)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
state_path = root / 'NORMAL_CONTINUATION_STATE.json'
state = json.loads(state_path.read_bytes())
state.update(referit_native_gpu_check_preparation=preparation,
    referit_gpu_source_publication=publication, current_turn_classification=classification,
    next_action='Sole normal observer21631/PID48772 dueOct11 01:03:05; Nr/Sr native GPU-check source reviewed only, do not run until final method and resource admission.',
    full_goal_complete=False)
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
note = '\nPV-Ground '+now+': section20.376.144 allcopies/GitHub synchronized MAIN '+publication['heads'][0]+', docSHA '+publication['doc_sha256']+'; prepared native Nr/Sr twoB8 mixed expression/detection training plus1295/model/Adam/scheduler/RNG restore check. Corresponding authorcores/freshG-A-B/provisionalC-off;15prior sources unchanged, onlytwoinit Cflagsfalse. Native fullget_loaders withoutdebug; source shows debugwouldreplace trainbyval. Srnonemptyanchor/detectionmultiGT protection andGdirectgradientchecks prepared; noGPU/data/model/loss/update/restore execution, noaccuracy, nofinalmethod/admission. FreshSOURCE_ONLY '+summary['review']['source_verdict']+'/0blocks, identityUNATTESTED same-family/provisional. ActiveC-off source/observer21631/PID48772 unchanged, firstdueOct11 01:03:05; noearlypoll. Best5677/4920 preserved; normaltraininggain/threeeffective/fullNrSr unproved, goalACTIVE_UNMET.\n'
for path in [Path('C:/Users/gb/memory/2026-10-10.md'), Path('C:/Users/gb/MEMORY.md')]:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='REFERIT_GPU_SOURCE_SECTION144_CANONICAL_AND_MEMORY_RECORDED',
    section=publication['section'], main=publication['heads'][0], full_goal_complete=False)))
