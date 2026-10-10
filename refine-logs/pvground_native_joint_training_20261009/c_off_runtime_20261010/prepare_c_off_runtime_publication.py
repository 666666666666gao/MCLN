"""Publish only observed facts; no model or remote training-status query."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
control = root / 'normal_controls_20261010'
diagnosis = root / 'normal_degradation_assessment_20261010'
terminal = json.loads((root / 'NORMAL_TERMINAL_SUMMARY_20261010.json').read_bytes())
launch = json.loads((control / 'ENGINEERING_LAUNCH.json').read_bytes())
assert json.loads((control / 'ENGINEERING_LAUNCH_EXIT.json').read_bytes())['exit_code'] == 0
observations = json.loads((control / 'ENGINEERING_OBSERVATIONS.json').read_bytes())
panel = json.loads((diagnosis / 'actual_panel/results/PANEL_RECEIPT.json').read_bytes())
assert panel['actual_neural_forwards'] == 16 and panel['new_optimizer_steps'] == 0
archive = json.loads((root / 'normal_terminal_recovery_20261010/ARCHIVE_COMPLETE.json').read_bytes())
assert len(archive['files']) == 2
retirement = json.loads((diagnosis / 'EXACT_E3_WEIGHT_RETIREMENT.json').read_bytes())
assert retirement['deleted_count'] == 1 and retirement['deleted_bytes'] == 841676832
reviews = {}
for kind in ('engineering', 'normal'):
    review = json.loads((control / (kind + '_source_review') / 'SOURCE_REVIEW.json').read_bytes())
    assert review['execution_scope'] == 'SOURCE_ONLY' and not review['blocking_findings']
    reviews[kind] = dict(verdict=review['verdict'], execution_scope='SOURCE_ONLY',
                         blocking_findings=[], actual_runtime_model='UNATTESTED',
                         independence='same-family', acceptance='provisional')
summary = dict(terminal)
summary.update(status='NORMAL_TERMINAL_DIAGNOSIS_AND_C_OFF_ENGINEERING_LAUNCH_RECORDED',
               prepared_cst=datetime.datetime.now().astimezone().isoformat(),
               best_and_latest_full_local_archive_complete=True, full_archive=archive,
               inferior_E3_remote_retirement=retirement, same64_panel=panel,
               same64_panel_does_not_explain_full_validation_decline=True,
               selected_mask_off_launch=launch, source_reviews=reviews,
               selected_mask_off_latest_observation=observations['observations'][-1],
               selected_mask_off_actual_M0_completion_observed=False,
               selected_mask_off_normal_training_launched=False,
               mask_off_engineering_never_used_for_formal_training=True,
               historical_pretrained_parents_already_C_adapted=True,
               no_history_never_C_claim=True,
               normal_storage_reserve_bytes=2605151860,
               normal_storage_after_E3_retirement_bytes=retirement['data_free_after'],
               normal_storage_gap_before_M0_bytes=2605151860-retirement['data_free_after'],
               no_normal_resource_gate_relaxed=True,
               retired_SR_cache_archive_complete=False, retired_SR_cache_deleted=False,
               next_action='Collect original engineering observer at planned 240-second cadence; actual audit; verified cache cleanup; one normal C-off arm.',
               full_goal_complete=False)
(control / 'C_OFF_RUNTIME_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
names = [
    'normal_controls_20261010/' + name for name in (
        'HANDOFF_C_OFF_RUNTIME_20261010.md', 'C_OFF_RUNTIME_SUMMARY.json',
        'native_c_off_preflight.py', 'c_off_engineering_controller.py',
        'launch_c_off_engineering_authorized.py', 'observe_c_off_engineering_authorized.py',
        'launch_c_off_normal_authorized.py', 'NORMAL_NATIVE_RUN_PROTOCOL.json',
        'init.json', 'normal_joint_controller.py', 'ENGINEERING_LAUNCH.json',
        'ENGINEERING_LAUNCH_STDOUT.json', 'ENGINEERING_LAUNCH_EXIT.json',
        'ENGINEERING_OBSERVATIONS.json', 'engineering_source_review/SOURCE_REVIEW.md',
        'engineering_source_review/SOURCE_REVIEW.json', 'normal_source_review/SOURCE_REVIEW.md',
        'normal_source_review/SOURCE_REVIEW.json')]
names += ['normal_degradation_assessment_20261010/' + name for name in (
    'PANEL_PLAN.md', 'PANEL_RESULT.json', 'actual_panel/results/PANEL_RECEIPT.json',
    'compare_retained_and_terminal_panel.py', 'source_review/SOURCE_REVIEW.md',
    'source_review/SOURCE_REVIEW.json', 'LOG_DEGRADATION_SUMMARY.json',
    'EXACT_E3_WEIGHT_RETIREMENT.json', 'E3_RETIREMENT_STDOUT.json', 'E3_RETIREMENT_EXIT.json',
    'prepare_verified_E3_retirement.py', 'retire_verified_E3_authorized.py',
    'SR_ARCHIVE_TRANSPORT_DIAGNOSIS.json')]
names += ['normal_terminal_recovery_20261010/ARCHIVE_COMPLETE.json',
          'prepare_c_off_runtime_publication.py']
assert all((root / name).is_file() for name in names)
assert not any('STDERR' in name or 'private/' in name or 'request' in name or 'response' in name for name in names)
(root / 'C_OFF_RUNTIME_PUBLIC_FILE_LIST.json').write_text(json.dumps(sorted(set(names)), indent=2) + '\n', encoding='utf-8')
publisher = (root / 'publish_normal_terminal_authorized.py').read_text()
publisher = publisher.replace("'normal_terminal_publication.json'", "'c_off_runtime_publication.json'")
publisher = publisher.replace("'normal_epoch2_publication.json'", "'normal_terminal_publication.json'")
publisher = publisher.replace("prior['section'] == '20.376.137'", "prior['section'] == '20.376.138'")
publisher = publisher.replace("b'20.376.138' not in old", "b'20.376.139' not in old")
publisher = publisher.replace("section='20.376.138'", "section='20.376.139'")
publisher = publisher.replace('NORMAL_TERMINAL_PUBLIC', 'C_OFF_RUNTIME_PUBLIC')
publisher = publisher.replace("root / 'NORMAL_TERMINAL_SUMMARY_20261010.json'", "root / 'normal_controls_20261010/C_OFF_RUNTIME_SUMMARY.json'")
publisher = publisher.replace("root / 'HANDOFF_NORMAL_TERMINAL_20261010.md'", "root / 'normal_controls_20261010/HANDOFF_C_OFF_RUNTIME_20261010.md'")
publisher = publisher.replace('normal_terminal_20261010/', 'c_off_runtime_20261010/')
publisher = publisher.replace('/normal_terminal_20261010\'', '/c_off_runtime_20261010\'')
publisher = publisher.replace('publish_normal_terminal_authorized.py', 'publish_c_off_runtime_authorized.py')
publisher = publisher.replace('normal_terminal_local_commit.json', 'c_off_runtime_local_commit.json')
publisher = publisher.replace('NORMAL_TERMINAL_ALL_', 'C_OFF_RUNTIME_ALL_')
publisher = publisher.replace('Record normal three-epoch terminal results and exact checkpoint recovery',
                              'Record bounded normal degradation diagnosis and C-off engineering launch')
(root / 'publish_c_off_runtime_authorized.py').write_text(publisher)
print(json.dumps(dict(status=summary['status'], public_files=len(names), normal_training_started=False,
                      engineering_observation=summary['selected_mask_off_latest_observation'])))
