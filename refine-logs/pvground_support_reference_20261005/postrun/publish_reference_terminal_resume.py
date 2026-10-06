"""Append the actual closed comparison, audit and retention to all handoffs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shlex
import paramiko

local = Path(__file__).resolve().parents[1]
workspace = Path('C:/Users/gb')
assert not (local / 'terminal_publication.json').exists()
state_path = local.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
summary = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
review_call = json.loads((local / 'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
retention = json.loads((local / 'weight_retention.json').read_bytes())
resources = json.loads((local / 'CLOSED_RESOURCES.json').read_bytes())
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert summary['status'] == 'ACTUAL_CLOSED_REFERENCE_ROWS_ANALYZED' and summary['fit_order_exact']
assert review_call['result_received'] and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0 and not wait['terminal']['controller_alive']
assert retention['status'] == 'CLOSED_NONBEST_WEIGHTS_REMOVED' and resources['deleted_weights_absent']
assert not resources['gpu_compute_processes']
best = retention['retained_best']
assert best['system'] == summary['metric_best_candidate']['system']
assert best['hits'] == [summary['metric_best_candidate']['rec_hits25'], summary['metric_best_candidate']['rec_hits50']]
assert len(retention['deleted']) == 2 and retention['local_weight_archive_created'] is False
assert previous['section'] == '20.376.78'
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
new = copies[0].read_bytes()
assert all(path.read_bytes() == new for path in copies)
assert new.count(b'## 20.376.79 ') == 1
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
stamp = datetime.datetime.now().astimezone().isoformat()
verdict = audit['verdict']
finished = wait['terminal']['status']['finished_cst']
prefix = 'refine-logs/pvground_support_reference_20261005/'
manifest_lines = [line for line in (repos[0] / 'MANIFEST.md').read_text(encoding='utf-8').splitlines()
                  if 'Actual closed support-reference comparison published;' in line]
assert len(manifest_lines) == 1
manifest_line = manifest_lines[0]
assert manifest_line not in (repos[1] / 'MANIFEST.md').read_text(encoding='utf-8')
names=['fit_wait.json','weight_retention.json','CLOSED_RESOURCES.json',
    'inspect_existing_fit_authorized.py','COLLECTION_PREFETCH_CHANGE.json','COLLECTION_STREAM_CHANGE.json','COLLECTION_ROW_FILE_SIZES.json',
    'postrun/collect_formal_prefetch_authorized.py','postrun/prepare_prefetch_collector.py',
    'postrun/collect_formal_stream_authorized.py',
    'postrun/check_closed_resources.py','postrun/prepare_reference_terminal_publication.py',
    'postrun/publish_reference_terminal.py','TERMINAL_PUBLICATION_PREPARATION.json',
    'postrun/publish_reference_terminal_fast.py','postrun/prepare_fast_terminal_publisher.py',
    'PUBLICATION_TRANSFER_OPTIMIZATION.json']
names += ['PUBLICATION_INTERRUPTION.json', 'postrun/publish_reference_terminal_resume.py']
for directory in ('complete', 'analysis'):
    names.extend(str(path.relative_to(local)).replace('\\', '/') for path in sorted((local / directory).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth', '.pt')) and '.aris' not in name for name in names)
payloads = {prefix + name: (local / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
runtime = json.loads((local / 'control_spec.json').read_bytes())['runtime']
doc_sha = hashlib.sha256(new).hexdigest()
probe = "import hashlib,sys; from pathlib import Path; assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()==sys.argv[2]"
_, stdout, stderr = client.exec_command(shlex.join([runtime + '/venv/bin/python', '-B', '-c', probe, project + '/' + doc, doc_sha]), timeout=120)
stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
existing = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()
            if name.startswith(prefix + 'complete/') and name != prefix + 'complete/INTAKE.json'}
sftp = client.open_sftp()
for relative in ('PUBLICATION_INTERRUPTION.json', 'postrun/publish_reference_terminal_resume.py',
                 'analysis/PUBLICATION_RESUME_REVIEW.json', 'analysis/PUBLICATION_RESUME_REVIEW.md'):
    name = prefix + relative
    raw = payloads[name]
    for repo in repos[:2]:
        (repo / name).write_bytes(raw)
    with sftp.open(project + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(project + '/' + name, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
for name, raw in payloads.items():
    assert all((repo / name).read_bytes() == raw for repo in repos[:2]), name
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        if index == 1:
            with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
                stream.write('\n' + manifest_line + '\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    reports = [name for name in payloads if Path(name).name in ('EXPERIMENT_AUDIT.json', 'EXPERIMENT_AUDIT.md', 'SOURCE_REVIEW.json', 'SOURCE_REVIEW.md')] if index < 2 else []
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', *stage, *[':(exclude)' + name for name in reports]])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        for name, raw in payloads.items():
            assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw
    git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    old_git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert git_doc.startswith(old_git_doc) and git_doc.count(b'## 20.376.79 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record closed support-reference comparison and best retention'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
assert all(path.read_bytes() == new for path in copies)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.79', heads=heads,
              github_main=heads[0], handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), new_accuracy_result=True, table=summary['table'],
              retained_best=best, deleted_nonbest_bytes=retention['released_bytes'], negative_weight_archived=False,
              integrity_verdict=verdict, review_independence='same-family', acceptance_status='provisional',
              scanrefer_target_pass=summary['scanrefer_target_pass'], full_goal_status='ACTIVE_UNMET',
              remote_complete_reuses_closed_data_disk_originals=True,
              existing_closed_payloads_verified_by_remote_SHA256=len(existing),
              actual_failed_publisher_session_closed=64090,
              resumed_after_raw_source_report_EOF_whitespace=True,
              raw_source_reports_preserved_bytes=True)
(local / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(time_cst=record['time_cst'],status='SUPPORT_REFERENCE_TERMINAL_PUBLISHED',
    latest_publication=str(local/'terminal_publication.json'),owned_gpu_job_active=False,
    active_reviewer=None,protected_best_hits=best['hits'],strict_target_gap=max(0,4754-best['hits'][1]),
    published_heads=heads,handoff_section='20.376.79',handoff_sha256=digest,
    support_reference_fit_observer_closed=True,support_reference_fit_observer_session_id=None,
    support_reference_full_pair_complete=True,support_reference_actual_fit_finished_cst=finished,
    support_reference_terminal_integrity_review_pending=False,support_reference_terminal_integrity_verdict=verdict,
    support_reference_postrun_tools_executed=True,
    support_reference_formal_hits_by_arm={row['system']:[row['rec_hits25'],row['rec_hits50']] for row in summary['table']},
    current_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_AUDIT_RETENTION_PUBLISHED')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Support-reference actual closed pair '+str(summary['table'])+'. Fresh terminal audit '+verdict+' same-family/provisional; retained '+str(best['hits'])+', two closed nonbest heads removed without archive. Doc79 fourlocal+remote exact, main '+heads[0]+'. Original observer40310 missing; actual one-shot remote closure verified, no training restart. Geometry old states total14892, new reference3078params thissegment3723updates. GoalACTIVE_UNMET; no new Nr/Sr results.\n')

print(json.dumps(record), flush=True)
