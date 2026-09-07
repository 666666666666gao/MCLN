import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archives = ['refine-logs/native_score_fit_20260907_v1', 'refine-logs/native_score_fit_20260907_v2']
archive = repo / archives[1]
master_name = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
tracker_name = 'refine-logs/EXPERIMENT_TRACKER.md'
report_name = 'docs/SCANREFER_NATIVE_SCORE_FIT_RESULT_2026-09-07.md'
plan_name = 'docs/SCANREFER_NATIVE_SCORE_FIT_AUDIT_PLAN_2026-09-07.md'
script_names = ['scripts/run_scanrefer_native_score_fit_audit.py', 'scripts/analyze_native_score_fit_audit.py', 'scripts/recover_native_score_fit_receipt.py']
desktop = Path('C:/Users/gb/Desktop/document') / Path(master_name).name
old = (repo / master_name).read_bytes()
assert hashlib.sha256(old).hexdigest() == 'f6a4605b75bcdecd1ec9a10d02ba227ffcfe3a3ef88099cdf47fe0c53c556032'
assert desktop.read_bytes() == old
receipt_raw = (archive / 'receipt.json').read_bytes()
receipt = json.loads(receipt_raw)
analysis = json.loads((archive / 'independent_analysis.json').read_text())
assert receipt['status'] == 'complete' and receipt['rows'] == analysis['rows'] == 512
assert analysis['runner_receipt_sha256'] == hashlib.sha256(receipt_raw).hexdigest()
assert analysis['all_candidate_threshold_decisions_match']
assert analysis == json.loads((archive / 'local_independent_analysis.json').read_text())


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(repo))


assert git('rev-parse', 'HEAD').decode().strip() == '09c07c2537e71279081d33842c67541093d047c7'
assert not git('diff', '--cached', '--name-only').strip()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition = Path('C:/Users/gb/.codex/tmp/native_score_fit_handoff_addition.md').read_text(encoding='utf-8').replace('TIME_CST', now)
new = old + addition.encode()
tracker = (repo / tracker_name).read_text(encoding='utf-8').splitlines()
tracker[2] = 'Updated: ' + now + '. Section20.142: fixed512 real-input score audit and independent CPU recount complete;no training/formal/Nr/Sr.'
tracker.insert(6, '| Native/adapter score actual fit audit | 512 fixed rows;actual evaluator and SourceChoice recorded;independent all256 score/IoU/candidate recount | Top1 disagreements ' + str(analysis['top1_disagreements']) + ';native Top1 excluded ' + str(analysis['native_top1_excluded_original']) + ';no deployment or metric-gain claim;see20.142 |')
for i, line in enumerate(tracker):
    if line.startswith('| Native default score definitions |'):
        tracker[i] = line.replace('real fit prevalence remains to measure', 'real fit prevalence now recorded in20.142')
new_tracker = ('\n'.join(tracker) + '\n').encode()
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
canonical = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote = '/root/autodl-tmp/mcln_native_score_fit_20260907_v2'
for name in [master_name, tracker_name]:
    with s.open(canonical + name, 'rb') as stream:
        assert stream.read() == git('show', 'HEAD:' + name), name
with s.open(remote + '/receipt.json', 'rb') as stream:
    assert stream.read() == receipt_raw
for name, raw in [(master_name, new), (tracker_name, new_tracker)] + [(n, (repo / n).read_bytes()) for n in [report_name, plan_name] + script_names]:
    with s.open(canonical + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with s.open(canonical + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
for archive_name in archives:
    target = repo / archive_name
    remote_archive = '/root/autodl-tmp/mcln_' + Path(archive_name).name
    binding = json.loads((target / 'input_manifest.json').read_text())
    for name, expected in binding['files'].items():
        with s.open(remote_archive + '/' + name, 'rb') as stream:
            content = stream.read()
        assert hashlib.sha256(content).hexdigest() == expected
        (target / name).write_bytes(content)
(repo / master_name).write_bytes(new)
desktop.write_bytes(new)
(repo / tracker_name).write_bytes(new_tracker)
proof = {'time_cst': now, 'master_sha256': hashlib.sha256(new).hexdigest(),
         'three_master_copies_equal': True, 'receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
         'analysis_sha256': hashlib.sha256((archive / 'independent_analysis.json').read_bytes()).hexdigest(),
         'rows': 512, 'optimizer_steps': 0, 'formal_rows': 0, 'goal_complete': False}
(archive / 'handoff_sync.json').write_bytes(json.dumps(proof, indent=2).encode() + b'\n')
(archive / 'publish_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['handoff_sync.json', 'collect_from_local.py', 'publish_from_local.py', 'recover_from_local.py', 'recovery_exit.json', 'local_verification.json', 'local_verify_from_source.py']:
    s.put(str(archive / name), remote + '/' + name)
s.close()
c.close()
with (repo / '.gitattributes').open('a', encoding='utf-8', newline='\n') as stream:
    for path in archives:
        stream.write(path + '/** -text\n')
        stream.write(path + '/*.txt -whitespace\n')
paths = [master_name, tracker_name, report_name, plan_name, '.gitattributes'] + script_names + archives
git('add', '--', *paths)
for name in git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0'):
    assert name in paths or any(name.startswith(p + '/') for p in archives), name
    content = git('show', ':' + name)
    inspected = gzip.decompress(content) if name.endswith('.gz') else content
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in inspected
    assert len(content) < 10 * 1024**2 and Path(name).suffix not in ['.pt', '.pth', '.pkl', '.npz']
    if any(name.startswith(p + '/') for p in archives):
        assert content == (repo / name).read_bytes()
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Measure native REC score semantics on fixed ScanRefer fit inputs').decode(), flush=True)
print(git('push', 'origin', 'HEAD:main').decode(), flush=True)
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert not git('status', '--porcelain').strip()
proof['commit'] = head
Path('C:/Users/gb/.codex/tmp/native_score_fit_publication.json').write_bytes(json.dumps(proof, indent=2).encode())
print(json.dumps(proof), flush=True)
