"""Diagnose SSH and read only pending static copies; never read normal-job state."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
destination = root / 'read_only_static_diagnostic_20261010_attempt1'
assert not destination.exists()
destination.mkdir()
publication = json.loads((root / 'native_direct_controls_publication.json').read_bytes())
assert publication['section'] == '20.376.130' and publication['remote_handoff_sync_complete'] is False
repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
prefix = publication['prefix']
expected_files = {path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((repo / prefix).rglob('*')) if path.is_file()}
assert len(expected_files) == publication['new_files']
payload = dict(current_doc_sha256=publication['doc_sha256'],
    prior_doc_sha256=publication['remote_last_confirmed_sha256'],
    expected_files=expected_files, evidence_prefix=prefix)
(destination / 'REQUEST.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
code = r'''import base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main')
doc=project/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
doc_sha=hashlib.sha256(doc.read_bytes()).hexdigest()
copied={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in b['expected_files'] if (project/name).is_file()}
cpu=Path('/root/autodl-tmp/pvground_native_direct_controls_cpu_20261010')
cpu_names=('CPU_BUNDLE.json','CPU_EXIT.json','CPU_MODULE_WITNESS.json','CPU_STDOUT.json','CPU_STDERR.txt')
cpu_files={name:base64.b64encode((cpu/name).read_bytes()).decode() for name in cpu_names if (cpu/name).is_file()}
out=dict(status='ACTUAL_READ_ONLY_REMOTE_STATIC_WITNESS',time_cst=datetime.datetime.now().astimezone().isoformat(),
 remote_doc_sha256=doc_sha,document_matches_current_local=doc_sha==b['current_doc_sha256'],
 document_matches_prior_confirmed=doc_sha==b['prior_doc_sha256'],
 copied_evidence_sha256=copied,all_current_evidence_matches=copied==b['expected_files'],
 CPU_root_exists=cpu.exists(),CPU_files=cpu_files,normal_training_reads=0,GPU_calls=0,neural_calls=0,writes=0,restart_calls=0)
print(json.dumps(out))
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-v', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'ConnectTimeout=30', '-o', 'StrictHostKeyChecking=yes',
    '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
started = datetime.datetime.now().astimezone().isoformat()
response = subprocess.run(argv, env=environment, input=json.dumps(payload).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(destination / 'RAW_STDOUT.json').write_bytes(response.stdout)
(destination / 'RAW_SSH_STDERR_PRIVATE.txt').write_bytes(response.stderr)
diagnostic = response.stderr.decode(errors='replace')
phase_lines = [line for line in diagnostic.splitlines() if any(
    marker in line for marker in ('Connecting to ', 'Connection established.',
    'Remote protocol version ', 'kex_exchange_identification:', 'Connection closed by '))]
transport = dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),
    exit_code=response.returncode,stdout_bytes=len(response.stdout),phase_lines=phase_lines,
    remote_command_receipt_received=bool(response.stdout),normal_training_reads=0,
    remote_state_proven_by_transport_failure=False,restart_calls=0)
(destination / 'TRANSPORT.json').write_text(json.dumps(transport, indent=2) + '\n', encoding='utf-8')
if response.returncode != 0:
    print(json.dumps(transport))
    raise SystemExit(response.returncode)
value = json.loads(response.stdout)
for name, encoded in value.pop('CPU_files').items():
    (destination / ('REMOTE_' + name)).write_bytes(base64.b64decode(encoded))
value['CPU_receipt_files'] = sorted(json.loads(response.stdout)['CPU_files'])
(destination / 'REMOTE_STATIC_WITNESS.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key:value[key] for key in ('status','time_cst','document_matches_current_local',
    'document_matches_prior_confirmed','all_current_evidence_matches','CPU_root_exists','CPU_receipt_files','normal_training_reads')}))
