"""Check only static publication state after an observed transport failure."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
assert json.loads((root / 'REFERIT_NATIVE_CPU_REMOTE_EXIT.json').read_bytes())['exit_code'] == 255
assert (root / 'REFERIT_NATIVE_CPU_REMOTE_STDOUT.json').read_bytes() == b''
target = root / 'REFERIT_NATIVE_CPU_STATIC_RECEIPT.json'
assert not target.exists()
prior = json.loads((root / 'native_control_cpu_closure_publication.json').read_bytes())
code = r'''import hashlib,json
from pathlib import Path
p=Path('/home/gb/new butd/butd_detr-main/MCLN-main')
d=p/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
leaf=p/'refine-logs/pvground_native_joint_training_20261009/referit_native_source_cpu_20261010'
print(json.dumps(dict(doc_sha256=hashlib.sha256(d.read_bytes()).hexdigest(),evidence_leaf_exists=leaf.exists(),normal_training_queries=0,GPU_calls=0,neural_calls=0,writes=0)))
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
response = subprocess.run(argv, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'REFERIT_NATIVE_CPU_STATIC_STDOUT.json').write_bytes(response.stdout)
(root / 'REFERIT_NATIVE_CPU_STATIC_STDERR.txt').write_bytes(response.stderr)
(root / 'REFERIT_NATIVE_CPU_STATIC_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n')
assert response.returncode == 0, 'Static read failed; preserve receipt without retry.'
receipt = json.loads(response.stdout)
assert receipt['doc_sha256'] == prior['doc_sha256'] and receipt['evidence_leaf_exists'] is False
receipt.update(time_cst=datetime.datetime.now().astimezone().isoformat(), prior_static_state_confirmed=True)
target.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
