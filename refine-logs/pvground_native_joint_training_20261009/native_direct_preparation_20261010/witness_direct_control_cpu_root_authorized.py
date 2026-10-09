"""Read only the isolated CPU receipt after a transport failure; no run restart."""
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
assert json.loads((controls / 'CPU_TRANSPORT_EXIT.json').read_bytes())['exit_code'] == 255
assert not (controls / 'CPU_STATIC_ROOT_READ_WITNESS.json').exists()
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
code = r'''import base64,datetime,json
from pathlib import Path
root=Path('/root/autodl-tmp/pvground_native_direct_controls_cpu_20261010')
names=('CPU_EXIT.json','CPU_MODULE_WITNESS.json','CPU_STDOUT.json','CPU_STDERR.txt')
files={name:base64.b64encode((root/name).read_bytes()).decode() for name in names if (root/name).is_file()}
print(json.dumps(dict(status='ISOLATED_CPU_ROOT_READ_ONLY',time_cst=datetime.datetime.now().astimezone().isoformat(),
 root=str(root),exists=root.exists(),files=files,normal_training_reads=0,GPU_calls=0,neural_calls=0,restart_calls=0)))
'''
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
response = subprocess.run(argv, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=subprocess.CREATE_NO_WINDOW)
(controls / 'CPU_STATIC_ROOT_READ_STDOUT.json').write_bytes(response.stdout)
(controls / 'CPU_STATIC_ROOT_READ_STDERR.txt').write_bytes(response.stderr)
(controls / 'CPU_STATIC_ROOT_READ_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.returncode == 0, response.stderr.decode()
value = json.loads(response.stdout)
for name, encoded in value.pop('files').items():
    (controls / ('RECOVERED_REMOTE_' + name)).write_bytes(base64.b64decode(encoded))
value['recovered_receipt_files'] = sorted(json.loads(response.stdout)['files'])
(controls / 'CPU_STATIC_ROOT_READ_WITNESS.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps(value))
