"""Read only the isolated CPU artifact root using the proven SSH route."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
f = root/'face_residual_preparation_20261010'
assert not (f/'STATIC_CPU_ROOT_READ.json').exists()
witness = json.loads((root.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
code = "import json; from pathlib import Path; p=Path('/root/autodl-tmp/pvground_face_residual_cpu_20261010'); print(json.dumps(dict(root=str(p),exists=p.exists(),CPU_execution_receipt_exists=(p/'CPU_EXECUTION.json').is_file(),training_status_queries=0)))"
argv = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u','-c',code])]
response = subprocess.run(argv,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    creationflags=subprocess.CREATE_NO_WINDOW)
(root/'FACE_CPU_STATIC_ROOT_PRIVATE_STDERR.txt').write_bytes(response.stderr)
(f/'STATIC_CPU_ROOT_TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode == 0
record = json.loads(response.stdout)
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(),unchanged_host_key_route_verified=True)
(f/'STATIC_CPU_ROOT_READ.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
