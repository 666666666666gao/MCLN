"""Read existing dataset cache sizes and host RAM; no model, job or GPU query."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
spec=json.loads((root/'CHECK_SPEC.json').read_bytes())
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
remote_code=r'''import datetime,json
from pathlib import Path
data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemTotal','MemAvailable')}
files={split:dict(path=str(data/(split+'_v3scans.pkl')),bytes=(data/(split+'_v3scans.pkl')).stat().st_size) for split in ('train','val')}
print(json.dumps(dict(observed_cst=datetime.datetime.now().astimezone().isoformat(),memory=memory,pickles=files,current_training_queries=0,model_calls=0)))
'''
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
environment=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',remote_code])]
response=subprocess.run(argv,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'RESOURCE_STDOUT.json').write_bytes(response.stdout)
(root/'RESOURCE_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root/'RESOURCE_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode==0
intake=json.loads(response.stdout)
intake.update(local_received_cst=datetime.datetime.now().astimezone().isoformat(),admitted_by_executor=False)
(root/'RESOURCE_INTAKE.json').write_text(json.dumps(intake,indent=2)+'\n')
print(json.dumps(intake))
