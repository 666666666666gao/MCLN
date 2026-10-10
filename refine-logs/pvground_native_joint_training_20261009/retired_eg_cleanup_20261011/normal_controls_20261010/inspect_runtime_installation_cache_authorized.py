"""Read only large cached wheels matching packages in the installed PV runtime."""
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
assert not (root/'RUNTIME_INSTALLATION_CACHE_INVENTORY.json').exists()
code=r'''import datetime,json,shutil,subprocess,sys,zipfile
from pathlib import Path
import pkg_resources
base=Path('/root/.cache/pip');files=[]
installed={p.key:p.version for p in pkg_resources.working_set}
allowed={'torch','numpy','scipy','spconv-cu111'}
if base.is_dir():
 for p in base.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.stat().st_size>=16*1024*1024 and zipfile.is_zipfile(str(p)):
   with zipfile.ZipFile(str(p)) as z:
    metadata=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
    if len(metadata)!=1:continue
    lines=z.read(metadata[0]).decode().splitlines()
    name=next(l[6:] for l in lines if l.startswith('Name: ')).lower().replace('_','-')
    version=next(l[9:] for l in lines if l.startswith('Version: '))
   if name in allowed and installed.get(name)==version:
    assert base.resolve() in p.resolve().parents
    files.append(dict(path=str(p),resolved_path=str(p.resolve()),bytes=p.stat().st_size,name=name,version=version,installed_version=installed[name],mtime_cst=datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat()))
processes=subprocess.check_output(['ps','-eo','pid=,comm=']).decode().splitlines()
managers=[int(line.split()[0]) for line in processes if line.split()[1] in ['pip','pip3','pip3.7']]
print(json.dumps(dict(status='READ_ONLY_INSTALLED_RUNTIME_MATCHING_WHEEL_CACHE_INVENTORY',time_cst=datetime.datetime.now().astimezone().isoformat(),runtime_executable=sys.executable,
 runtime_torch_version=installed.get('torch'),cache_base=str(base),cache_base_resolved=str(base.resolve()),files=files,total_candidate_bytes=sum(r['bytes'] for r in files),
 package_manager_comm_pids=managers,storage={path:dict(zip(['total','used','free'],shutil.disk_usage(path))) for path in ['/','/root/autodl-tmp']},neural_calls=0,training_status_reads=0,deletions=0)))
'''
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
env=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts','-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1','root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
r=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'RUNTIME_CACHE_INVENTORY_STDOUT.json').write_bytes(r.stdout)
(root/'RUNTIME_CACHE_INVENTORY_STDERR.txt').write_bytes(r.stderr)
(root/'RUNTIME_CACHE_INVENTORY_EXIT.json').write_text(json.dumps(dict(exit_code=r.returncode))+'\n')
assert r.returncode==0,'Inspect the preserved private cache inventory error; no deletion occurred'
value=json.loads(r.stdout)
(root/'RUNTIME_INSTALLATION_CACHE_INVENTORY.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
