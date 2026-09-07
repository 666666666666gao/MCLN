import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/mask_geometry_initialization_preparation_20260907_v1'
remote='/root/autodl-tmp/mcln_mask_geometry_initialization_preparation_20260907_v1'
local.mkdir(exist_ok=False);(local/'scripts').mkdir();(local/'tests').mkdir()
old=json.loads((repo/'refine-logs/native_box_initialization_preparation_20260907_v1/input_manifest.json').read_bytes())
manifest={key:old[key] for key in ['native_preparation','scan_backbone','scan_source','scan_source_manifest_sha256']}
manifest.update(schema='mcln-mask-geometry-initialization-preparation-v1',
    training_directory='/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1',
    training_manifest_sha256='15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3',
    synthetic_fixture_only=True,actual_trained_endpoint_loaded=False,gpu_forwards=0,optimizer_steps=0,
    temporary_checkpoint_write_expected=True,temporary_checkpoint_deleted_after_verify=True)
files={}
for name in ['scripts/export_mask_geometry_initialization.py','tests/test_mask_geometry_initialization.py']:
    raw=(repo/name).read_bytes();(local/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
(local/'scripts/__init__.py').write_bytes(b'');files['scripts/__init__.py']=hashlib.sha256(b'').hexdigest()
raw=Path('C:/Users/gb/.codex/tmp/check_mask_geometry_initialization_cpu_20260907.py').read_bytes()
(local/'check_mask_geometry_initialization_cpu.py').write_bytes(raw)
files['check_mask_geometry_initialization_cpu.py']=hashlib.sha256(raw).hexdigest();manifest['files']=files
(local/'input_manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests > cpu_tests.txt 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  /root/miniconda3/envs/bdetr/bin/python -u check_mask_geometry_initialization_cpu.py > cpu_load_stdout.txt 2> cpu_load_stderr.txt
  status=$?
fi
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.format(remote=remote)
(local/'controller.sh').write_bytes(controller.encode())
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(remote);s.mkdir(remote+'/scripts');s.mkdir(remote+'/tests')
for p in local.rglob('*'):
    if p.is_file():s.put(str(p),remote+'/'+p.relative_to(local).as_posix())
check="import ast,hashlib,json,shutil;from pathlib import Path;p=Path("+repr(remote)+");m=json.loads((p/'input_manifest.json').read_text());[ast.parse((p/n).read_text()) for n in m['files'] if n.endswith('.py')];assert shutil.disk_usage('/root/autodl-tmp').free>2*1024**3;print('OriginalPython syntax and disk check pass')"
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check)+' && bash -n '+shlex.quote(remote+'/controller.sh'))
body=o.read().decode();assert o.channel.recv_exit_status()==0,e.read().decode();print(body,flush=True)
command='screen -dmS mcln_mask_geometry_native_cpu_v1 bash -c '+shlex.quote('exec bash '+remote+'/controller.sh > '+remote+'/run.log 2>&1')
_,o,e=c.exec_command(command);assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command("screen -ls\nps -eo pid,ppid,comm,etime,args | grep '[m]cln_mask_geometry_initialization_preparation_20260907_v1/controller.sh'",timeout=30)
live=o.read().decode();assert 'mcln_mask_geometry_native_cpu_v1' in live
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'remote_directory':remote,'screen':'mcln_mask_geometry_native_cpu_v1','live_processes':live,'command':command,
    'manifest_sha256':hashlib.sha256((local/'input_manifest.json').read_bytes()).hexdigest(),
    'training_sources_not_modified':True,'new_gpu_jobs':0,'scope':'CPU exporter/loader format check only'}
(local/'launch.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());(local/'stage_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close();print(json.dumps(record),flush=True)
