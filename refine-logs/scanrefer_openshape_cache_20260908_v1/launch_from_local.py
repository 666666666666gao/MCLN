import hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/scanrefer_openshape_cache_20260908_v1'
archive.mkdir()
runtime='/root/autodl-tmp/mcln_openshape_object_source_20260907_v1'
root='/root/autodl-tmp/mcln_scanrefer_openshape_cache_20260908_v1'
prior=repo/'refine-logs/pretrained_object_source_feasibility_20260907_v1'
assert json.loads((prior/'runtime_review.json').read_bytes())['decision']=='PASS'
assert (prior/'probe.exit').read_text().strip()=='0'
assert json.loads((prior/'feature_separation_analysis.json').read_bytes())['repeat_nearest_same_scene_correct']==181
config=json.loads((prior/'input_manifest.json').read_bytes())
spec={k:config[k] for k in ['model_source','source_manifest_sha256','data_root']}
spec.update(runtime_root=runtime,runtime_spec_sha256='fe8ac66b8f51ed0e179a279717d0d1c2213e9a6b463785ccaa6a6e5eaf2385ac',
            scene_list=config['data_root']+'scanrefer/ScanRefer_filtered_train.txt',scene_list_sha256='88b7476a84a6e0bc1d44cb4fa191e00ef0582090c38b3b68097d3ecf37ce5472',scene_count=562,
            feature_source='official frozen OpenShape G14 RGB',optimizer_steps=0,formal_rows=0)
script=repo/'scripts/cache_scanrefer_openshape_objects.py'
spec['script_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
(archive/'manifest.json').write_bytes(json.dumps(spec,indent=2).encode()+b'\n')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
_,o,e=c.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits')
gpu=int(o.read().decode().strip());assert o.channel.recv_exit_status()==0 and gpu<500,gpu
s.mkdir(root);s.mkdir(root+'/features')
s.put(str(script),root+'/cache.py');s.put(str(archive/'manifest.json'),root+'/manifest.json')
shell='cd '+shlex.quote(root)+'\nexport PYTHONPATH='+shlex.quote(runtime+'/vendor:'+runtime+'/deps')+' DGLBACKEND=pytorch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0\nflock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u cache.py --manifest '+shlex.quote(root+'/manifest.json')+' > cache.log 2>&1\ncode=$?\nprintf "%s\\n" "$code" > cache.exit\nexit "$code"\n'
(archive/'run.sh').write_bytes(shell.encode())
with s.open(root+'/run.sh','wb') as f:f.write(shell.encode())
_,o,e=c.exec_command('screen -dmS mcln_os_scan_cache_v1 bash '+shlex.quote(root+'/run.sh'))
assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command('screen -ls');screen=o.read().decode();print(screen)
(archive/'launch.json').write_bytes(json.dumps(dict(root=root,screen_output=screen,prelaunch_gpu_mib=gpu,manifest_sha256=hashlib.sha256((archive/'manifest.json').read_bytes()).hexdigest(),training_started=False),indent=2).encode()+b'\n')
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close()
