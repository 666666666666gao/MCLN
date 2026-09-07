import hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/scanrefer_object_appearance_pair_20260908_v1';archive.mkdir()
root='/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1'
native='/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1'
cache='/root/autodl-tmp/mcln_scanrefer_openshape_cache_20260908_v1'
base=json.loads((repo/'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/input_manifest.json').read_bytes())
keys=['artifacts','batch_size','clip_norm','core_learning_rate','data_root','epochs','loss','readouts_frozen','split_protocol','split_protocol_sha256','split_salt','steps_per_arm','superpoint_files','weight_decay']
m={k:base[k] for k in keys}
m.update(schema='mcln-scanrefer-object-appearance-pair-input-v1',mode='train',appearance_learning_rate=1e-4,
         model_source=native+'/model_source',source_manifest_sha256='190d0011bc5bfefab4a3965d972606f4dc21a946f68b2382f5a90837b3a4c4ef',
         native_probe_receipt=native+'/receipt.json',native_probe_receipt_sha256=hashlib.sha256((repo/'refine-logs/scanrefer_object_appearance_native_20260908_v1/receipt.json').read_bytes()).hexdigest(),
         cache_root=cache,cache_receipt_sha256=hashlib.sha256((repo/'refine-logs/scanrefer_openshape_cache_20260908_v1/receipt.json').read_bytes()).hexdigest(),
         plan_sha256=hashlib.sha256((repo/'docs/SCANREFER_OBJECT_APPEARANCE_PAIR_PLAN_2026-09-08.md').read_bytes()).hexdigest(),
         core_prefixes=['cross_encoder.','decoder.','prediction_heads.'],formal_rows=0)
files=['scripts/run_scanrefer_object_appearance_pair.py','scripts/scanrefer_joint_readout.py','scripts/scanrefer_data_contract.py','scripts/scanrefer_rec_evaluation.py']
m['files']={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in files}
(archive/'input_manifest.json').write_bytes(json.dumps(m,indent=2).encode()+b'\n')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp()
with s.open(native+'/receipt.json','rb') as f:assert hashlib.sha256(f.read()).hexdigest()==m['native_probe_receipt_sha256']
_,o,e=c.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits');used=int(o.read().decode());assert used<500,used
s.mkdir(root);s.mkdir(root+'/scripts')
for name in files:s.put(str(repo/name),root+'/'+name)
s.put(str(archive/'input_manifest.json'),root+'/input_manifest.json')
s.put(str(repo/'docs/SCANREFER_OBJECT_APPEARANCE_PAIR_PLAN_2026-09-08.md'),root+'/plan.md')
shell='cd '+shlex.quote(root)+'\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0\nflock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_object_appearance_pair.py --manifest '+shlex.quote(root+'/input_manifest.json')+' > controller.log 2>&1\ncode=$?\nprintf "%s\\n" "$code" > controller.exit\nexit "$code"\n'
(archive/'run.sh').write_bytes(shell.encode())
with s.open(root+'/run.sh','wb') as f:f.write(shell.encode())
_,o,e=c.exec_command('screen -dmS mcln_os_appearance_pair_v1 bash '+shlex.quote(root+'/run.sh'))
assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command('screen -ls');screens=o.read().decode();print(screens)
(archive/'launch.json').write_bytes(json.dumps(dict(root=root,screen_output=screens,prelaunch_gpu_mib=used,phase='launched; capacity and baseline precede first optimizer update'),indent=2).encode()+b'\n')
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close()
