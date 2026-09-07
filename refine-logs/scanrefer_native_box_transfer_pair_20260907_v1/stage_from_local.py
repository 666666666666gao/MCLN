import hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1'
local.mkdir(exist_ok=False);(local/'scripts').mkdir();(local/'tests').mkdir()
base=json.loads((repo/'refine-logs/scanrefer_native_box_transfer_probe_20260907_v1/input_manifest.json').read_bytes())
names=['scripts/run_scanrefer_native_box_transfer_pair.py','scripts/audit_scanrefer_native_box_transfer_pair.py',
       'scripts/native_teacher_box_transfer.py','scripts/scanrefer_joint_readout.py',
       'scripts/scanrefer_rec_evaluation.py','scripts/scanrefer_data_contract.py',
       'scripts/audit_scanrefer_joint_readout_pair.py','tests/test_native_teacher_box_transfer.py']
files={}
for name in names:
 raw=(repo/name).read_bytes();(local/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
(local/'scripts/__init__.py').write_bytes(b'');files['scripts/__init__.py']=hashlib.sha256(b'').hexdigest()
manifest={k:base[k] for k in ['model_source','source_manifest_sha256','artifacts','data_root','train_superpoint_files','split_salt','split_protocol','split_protocol_sha256','environment_reuse']}
probe=repo/'refine-logs/scanrefer_native_box_transfer_probe_20260907_v1/receipt.json'
assert json.loads(probe.read_bytes())['status']=='pass'
manifest.update(schema='mcln-scanrefer-native-box-transfer-training-input-v1',
 core_learning_rate=1e-6,teacher_box_loss_weight=1.,weight_decay=.0005,clip_norm=.1,
 files=files,native_probe_receipt='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_probe_20260907_v1/receipt.json',
 native_probe_receipt_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
 epochs=1,batch_size=12,steps_per_arm=2482,fit_rows=29778,holdout_rows=6887,formal_rows=0,
 readouts_frozen=True,teacher_frozen=True,model_mode='eval; only final center/size head parameters train',
 candidate_predeclared='gt_teacher_box',control='gt_only',save_policy='Two fixed 16-parameter endpoints plus optimizer; preserve protected E71 for exact reconstruction',
 selection='Existing GT-free native decision and frozen V99 system; report separately',
 module_gate='System REC025 and REC050 no lower than identical baseline and GT-only control; then fixed formal ScanRefer check',
 current_formal_targets={'scanrefer_rec_floor':[58.6033,50.4523],'scanrefer_rec_stretch':[59.,51.],
 'scanrefer_mask_floor':[58.70,50.70,44.72],'nr3d_rec_floor':[59.82,51.38],'sr3d_rec_floor':[68.43,57.30],
 'nr3d_sr3d_mask_gate':False},purpose='Transfer actual GT-supported V99 teacher geometry into existing native regression heads; no new reranker or backbone replacement')
(local/'input_manifest.json').write_bytes((json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_native_box_transfer_pair.py --manifest {remote}/input_manifest.json
status=$?
printf '%s\\n' "$status" > training.exit
if [ "$status" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python -u scripts/audit_scanrefer_native_box_transfer_pair.py --manifest {remote}/input_manifest.json
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
command='cd '+shlex.quote(manifest['model_source'])+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(remote+':'+manifest['model_source'])+' /root/miniconda3/envs/bdetr/bin/python -m pytest -q '+shlex.quote(remote+'/tests/test_native_teacher_box_transfer.py')
_,o,e=c.exec_command(command);output=o.read().decode()+e.read().decode();assert o.channel.recv_exit_status()==0,output
(local/'cpu_tests.txt').write_text(output,encoding='utf-8');print(output,flush=True)
check="import ast,json;from pathlib import Path;p=Path("+repr(remote)+");m=json.loads((p/'input_manifest.json').read_text());[ast.parse((p/n).read_text()) for n in m['files'] if n.endswith('.py')];print('All frozen Python files parse under original Python 3.7')"
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check)+' && bash -n '+shlex.quote(remote+'/controller.sh'))
result=o.read().decode()+e.read().decode();assert o.channel.recv_exit_status()==0,result
(local/'original_environment_syntax.txt').write_text(result,encoding='utf-8');print(result,flush=True)
(local/'stage_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close()
print('STAGED; NOT LAUNCHED',remote,flush=True)
