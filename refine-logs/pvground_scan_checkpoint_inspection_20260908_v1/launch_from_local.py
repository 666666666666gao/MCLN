import datetime, hashlib, json, os, shlex
from pathlib import Path
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo/'refine-logs/pvground_scan_checkpoint_inspection_20260908_v1'
remote = '/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1'
resources = json.loads((repo/'refine-logs/pvground_pretrained_resources_20260908_v1/receipt.json').read_bytes())
entry = next(v for v in resources['model_files'] if v['rfilename'] == 'PV-Ground_ScanRefer.pth')
plan = {
    'upstream': 'https://github.com/AaNnWwTt/PV-Ground',
    'github_commit': resources['github_commit'], 'huggingface_revision': resources['huggingface_revision'],
    'url': 'https://huggingface.co/AaNnWwTt/PV-Ground/resolve/'+resources['huggingface_revision']+'/'+entry['rfilename'],
    'filename': entry['rfilename'], 'bytes': entry['size'], 'sha256': entry['lfs']['sha256'],
    'purpose': 'CPU inventory of official full ScanRefer checkpoint for pretrained geometry/decoder reuse; no model forward, new training or formal evaluation',
    'do_not_replace_protected_model': True, 'dataset': 'ScanRefer',
}
script = '''import collections, datetime, hashlib, json, os, shutil, time, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_bytes())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
started=time.time()
before=shutil.disk_usage(str(root)).free
assert before > plan['bytes'] + 2*1024**3
destination=root/plan['filename']
part=root/(plan['filename']+'.part')
assert not destination.exists() and not part.exists()
with urllib.request.urlopen(plan['url'],timeout=60) as response, part.open('xb') as target:
    status=response.status
    assert status==200
    digest=hashlib.sha256(); size=0
    for chunk in iter(lambda:response.read(8*1024*1024), b''):
        target.write(chunk); digest.update(chunk); size+=len(chunk)
assert size==plan['bytes'] and digest.hexdigest()==plan['sha256']
part.rename(destination)
download={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'bytes':size,'sha256':digest.hexdigest(),'elapsed_seconds':time.time()-started,'free_before':before,'free_after':shutil.disk_usage(str(root)).free,'file':str(destination),'url':plan['url']}
(root/'download_receipt.json').write_text(json.dumps(download,indent=2)+'\\n')
print('PV SCAN CHECKPOINT DOWNLOADED '+json.dumps(download),flush=True)
import torch
payload=torch.load(str(destination),map_location='cpu')
assert isinstance(payload,dict)
print('PV SCAN PAYLOAD KEYS '+json.dumps(sorted(payload)),flush=True)
state=payload['model']
assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
prefixes=collections.defaultdict(lambda:{'tensors':0,'elements':0})
inventory={}
for name,tensor in state.items():
    assert torch.isfinite(tensor).all(),name
    prefix=name.split('.')[1] if name.startswith('module.') else name.split('.')[0]
    prefixes[prefix]['tensors']+=1;prefixes[prefix]['elements']+=tensor.numel()
    inventory[name]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),'elements':tensor.numel()}
receipt={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'checkpoint_sha256':plan['sha256'],'payload_keys':sorted(payload),'model_tensor_count':len(state),'parameter_prefixes':dict(prefixes),'torch_version':torch.__version__,'gpu_forwards':0,'optimizer_steps':0,'formal_rows':0,'not_our_model_result':True}
(root/'state_inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\\n')
receipt['state_inventory_sha256']=hashlib.sha256((root/'state_inventory.json').read_bytes()).hexdigest()
(root/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\\n')
print('PV SCAN CPU INVENTORY COMPLETE '+json.dumps(receipt),flush=True)
'''
controller = '''#!/usr/bin/env bash
set -euo pipefail
cd ROOT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'printf "%s\\n" "$?" > controller.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -u inspect_checkpoint.py
'''.replace('ROOT', shlex.quote(remote))
files = {'plan.json':(json.dumps(plan,indent=2)+'\n').encode(), 'inspect_checkpoint.py':script.encode(),
         'controller.sh':controller.encode(), 'launch_from_local.py':Path(__file__).read_bytes()}
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s=c.open_sftp(); archive.mkdir(); s.mkdir(remote)
for name, blob in files.items():
    (archive/name).write_bytes(blob)
    with s.open(remote+'/'+name,'wx') as stream: stream.write(blob)
    with s.open(remote+'/'+name,'rb') as stream: assert stream.read()==blob
command='exec bash '+shlex.quote(remote+'/controller.sh')+' > '+shlex.quote(remote+'/run.log')+' 2>&1'
_,stdout,stderr=c.exec_command('screen -dmS mcln_pv_scan_inventory_v1 bash -c '+shlex.quote(command),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=c.exec_command('screen -ls',timeout=30)
text=stdout.read().decode(); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
matches=[line.strip() for line in text.splitlines() if '.mcln_pv_scan_inventory_v1' in line]
assert len(matches)==1
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'screen':matches[0],
        'plan_sha256':hashlib.sha256(files['plan.json']).hexdigest(),'estimated_duration_seconds':300,'checkpoint_download_complete':False,'model_inspection_complete':False}
blob=(json.dumps(launch,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(blob)
with s.open(remote+'/launch.json','wx') as stream:stream.write(blob)
s.close();c.close();print(json.dumps(launch),flush=True)
