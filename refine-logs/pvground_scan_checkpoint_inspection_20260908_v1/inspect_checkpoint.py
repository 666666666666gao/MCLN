import collections, datetime, hashlib, json, os, shutil, time, urllib.request
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
(root/'download_receipt.json').write_text(json.dumps(download,indent=2)+'\n')
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
(root/'state_inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
receipt['state_inventory_sha256']=hashlib.sha256((root/'state_inventory.json').read_bytes()).hexdigest()
(root/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('PV SCAN CPU INVENTORY COMPLETE '+json.dumps(receipt),flush=True)
