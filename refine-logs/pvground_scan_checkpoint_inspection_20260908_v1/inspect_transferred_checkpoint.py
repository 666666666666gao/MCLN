from pathlib import Path
import collections,datetime,hashlib,json,os,torch
root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_bytes())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
destination=root/plan['filename']
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
