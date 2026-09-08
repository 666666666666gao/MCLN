"""Stage fixed source-reading replay and full-native-loss updates on existing fit fixtures."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_observation_interface_20260909_v1'
archive=repo/'refine-logs/pvground_observation_interface_20260909_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
model_source='/root/autodl-tmp/mcln_pvground_observation_source_20260909_v1/PV-Ground'
fixtures='/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1/fixtures'

def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
for dependency in ['mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1','mcln_pvground_scanrefer_endpoint_audit_20260908_sourcequery_v1','mcln_pvground_scanrefer_formal_20260908_sourcequery_v1']:
    with s.open('/root/autodl-tmp/'+dependency+'/controller.exit') as f:assert f.read().strip()==b'0'
with s.open(runtime+'/env_spec.json','rb') as f:env=json.loads(f.read())
env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
with s.open(model_source+'/../source_port.json','rb') as f:port_raw=f.read();port=json.loads(port_raw)
assert port['parent_unchanged'] and port['runtime_unchanged']
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0,err.read().decode()
original=(repo/'scripts/check_pvground_training_interface.py').read_bytes()
code=original.decode()
code=replace_once(code,'root = args.runtime.resolve()', 'root = args.runtime.resolve()\n    model_source = Path('+repr(model_source)+')')
start=code.index("    port = json.loads((root / 'source_port.json')")
end=code.index('    fixture_receipt =',start)
code=code[:start]+"    port = json.loads((model_source.parent/'source_port.json').read_bytes())\n    assert sha(model_source.parent/'source_port.json')=="+repr(hashlib.sha256(port_raw).hexdigest())+"\n    for name,digest in port['files'].items():assert sha(model_source/name)==digest,name\n"+code[end:]
code=code.replace("root / 'PV-Ground'",'model_source')
code=replace_once(code,'    model.cuda()',
    "    from pvground_observation_query import install_observation_query_read\n    install_observation_query_read(model)\n    native_state_count=len(state)\n    added=set(model.state_dict())-set(state)\n    assert added and all(k.startswith('decoder.5.source_query_read.') for k in added)\n    state.update({k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in added})\n    reader=model.decoder[-1].source_query_read\n    model.cuda()")
helper='''    replay_records=[]
    def rng_state():
        return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def restore_rng(saved):
        random.setstate(saved[0]);np.random.set_state(saved[1])
        torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])
    def same_rng(a,b):
        return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1])
                and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])
                and all(torch.equal(x,y) for x,y in zip(a[3],b[3])))
    def paired_replay(inputs,label):
        rng=rng_state()
        buffers={k:v.detach().clone() for k,v in model.named_buffers()}
        source_outputs=[]
        handle=reader.register_forward_hook(lambda module,args,value:source_outputs.append(float(value.detach().abs().max())))
        with torch.no_grad():
            reader.enabled=False
            native=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
            native_end=rng_state()
            for k,v in model.named_buffers():v.copy_(buffers[k])
            restore_rng(rng);reader.enabled=True
            output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
        handle.remove()
        assert source_outputs==[0.0,0.0]
        assert same_rng(native_end,rng_state())
        rec_keys=['source_features','seed_features','query_points_xyz','query_points_feature',
                  'last_center','last_pred_size','last_sem_cls_scores','last_proj_queries','proj_tokens']
        for key in rec_keys:assert torch.equal(output[key],native[key]),(label,key)
        # The disabled/disabled diagnostic already measured pre-existing superpoint
        # and Mask numeric variation. Record it; do not claim full-Mask bitwise replay.
        states=output['source_observations']
        assert [s.shape[-1] for s in states]==[10,13,13,13,13,13]
        assert all(torch.isfinite(s).all() and not s.requires_grad for s in states)
        state_summary=[dict(shape=list(s.shape),first_radius_empty=int((s[:,:,0]==0).sum()),second_radius_empty=int((s[:,:,6]==0).sum())) for s in states[1:]]
        mask_max={key:max(float((a-b).abs().max()) for a,b in zip(output[key],native[key]))
                  for key in ['sp_last_pred_masks','last_pred_masks','super_xyz_list']}
        replay_records.append(dict(mode=label,rec_keys_exact=rec_keys,rng_end_equal=True,
            source_output_max=source_outputs,mask_max_abs=mask_max,spatial_observation_summary=state_summary,bev_observation_shape=list(states[0].shape)))
        return output

'''
code=replace_once(code,'    evaluator = GroundingEvaluator',helper+'    evaluator = GroundingEvaluator')
code=replace_once(code,'            output = model(inputs)',"            output = paired_replay(inputs,'eval_'+str(start))")
train_replay='''    training_rng=rng_state()
    model.train()
    replay_inputs,replay_batch=batch_at(0,train_owner)
    before_rng=rng_state()
    layer=model.decoder[-1];original_forward=layer.forward
    captured={}
    def capture_forward(*args,**kwargs):
        captured['args']=tuple(v.detach().clone() if torch.is_tensor(v) else v for v in args)
        captured['kwargs']={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in kwargs.items()}
        return original_forward(*args,**kwargs)
    layer.forward=capture_forward
    # First hold the reader disabled twice: quantify upstream training-mode repeat
    # differences before claiming anything about the new last-layer computation.
    upstream=[];upstream_rng=[]
    with torch.no_grad():
        for repeat in range(2):
            model.load_state_dict(state,strict=True);restore_rng(before_rng);reader.enabled=False
            replay_output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in replay_inputs.items()})
            upstream.append(replay_output['source_features'].detach().clone())
            upstream_rng.append(rng_state())
            del replay_output
    layer.forward=original_forward
    assert same_rng(upstream_rng[0],upstream_rng[1])
    upstream_difference=float((upstream[0]-upstream[1]).abs().max())
    del upstream
    # The same real layer inputs remove upstream sparse/BN numerical differences
    # from the causal question of inserting a zero residual before native dropout.
    layer_rng=rng_state();layer_outputs=[];layer_end_rng=[];source_outputs=[]
    handle=reader.register_forward_hook(lambda module,args,value:source_outputs.append(float(value.detach().abs().max())))
    with torch.no_grad():
        for enabled in [False,True]:
            model.load_state_dict(state,strict=True);restore_rng(layer_rng);reader.enabled=enabled
            layer_outputs.append(original_forward(*captured['args'],**captured['kwargs']).detach().clone())
            layer_end_rng.append(rng_state())
    handle.remove()
    assert torch.equal(layer_outputs[0],layer_outputs[1])
    assert same_rng(layer_end_rng[0],layer_end_rng[1])
    assert source_outputs==[0.0,0.0]
    train_layer_replay=dict(same_actual_inputs=True,output_exact=True,rng_end_equal=True,
        disabled_upstream_repeat_max_abs=upstream_difference,full_train_model_bitwise_claim=False,
        full_model_forwards=2,extra_last_layer_forwards=2,source_output_max=source_outputs)
    del captured,layer_outputs,replay_inputs,replay_batch
    model.load_state_dict(state,strict=True);reader.enabled=True
    restore_rng(training_rng)
    training = copy.copy(config)'''
code=replace_once(code,'    training = copy.copy(config)',train_replay)
code=replace_once(code,'        groups = defaultdict',
    "        source_grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in reader.named_parameters()}\n        assert source_grads['output.weight']>0\n        if step==2:\n            assert source_grads['attention.in_proj_weight']>0\n            assert all(source_grads['projections.%d.weight'%i]>0 for i in range(6))\n        if step==2:\n            assert all(source_grads['observation_keys.%d'%i]>0 and source_grads['observation_values.%d'%i]>0 for i in range(6))\n        groups = defaultdict")
code=replace_once(code,"            'all_present_gradients_finite':True,", "            'source_gradient_norms':source_grads,'all_present_gradients_finite':True,")
code=replace_once(code,"        'strict_state_tensors':len(state)","        'strict_state_tensors':native_state_count,'added_state_tensors':len(added)")
code=replace_once(code,"        'eval_forwards':2,'train_forwards':2", "        'eval_forwards':4,'train_forwards':4")
code=replace_once(code,"    (args.output/'receipt.json').write_text",
    "    receipt.update(source_query_read=True,observation_state=True,module_sha256=sha(Path(__file__).parent/'pvground_observation_query.py'),model_source=str(model_source),source_port_sha256=sha(model_source.parent/'source_port.json'),zero_residual_eval_rec_replay_exact=True,train_layer_replay=train_layer_replay,replay_records=replay_records,mask_bitwise_equality_claim=False,new_parameters=sum(p.numel() for p in reader.parameters()))\n    (args.output/'receipt.json').write_text")
code=code.replace('PVG_TRAINING_INTERFACE_PASS','PVG_SOURCE_QUERY_INTERFACE_PASS')
argv=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',runtime+'/venv/bin/python','-u',root+'/check.py','--runtime',runtime,'--fixtures',fixtures,'--output',root+'/results']
controller=('import hashlib,json,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
            '(root/"controller.pid").write_text(str(os.getpid())+"\\n")\n'
            'spec=json.loads((root/"spec.json").read_bytes())\n'
            'for name,digest in spec["files"].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n'
            'env=os.environ.copy();env.update('+repr(env['env'])+')\n'
            'with (root/"run.log").open("xb") as log:\n'
            '    result=subprocess.run('+repr(argv)+',env=env,stdout=log,stderr=subprocess.STDOUT)\n'
            '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\nraise SystemExit(result.returncode)\n')
files={'check.py':code.encode(),'pvground_source_query.py':(repo/'models/pvground_source_query.py').read_bytes(),'pvground_observation_query.py':(repo/'models/pvground_observation_query.py').read_bytes(),'controller.py':controller.encode(),'source_port.json':port_raw,'plan.md':(repo/'docs/PVG_OBSERVATION_QUERY_CONTROL_2026-09-09.md').read_bytes()}
spec=dict(root=root,runtime=runtime,model_source=model_source,env_spec_sha256=env_sha,seed=2027,eval_forwards=4,train_forwards=4,optimizer_steps=2,new_checkpoints=0,formal_rows=0,files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
command=['screen','-dmS','mcln_pvg_observation_interface_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,command)),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,root=root,expected_seconds=120,formal_rows=0)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wb') as f:f.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
