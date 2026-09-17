"""Run D's CUDA routing checks and real fit-fixture native-loss interface probe."""
import datetime, hashlib, json, os, shlex
from pathlib import Path
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_task_observation_interface_20260917_v1'
archive = repo/'refine-logs/pvground_task_observation_interface_20260917_v1'
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
model_source = '/root/autodl-tmp/mcln_pvground_task_observation_source_20260917_v1/PV-Ground'
fixtures = '/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1/fixtures'

def replace_once(text, old, new):
    assert text.count(old)==1, old
    return text.replace(old,new)

c=paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
for dependency in ['scanrefer_finetune','scanrefer_endpoint_audit','scanrefer_formal']:
    with s.open('/root/autodl-tmp/mcln_pvground_'+dependency+'_20260909_observation_v1/controller.exit') as f:
        assert f.read().strip()==b'0'
with s.open(runtime+'/env_spec.json','rb') as f: env=json.loads(f.read())
env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
with s.open(model_source+'/../source_port.json','rb') as f: port_raw=f.read()
assert json.loads(port_raw)['parent_unchanged']
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0,err.read().decode()
code=(repo/'scripts/check_pvground_training_interface.py').read_text(encoding='utf-8')
code=replace_once(code,'root = args.runtime.resolve()', 'root = args.runtime.resolve()\n    model_source = Path('+repr(model_source)+')')
start=code.index("    port = json.loads((root / 'source_port.json')")
end=code.index('    fixture_receipt =',start)
code=code[:start]+"    port = json.loads((model_source.parent/'source_port.json').read_bytes())\n    assert sha(model_source.parent/'source_port.json')=="+repr(hashlib.sha256(port_raw).hexdigest())+"\n    for name,digest in port['files'].items():assert sha(model_source/name)==digest,name\n"+code[end:]
code=code.replace("root / 'PV-Ground'",'model_source')
code=replace_once(code,'    model.cuda()',
    "    from pvground_task_observation_query import install_task_observation_query_read\n"
    "    from pvground_observation_query import ObservationQueryRead\n"
    "    install_task_observation_query_read(model)\n"
    "    native_state_count=len(state)\n"
    "    added=set(model.state_dict())-set(state)\n"
    "    assert len(added)==37 and all(k.startswith('decoder.5.source_query_read.') for k in added)\n"
    "    state.update({k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in added})\n"
    "    reader=model.decoder[-1].source_query_read\n    model.cuda()")
helper='''    replay_records=[]
    routes={}
    def capture_layer(module,args,value):
        if isinstance(value,tuple): routes['semantic'],routes['geometry']=value
    route_handles=[model.decoder[-1].register_forward_hook(capture_layer)]
    def capture(name):
        def hook(module,args): routes[name]=args[0]
        return hook
    for name,module in [('sem_head',model.prediction_heads[-1].sem_cls_scores_head),
                        ('center_head',model.prediction_heads[-1].center_residual_head),
                        ('size_head',model.prediction_heads[-1].size_pred_head),
                        ('query_mask',model.x_query),('contrastive',model.contrastive_align_projection_image)]:
        route_handles.append(module.register_forward_pre_hook(capture(name)))
    def verify_routes():
        assert torch.equal(routes['sem_head'],routes['semantic'].transpose(1,2))
        assert torch.equal(routes['contrastive'],routes['semantic'])
        for name in ['center_head','size_head','query_mask']:
            assert torch.equal(routes[name],routes['geometry'].transpose(1,2)),name
        routes.clear()
    def paired_replay(inputs,label):
        rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
        python_rng=random.getstate();numpy_rng=np.random.get_state()
        layer=model.decoder[-1];task_forward=reader.forward
        layer.task_read=False
        reader.forward=ObservationQueryRead.forward.__get__(reader)
        control=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
        end_cpu=torch.get_rng_state();end_cuda=torch.cuda.get_rng_state_all()
        layer.task_read=True;reader.forward=task_forward
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
        random.setstate(python_rng);np.random.set_state(numpy_rng)
        output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
        verify_routes()
        assert torch.equal(end_cpu,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(end_cuda,torch.cuda.get_rng_state_all()))
        keys=['source_features','seed_features','query_points_xyz','query_points_feature',
              'last_center','last_pred_size','last_sem_cls_scores','last_proj_queries','proj_tokens']
        differences={k:float((output[k]-control[k]).abs().max()) for k in keys}
        for k in keys: assert torch.equal(output[k],control[k]),(label,k,differences[k])
        mask_max={k:max(float((a-b).abs().max()) for a,b in zip(output[k],control[k]))
                  for k in ['sp_last_pred_masks','last_pred_masks']}
        replay_records.append(dict(mode=label,rec_max_abs=differences,rng_equal=True,mask_max_abs=mask_max))
        return output

'''
code=replace_once(code,'    evaluator = GroundingEvaluator',helper+'    evaluator = GroundingEvaluator')
code=replace_once(code,'            output = model(inputs)', "            output = paired_replay(inputs,'eval_'+str(start))")
code=replace_once(code,'        output = model(inputs)\n        loss, output, components',
    '        output = model(inputs)\n        verify_routes()\n        loss, output, components')
code=replace_once(code,'        groups = defaultdict',
    "        task_gradient=reader.task_queries.grad\n"
    "        assert task_gradient is not None and torch.isfinite(task_gradient).all()\n"
    "        task_norms=[float(g.norm()) for g in task_gradient]\n"
    "        if step==2:assert all(v>0 for v in task_norms)\n"
    '        groups = defaultdict')
code=replace_once(code,"            'all_present_gradients_finite':True,",
    "            'task_query_gradient_norms':task_norms,'all_present_gradients_finite':True,")
code=replace_once(code,"        'strict_state_tensors':len(state)",
    "        'strict_state_tensors':native_state_count,'added_state_tensors':len(added)")
code=replace_once(code,"        'eval_forwards':2,'train_forwards':2", "        'eval_forwards':4,'train_forwards':2")
code=replace_once(code,"    (args.output/'receipt.json').write_text",
    "    assert all(float(w.abs().max())>0 for w in reader.task_queries)\n"
    "    for handle in route_handles:handle.remove()\n"
    "    import io\n"
    "    buffer=io.BytesIO()\n"
    "    cpu_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}\n"
    "    torch.save(cpu_state,buffer);buffer.seek(0)\n"
    "    restored=torch.load(buffer,map_location='cpu')\n"
    "    model.cpu();model.load_state_dict(restored,strict=True)\n"
    "    assert all(torch.equal(v,restored[k]) for k,v in model.state_dict().items())\n"
    "    receipt.update(task_read=True,source_query_read=True,observation_state=True,strict_cpu_restore=True,restored_tensors=len(restored),direct_routing_verified=True,module_sha256=sha(Path(__file__).parent/'pvground_task_observation_query.py'),source_port_sha256=sha(model_source.parent/'source_port.json'),replay_records=replay_records,new_parameters=sum(p.numel() for p in reader.parameters()))\n"
    "    (args.output/'receipt.json').write_text")
code=code.replace('PVG_TRAINING_INTERFACE_PASS','PVG_TASK_OBSERVATION_INTERFACE_PASS')
commands=[['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',runtime+'/venv/bin/python','-u',root+'/unit.py'],
          ['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',runtime+'/venv/bin/python','-u',root+'/check.py','--runtime',runtime,'--fixtures',fixtures,'--output',root+'/results']]
controller=('import hashlib,json,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
    '(root/"controller.pid").write_text(str(os.getpid())+"\\n")\n'
    'spec=json.loads((root/"spec.json").read_bytes())\n'
    'for name,digest in spec["files"].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n'
    'env=os.environ.copy();env.update('+repr(env['env'])+')\n'
    'for index,argv in enumerate('+repr(commands)+'):\n'
    '    with (root/("unit.log" if index==0 else "run.log")).open("xb") as log:\n'
    '        result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)\n'
    '    (root/("unit.exit" if index==0 else "check.exit")).write_text(str(result.returncode)+"\\n")\n'
    '    if result.returncode:break\n'
    '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\nraise SystemExit(result.returncode)\n')
files={'check.py':code.encode(),'unit.py':(repo/'scripts/check_pvground_task_observation.py').read_bytes(),
    'controller.py':controller.encode(),'source_port.json':port_raw,
    'plan.md':(repo/'docs/PVG_TASK_OBSERVATION_CONTROL_2026-09-17.md').read_bytes()}
for name in ['pvground_source_query.py','pvground_observation_query.py','pvground_task_observation_query.py']:
    files[name]=(repo/'models'/name).read_bytes()
spec=dict(root=root,runtime=runtime,model_source=model_source,env_spec_sha256=env_sha,seed=2027,
    eval_forwards=4,train_forwards=2,optimizer_steps=2,new_checkpoints=0,formal_rows=0,
    files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
command=['screen','-dmS','mcln_pvg_task_observation_interface_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,command)),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    process=process,root=root,expected_seconds=180,formal_rows=0)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wb') as f:f.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
