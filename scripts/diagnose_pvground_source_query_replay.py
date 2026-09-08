"""Measure disabled/disabled/enabled repeat differences without changing the model."""
import hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
old=repo/'refine-logs/pvground_source_query_interface_20260908_v1'
root='/root/autodl-tmp/mcln_pvground_source_query_replay_diagnostic_20260908_v1'
archive=repo/'refine-logs/pvground_source_query_replay_diagnostic_20260908_v1'
code=(old/'check.py').read_text()
start=code.index('    evaluator = GroundingEvaluator')
end=code.index("\n\nif __name__ == '__main__':")
diagnostic='''    model.eval()
    inputs,batch=batch_at(0,eval_owner)
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def restore():
        random.setstate(states[0]);np.random.set_state(states[1])
        torch.set_rng_state(states[2]);torch.cuda.set_rng_state_all(states[3])
    traces=[];rngs=[];source_outputs=[]
    handle=reader.register_forward_hook(lambda module,args,value:source_outputs.append(value.detach().cpu().clone()))
    with torch.no_grad():
        for enabled in [False,False,True]:
            restore();reader.enabled=enabled
            output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
            trace={}
            for key,value in output.items():
                if torch.is_tensor(value):trace[key]=value.detach().cpu().clone()
                elif isinstance(value,list) and value and all(torch.is_tensor(v) for v in value):
                    for index,item in enumerate(value):trace[key+'.'+str(index)]=item.detach().cpu().clone()
            traces.append(trace)
            rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
            del output
    handle.remove()
    differences=[]
    for trace in traces[1:]:
        pairs={}
        for key,base in traces[0].items():
            value=trace[key]
            pairs[key]=dict(exact=torch.equal(base,value),shape=list(base.shape),
                max_abs=float((base.double()-value.double()).abs().max()) if base.numel() else 0.0)
        differences.append(pairs)
    receipt=dict(status='diagnosed',cases=['disabled','disabled_repeat','enabled'],
        differences=differences,source_output_max=[float(v.abs().max()) for v in source_outputs],
        rng_end_equal=[torch.equal(rngs[0][0],r[0]) and all(torch.equal(a,b) for a,b in zip(rngs[0][1],r[1])) for r in rngs[1:]],
        model_forwards=3,optimizer_steps=0,formal_rows=0,module_sha256=sha(Path(__file__).parent/'pvground_source_query.py'),
        script_sha256=sha(__file__),time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
    print(json.dumps(dict(nonexact=[{k:v for k,v in d.items() if not v['exact']} for d in differences],
        source_output_max=receipt['source_output_max'],rng_end_equal=receipt['rng_end_equal'])),flush=True)
'''
code=code[:start]+diagnostic+code[end:]
spec=json.loads((old/'spec.json').read_bytes())
controller=(old/'controller.py').read_text().replace(spec['root'],root)
files={'check.py':code.encode(),'controller.py':controller.encode(),'pvground_source_query.py':(repo/'models/pvground_source_query.py').read_bytes()}
spec.update(root=root,eval_forwards=3,train_forwards=0,optimizer_steps=0)
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(json.loads((old/'spec.json').read_bytes())['root']+'/controller.exit') as f:assert f.read().strip()==b'1'
assert Path(root).name not in s.listdir('/root/autodl-tmp');s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
cmd=['screen','-dmS','mcln_pvg_source_replay_diag_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,cmd)),timeout=30);assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
s.close();c.close();print(json.dumps(dict(process=process,root=root,planned_forwards=3,optimizer_steps=0)),flush=True)
