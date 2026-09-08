"""Generate the fixed native-loss trial with explicit source-reader initialization/restoration."""
from pathlib import Path

repo=Path(__file__).resolve().parents[1]

def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

train=(repo/'scripts/run_pvground_scanrefer_vsa_order.py').read_text()
train=replace_once(train,"    assert interface['env_spec_sha256']==env_sha",
    "    assert interface['env_spec_sha256']==env_sha\n    assert interface['source_query_read'] and spec['source_query_read']\n    assert interface['module_sha256']==spec['source_query_module_sha256']==sha(output/'pvground_source_query.py')\n    assert interface['zero_residual_eval_rec_replay_exact'] and interface['train_layer_replay']['output_exact']\n    assert interface['train_layer_replay']['rng_end_equal']\n    assert interface['source_port_sha256']==sha(spec['source_port'])")
begin=train.index("    for name, entry in upstream['sources']['PV-Ground']['files'].items():")
end=train.index("    checkpoint=",begin)
train=train[:begin]+"    for name,digest in port['files'].items():assert sha(model_source/name)==digest,name\n"+train[end:]
train=replace_once(train,'    model.load_state_dict(initial,strict=True)\n    model.cuda()',
    "    model.load_state_dict(initial,strict=True)\n    from pvground_source_query import install_source_query_read\n    install_source_query_read(model)\n    added=set(model.state_dict())-set(initial)\n    assert added and all(n.startswith('decoder.5.source_query_read.') for n in added)\n    initial.update({n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n in added})\n    assert sum(p.numel() for p in model.decoder[-1].source_query_read.parameters())==interface['new_parameters']\n    model.cuda()")
train=replace_once(train,'    assert len(trainable)==783 and sum(p.numel() for p in trainable.values())==27959611',
    "    assert len(trainable)==interface['trainable_tensors']\n    assert sum(p.numel() for p in trainable.values())==interface['trainable_parameters']")
train=replace_once(train,"        temporary=output/(name+'.tmp')",
    "        data.update(source_query_read=True,source_query_module_sha256=spec['source_query_module_sha256'],source_port_sha256=sha(spec['source_port']))\n        temporary=output/(name+'.tmp')")
train=replace_once(train,"    write_json(output/'receipt.json',receipt)",
    "    receipt.update(source_query_read=True,source_query_module_sha256=spec['source_query_module_sha256'],source_port_sha256=sha(spec['source_port']),added_state_tensors=len(added),new_parameters=interface['new_parameters'])\n    write_json(output/'receipt.json',receipt)")
path=repo/'scripts/run_pvground_scanrefer_source_query.py'
compile(train,str(path),'exec');path.write_text(train,encoding='utf-8')
print('Prepared fixed training source; no remote job launched.')
