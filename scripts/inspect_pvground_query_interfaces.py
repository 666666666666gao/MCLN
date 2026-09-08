import ast,hashlib,json
from pathlib import Path

root=Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground')
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
manifest=json.loads((runtime/'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']['files']
port=json.loads((root.parent/'source_port.json').read_bytes())
requests={'models/pv_ground.py':{'PVGround':['_run_backbones','_generate_queries','forward'],'GumbelSampling':['forward']},
          'models/encoder_decoder_layers.py':{'BiDecoderLayer':['__init__','forward']},
          'models/pv_utils.py':{'VoxelSetAbstraction':['forward']}}
records={}
for name,classes in requests.items():
    raw=(root/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    expected=port['after_sha256'] if 'PV-Ground/'+name==port['file'] else manifest[name]['sha256']
    assert digest==expected,name
    text=raw.decode();lines=text.splitlines();tree=ast.parse(text)
    snippets={}
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef) and cls.name in classes:
            for index,node in enumerate(cls.body):
                if isinstance(node,ast.FunctionDef) and node.name in classes[cls.name]:
                    start=node.lineno-1
                    following=[x.lineno-1 for x in cls.body[index+1:] if hasattr(x,'lineno')]
                    stop=min(following) if following else max(x.lineno for x in ast.walk(cls) if hasattr(x,'lineno'))
                    snippets[cls.name+'.'+node.name]=dict(start_line=start+1,end_line=stop,source='\n'.join(lines[start:stop]))
    assert len(snippets)==sum(len(x) for x in classes.values()),(name,list(snippets))
    records[name]=dict(sha256=digest,snippets=snippets)
print(json.dumps(dict(model_source=str(root),model_forwards=0,optimizer_steps=0,formal_rows=0,files=records)))
