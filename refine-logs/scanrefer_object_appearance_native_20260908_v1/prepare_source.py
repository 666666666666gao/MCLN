import ast,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);config=json.loads((root/'manifest.json').read_text());parent=Path(config['parent_source'])
raw=(parent/'native_source_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==config['parent_source_manifest_sha256']
files=json.loads(raw)['files'];source=root/'model_source';source.mkdir()
for name,digest in files.items():
    content=(parent/name).read_bytes();assert hashlib.sha256(content).hexdigest()==digest,name
    target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
before=files['models/mcln.py'];path=source/'models/mcln.py';code=path.read_text()
for old,new in json.loads((root/'patches.json').read_text()):
    assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
ast.parse(code);path.write_bytes(code.encode());files['models/mcln.py']=hashlib.sha256(path.read_bytes()).hexdigest()
for name,digest in config['overlays'].items():
    content=(root/'overlays'/name).read_bytes();assert hashlib.sha256(content).hexdigest()==digest
    if name in files:assert files[name]==digest,(name,'unexpected helper drift')
    (source/name).write_bytes(content);files[name]=digest
record=dict(parent_source=config['parent_source'],parent_manifest_sha256=config['parent_source_manifest_sha256'],base_mcln_sha256=before,files=files)
(source/'appearance_source_manifest.json').write_bytes(json.dumps(record,indent=2).encode()+b'\n')
print(json.dumps(dict(files=len(files),source_sha256=hashlib.sha256((source/'appearance_source_manifest.json').read_bytes()).hexdigest())))
