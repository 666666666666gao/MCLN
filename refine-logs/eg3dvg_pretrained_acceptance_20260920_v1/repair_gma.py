import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
assert (r/'controller.exit').read_text().strip()=='1'
assert (r/'preflight.exit').read_text().strip()=='1'
assert not (r/'formal').exists()
archive=r/'failed_gma';archive.mkdir()
for name in ['spec.json','launch.json','controller.log','controller.exit','preflight.log','preflight.exit','preflight']:
 (r/name).rename(archive/name)
p=r/'source/models/encoder_decoder_layers.py';before=p.read_bytes()
manifest=json.loads((r/'upstream_manifest.json').read_text())
assert hashlib.sha256(before).hexdigest()==manifest['files']['models/encoder_decoder_layers.py']
assert before.count(b'self.spatial_n_head')==2
after=before.replace(b'self.spatial_n_head',b'self.n_head')
compile(after,str(p),'exec');p.write_bytes(after)
rec={'file':'models/encoder_decoder_layers.py','before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'changes':'Two undefined self.spatial_n_head references replaced with existing self.n_head.','reason':'Constructor stores n_head=8; checkpoint lang projection72 and pairwise projection9 require 8 heads.','learned_parameters_added_or_removed':0,'formal_rows_before_fix':0,'failed_model_forward_attempts':1}
(r/'gma_repair.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
