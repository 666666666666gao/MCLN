import ast
import difflib
import hashlib
import json
from pathlib import Path

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
native=Path('C:/Users/gb/.codex/tmp/pvg_native_dataset_exact_20260908.py')
published=Path('C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/joint_det_dataset.py')
expected=json.loads((repo/'refine-logs/scanrefer_object_appearance_native_20260908_v1/appearance_source_manifest.json').read_bytes())['files']['src/joint_det_dataset.py']
assert hashlib.sha256(native.read_bytes()).hexdigest()==expected
texts=[p.read_text(encoding='utf-8') for p in (native,published)]
trees=[ast.parse(t) for t in texts]
results={}
for name in ('_get_pc','_get_detected_objects','_get_scene_objects'):
    nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name) for tree in trees]
    identical=ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
    code=[ast.get_source_segment(t,n) for t,n in zip(texts,nodes)]
    diff='\n'.join(difflib.unified_diff(code[0].splitlines(),code[1].splitlines(),fromfile='native',tofile='PV-Ground',lineterm=''))
    results[name]={'ast_identical':identical,'diff':diff}
report={'native_source_sha256':expected,'published_source_sha256':hashlib.sha256(published.read_bytes()).hexdigest(),'functions':results,'model_forwards':0,'optimizer_steps':0}
output=repo/'refine-logs/pvground_pretrained_forward_20260908_v2/input_function_comparison.json'
output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(report,indent=2))
