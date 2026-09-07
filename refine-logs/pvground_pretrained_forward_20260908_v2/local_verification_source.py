import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
runtime=repo/'refine-logs/pvground_runtime_20260908_v1'
forward=repo/'refine-logs/pvground_pretrained_forward_20260908_v2'
fixture=repo/'refine-logs/pvground_train_fixtures_20260908_v2'
receipt=json.loads((forward/'receipt.json').read_bytes())
assert receipt['status']=='pass' and receipt['full_model_forwards']==2 and receipt['training_fixture_rows']==4
assert receipt['optimizer_steps']==receipt['formal_rows']==0 and receipt['loaded_state_unchanged']
assert (forward/'controller.exit').read_text().strip()=='0'
assert receipt['script_sha256']==hashlib.sha256((repo/'scripts/verify_pvground_pretrained_forward.py').read_bytes()).hexdigest()
assert receipt['fixture_receipt_sha256']==hashlib.sha256((fixture/'receipt.json').read_bytes()).hexdigest()
spec=json.loads((repo/'configs/pvground_runtime_env_20260908.json').read_bytes())
assert receipt['env_spec_sha256']==hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
strict=receipt['strict_load']
assert strict['state_tensors']==1234 and strict['missing_keys']==strict['unexpected_keys']==[]
assert strict['position_ids_serialization_port']=={'shape':[1,514],'persistent':False,'values_changed':False}
build=json.loads((runtime/'witness_repair_v3/build_receipt.json').read_bytes())
agent=json.loads((runtime/'agent_follows_doc/agent_kernel_receipt.json').read_bytes())
review=json.loads((runtime/'agent_follows_doc/agent_follows_doc_report.json').read_bytes())
assert build['status']==agent['status']=='pass'
assert review['environment_execution_verdict']=='PASS' and review['overall_verdict']=='WARN'
assert (runtime/'build_repair_v2/original_base_packages_before.json').read_bytes()==(runtime/'witness_repair_v3/base_packages_after.json').read_bytes()
results=[]
for row in receipt['rows']:
    path=forward/('row_%05d_boxes.npy'%row['training_row_id'])
    raw=path.read_bytes()
    assert raw[:8]==b'\x93NUMPY\x01\x00'
    length=struct.unpack('<H',raw[8:10])[0]
    header=ast.literal_eval(raw[10:10+length].decode('latin1').strip())
    assert header=={'descr':'<f4','fortran_order':False,'shape':(256,6)}
    values=struct.unpack('<1536f',raw[10+length:])
    assert all(math.isfinite(x) for x in values)
    positive=sum(all(values[i*6+j]>0 for j in (3,4,5)) for i in range(256))
    assert positive==row['positive_size_boxes']
    results.append({'training_row_id':row['training_row_id'],'positive_size_boxes':positive,'npy_sha256':hashlib.sha256(raw).hexdigest()})
report={'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'verification':'independent standard-library NPY decode and float checks; receipt/source/fixture/spec hashes; strict-load keys; unchanged base packages',
    'boxes_decoded':1024,'positive_size_boxes':sum(x['positive_size_boxes'] for x in results),
    'nonpositive_size_boxes':1024-sum(x['positive_size_boxes'] for x in results),'rows':results,
    'new_model_forwards':0,'optimizer_steps':0,'formal_rows':0,'review_warning_retained':True}
(forward/'local_verification.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
(forward/'local_verification_source.py').write_bytes(Path(__file__).read_bytes())
(forward/'input_function_comparison_source.py').write_bytes(Path('C:/Users/gb/.codex/tmp/compare_pvg_input_functions_20260908.py').read_bytes())
print(json.dumps(report))
