import importlib.util,copy,json
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_scanrefer_appearance_readout_transfer_preparation_20260908_v1/analyze.py')
s=importlib.util.spec_from_file_location('transfer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Each row independently describes before/after native and deployed threshold decisions.
rows=[]
for i,(bn,bf,an,af) in enumerate([(0,1,1,0),(1,1,0,0),(0,0,1,1),(1,0,1,1)]):
 d={'row_id':i}
 for arm,ns,fs in [('initial',bn,bf),('control',bn,bf),('appearance',an,af)]:
  d[arm+'_native_iou']=.8 if ns else .0;d[arm+'_system_iou']=.8 if fs else .0
 rows.append(d)
r=m.summarize(rows)['0.5']['comparisons']['control']
assert r['native_effect']=={'repair':2,'damage':1,'net':1}
assert r['system_effect']=={'repair':2,'damage':2,'net':0}
assert r['system_lift_change']==-1
assert r['system_damage_appearance_native_pass']==1
assert r['cross_table']=={'0110':1,'1100':1,'0011':1,'1011':1}
base=[{'row_id':i,'scan_id':'scene','physical_space':'physical','point_sha256':str(i),'rec_iou':.4,'mask_iou':.3} for i in range(2)]
native={arm:copy.deepcopy(base) for arm in ['initial','control','appearance']};full=copy.deepcopy(native)
assert len(m.join_rows(native,full))==2
full['appearance'].reverse()
rejected=False
try:m.join_rows(native,full)
except AssertionError:rejected=True
assert rejected
print(json.dumps({'checks':['independent4rowdecisionalgebra','native-success-system-failure-count','cross-table-bits','matching-identities','mismatched-row-order-rejected'],'pass':True,'real6887_analysis':False}))
