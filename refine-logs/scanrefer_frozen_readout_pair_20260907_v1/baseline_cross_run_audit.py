import datetime,hashlib,json,math
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1')
old=Path('/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
current_manifest=read(root/'input_manifest.json')
old_manifest=read(old/'input_manifest.json')
assert current_manifest['artifacts']==old_manifest['artifacts']
assert current_manifest['source_manifest_sha256']==old_manifest['source_manifest_sha256']
assert current_manifest['data_root']==old_manifest['data_root']
assert current_manifest['split_protocol_sha256']==old_manifest['split_protocol_sha256']
rows=read(root/'baseline_rows.json')
recorded=read(root/'baseline_metrics.json')
native_recorded=read(root/'baseline_native_metrics.json')
assert rows['native_only']==rows['frozen_gt'] and recorded['native_only']==recorded['frozen_gt']
previous=read(old/'baseline_rows.json')['control']
actual=rows['native_only']
assert len(actual)==len(previous)==6887
keys=['row_id','scan_id','physical_space','point_sha256']
identity_differences=[]
value_differences={field:[] for field in ['rec_iou','mask_iou','selected_variant_position']}
for before,after in zip(previous,actual):
 if any(before[key]!=after[key] for key in keys):identity_differences.append(after['row_id'])
 for field in value_differences:
  if before[field]!=after[field]:value_differences[field].append(after['row_id'])
metrics={'rows':len(actual),'mask_miou':sum(row['mask_iou'] for row in actual)/len(actual)*100.}
for field,prefix in [('rec_iou','rec'),('mask_iou','mask')]:
 assert all(math.isfinite(row[field]) and 0<=row[field]<=1 for row in actual)
 for threshold,suffix in [(.25,'025'),(.5,'050')]:metrics[prefix+'_hits'+suffix]=sum(row[field]>threshold for row in actual)
assert metrics==recorded['native_only']
native={'rows':len(actual)}
for threshold,suffix in [(.25,'025'),(.5,'050')]:native['rec_hits'+suffix]=sum(row['native_rec_iou']>threshold for row in actual)
assert native==native_recorded['native_only']==native_recorded['frozen_gt']
result={'schema':'mcln-frozen-readout-baseline-cross-run-audit-v1','status':'complete',
 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'rows':6887,'current_arms_exact_row_parity':True,'current_native_and_system_metrics_recomputed':True,
 'same_protected_artifacts_source_mesh_and_split':True,'identity_difference_row_ids':identity_differences,
 'cross_run_difference_row_ids':value_differences,'system_metrics':metrics,'native_rec_metrics':native,
 'baseline_rows_sha256':sha(root/'baseline_rows.json'),'baseline_metrics_sha256':sha(root/'baseline_metrics.json'),
 'baseline_native_metrics_sha256':sha(root/'baseline_native_metrics.json'),
 'prior_range_baseline_rows_sha256':sha(old/'baseline_rows.json'),
 'training_manifest_sha256':sha(root/'input_manifest.json'),
 'gpu_forwards':0,'optimizer_steps':0,'checkpoint_writes':0,'formal_rows':0,
 'baseline_only_not_quality_gain':True,'is_new_promotion_gate':False,
 'retired_range_weights_not_read':True}
with (root/'baseline_cross_run_audit.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
print(json.dumps(result))
