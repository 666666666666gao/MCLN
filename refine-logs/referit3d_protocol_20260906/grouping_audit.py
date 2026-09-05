import csv,hashlib,json
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_referit3d_protocol_20260906')
r=json.loads((root/'receipt.json').read_text())
path=Path('/root/autodl-tmp/DATA_ROOT/refer_it_3d/sr3d.csv')
assert hashlib.sha256(path.read_bytes()).hexdigest()==r['input_files'][str(path)]['sha256']
scans=set(r['annotations']['sr3d']['splits']['train']['scene_ids'])
with path.open() as f:rows=[row for row in csv.DictReader(f) if row['scan_id'] in scans and str(row['mentions_target_class']).lower()=='true']
m=json.loads(Path('/root/autodl-tmp/mcln_object_appearance_pair_20260906_v2/input_manifest.json').read_text())
salt=m['split_salt']
def fold(key):return int(hashlib.sha256((salt+'\0'+key).encode()).hexdigest()[:8],16)%5
def profile(use_space):
 sets={'fit':[],'holdout':[]}
 for row in rows:
  key=row['scan_id'].split('_')[0] if use_space else row['scan_id']
  sets['holdout' if fold(key)==0 else 'fit'].append(row)
 out={}
 for split,items in sets.items():
  out[split]={'rows':len(items),'scans':len({v['scan_id'] for v in items}),
    'physical_spaces':len({v['scan_id'].split('_')[0] for v in items})}
 shared=sorted({v['scan_id'].split('_')[0] for v in sets['fit']}&{v['scan_id'].split('_')[0] for v in sets['holdout']})
 out['spaces_in_both_module_partitions']=shared
 out['spaces_in_both_module_partitions_count']=len(shared)
 first=sets['fit'][:2048]
 out['first2048_csv_fit']={'rows':len(first),'scans':len({v['scan_id'] for v in first}),
  'physical_spaces':len({v['scan_id'].split('_')[0] for v in first})}
 return out
result={'schema':'mcln-sr3d-prospective-module-grouping-v1','status':'complete',
 'scan_id_grouping':profile(False),'physical_space_grouping':profile(True),
 'salt_from_existing_nr_experiment':salt,'official_partitions_modified':False,
 'actual_sr_module_split_created':False,'training_runs_started':0,'gpu_forwards':0,'optimizer_steps':0,
 'counterfactual_diagnostic_only':True,
 'source_for_space_id_semantics':'https://github.com/ScanNet/ScanNet#data-organization'}
with (root/'grouping_receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({key:({a:b for a,b in value.items() if a!='spaces_in_both_module_partitions'} if key.endswith('_grouping') else value) for key,value in result.items()}))
