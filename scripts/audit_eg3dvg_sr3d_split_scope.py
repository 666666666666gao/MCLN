import ast,csv,hashlib,json,pickle
from pathlib import Path
acceptance=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
root=Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1')
source=acceptance/'referit_input_source'
data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
train_file=source/'data/meta_data/sr3d_train_scans.txt'
train_set=set(ast.literal_eval(train_file.read_text()))
csv_file=acceptance/'referit_input_cache/data_view/ReferIt3D/sr3d.csv'
with csv_file.open() as f:
    selected=[row for row in csv.DictReader(f) if row['scan_id'] in train_set and row['mentions_target_class'].lower()=='true']
with (acceptance/'referit_input_cache/sr3d_annotations.pkl').open('rb') as f:
    validation=pickle.load(f)
val_scenes={row['scan_id'] for row in validation}
train_scenes={row['scan_id'] for row in selected}
scan_train=set((data/'ScanRefer/ScanRefer_filtered_train.txt').read_text().splitlines())
physical=lambda values:{value.rsplit('_',1)[0] for value in values}
result={'sr_train_rows_before_parser':len(selected),'sr_train_scenes':len(train_scenes),'sr_val_rows':len(validation),
        'sr_val_scenes':len(val_scenes),'sr_train_val_scene_overlap':sorted(train_scenes&val_scenes),
        'sr_train_val_physical_room_overlap':sorted(physical(train_scenes)&physical(val_scenes)),
        'scanrefer_train_sr_val_scene_overlap':sorted(scan_train&val_scenes),
        'scanrefer_train_sr_val_physical_room_overlap':sorted(physical(scan_train)&physical(val_scenes)),
        'csv_sha256':hashlib.sha256(csv_file.read_bytes()).hexdigest(),'train_split_sha256':hashlib.sha256(train_file.read_bytes()).hexdigest(),
        'claim_boundary':'Checks local author split files, not provenance of every training sample used by the released checkpoint.',
        'model_forwards':0,'optimizer_steps':0}
(root/'split_scope.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
