import ast,csv,hashlib,json,os,sys,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_referit3d_protocol_20260906')
m=json.loads((root/'inputs.json').read_text());source=Path(m['model_source'])
data=Path('/root/autodl-tmp/DATA_ROOT');os.chdir(str(source));sys.path.insert(0,str(source))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
started=time.time();partitions={};metadata={};inputs={}
for dataset in ['nr3d','sr3d']:
 csv_path=data/'refer_it_3d'/(dataset+'.csv')
 with csv_path.open() as f:raw=list(csv.DictReader(f))
 inputs[str(csv_path)]={'bytes':csv_path.stat().st_size,'sha256':sha(csv_path)}
 metadata[dataset]={'csv_rows':len(raw),'splits':{}}
 for split in ['train','test']:
  path=source/'data/meta_data'/(dataset+'_'+split+'_scans.txt')
  scene_list=set(ast.literal_eval(path.read_text()))
  inputs[str(path)]={'bytes':path.stat().st_size,'sha256':sha(path)}
  selected=[row for row in raw if row['scan_id'] in scene_list
    and (str(row['mentions_target_class']).lower()=='true' if dataset=='sr3d'
         else split!='test' or str(row['correct_guess']).lower()=='true')]
  scenes=sorted({row['scan_id'] for row in selected})
  partitions[dataset+'_'+split]=set(scenes)
  metadata[dataset]['splits'][split]={'metadata_scene_count':len(scene_list),'effective_rows':len(selected),
    'effective_scene_count':len(scenes),'scene_ids':scenes,
    'effective_row_keys_sha256':hashlib.sha256(json.dumps([(r['scan_id'],r['target_id'],r['utterance']) for r in selected]).encode()).hexdigest()}
intersections={a+'__'+b:sorted(partitions[a]&partitions[b]) for a,b in [
 ('nr3d_train','nr3d_test'),('sr3d_train','sr3d_test'),('nr3d_train','sr3d_test'),
 ('sr3d_train','nr3d_test'),('nr3d_train','sr3d_train'),('nr3d_test','sr3d_test')]}
classes_path=source/'data/cls_results.json';classes=json.loads(classes_path.read_text())
inputs[str(classes_path)]={'bytes':classes_path.stat().st_size,'sha256':sha(classes_path)}
availability={}
for split,disk_split in [('train','train'),('test','val')]:
 scenes=sorted(partitions['sr3d_'+split])
 availability[split]={'missing_predicted_classes':[scene for scene in scenes if scene not in classes],
  'missing_superpoints':[scene for scene in scenes if not (data/'superpoints'/disk_split/(scene+'_superpoint.pth')).is_file()],
  'missing_group_free_files':[scene for scene in scenes if not (data/'group_free_pred_bboxes'/('group_free_pred_bboxes_'+disk_split)/(scene+'.npy')).is_file()]}
for name in ['src/joint_det_dataset.py','train_dist_mod.py','models/mcln.py']:
 path=source/name;inputs[str(path)]={'bytes':path.stat().st_size,'sha256':sha(path)}
import torch
parent_path=Path(m['checkpoint']);assert sha(parent_path)==m['checkpoint_sha256']
parent=torch.load(str(parent_path),map_location='cpu');config=vars(parent['config'])
keys=['dataset','test_dataset','joint_det','butd','butd_gt','butd_cls','use_color','use_height',
 'use_multiview','num_target','num_decoder_layers','use_soft_token_loss','use_contrastive_align',
 'use_source_choice_selector','eval_use_selector_choice_scores','use_source_moe','use_sacr_source','detect_intermediate','pp_checkpoint']
selected_config={key:config[key] for key in keys if key in config}
sr_parent=data/'output/network_v99_baseline_gt/sr3d/control/official_rec_monitor/official_best_rec025_epoch_26_0p68481327.pth'
result={'schema':'mcln-referit3d-protocol-audit-v1','status':'complete','annotations':metadata,
 'intersections':intersections,'intersection_counts':{key:len(value) for key,value in intersections.items()},
 'sr_train_scenes_outside_nr_train':sorted(partitions['sr3d_train']-partitions['nr3d_train']),
 'sr_test_scenes_outside_nr_test':sorted(partitions['sr3d_test']-partitions['nr3d_test']),
 'sr_input_file_availability':availability,'input_files':inputs,
 'documented_sr_parent_exists':sr_parent.is_file(),'documented_sr_parent_path':str(sr_parent),
 'known_nr_parent_sha256':m['checkpoint_sha256'],'known_nr_parent_config':selected_config,
 'known_nr_parent_evaluation_only':parent['evaluation_only'],'known_nr_parent_has_optimizer':'optimizer' in parent,
 'full_checkpoint_training_lineage_audited':False,'gpu_forwards':0,'optimizer_steps':0,
 'formal_model_evaluation_rows':0,'annotation_only_audit':True,'elapsed_seconds':time.time()-started}
with (root/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({key:value for key,value in result.items() if key not in ['annotations','intersections','input_files']}))
