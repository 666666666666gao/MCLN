"""Build fixed ReferIt3D fit/holdout row maps using the actual native loaders."""
import argparse,datetime,hashlib,json,os,sys,time
from pathlib import Path

def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
 return d.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);args=p.parse_args()
 root=args.spec.parent;spec=json.loads(args.spec.read_bytes());started=time.time()
 assert sha(__file__)==spec['script_sha256']
 selection=Path(spec['selection_root']);assert sha(selection/'manifest.json')==spec['selection_manifest_sha256']
 manifest=json.loads((selection/'manifest.json').read_bytes());source=Path(manifest['model_source'])
 assert sha(source/'appearance_source_manifest.json')==manifest['source_manifest_sha256']
 for name,digest in json.loads((source/'appearance_source_manifest.json').read_bytes())['files'].items():assert sha(source/name)==digest,name
 annotation=json.loads((selection/'annotation_receipt.json').read_bytes())
 assert sha(selection/'annotation_receipt.json')==manifest['input_files'][str(selection/'annotation_receipt.json')]
 for path,item in annotation['annotations_and_split_files'].items():
  actual=Path(path) if '/DATA_ROOT/' in path else source/'data/meta_data'/Path(path).name
  assert sha(actual)==item['sha256'],str(actual)
 points=Path(manifest['data_root'])/'train_v3scans.pkl';points_digest=sha(points);assert points_digest==manifest['input_files'][str(points.resolve())]
 assert sha(spec['scan_input_manifest'])==spec['scan_input_manifest_sha256']
 full_input=json.loads(Path(spec['scan_input_manifest']).read_bytes())
 assert Path(full_input['data_root'])==Path(manifest['data_root'])
 reference=json.loads((root/'nr_reference.json').read_bytes());assert sha(root/'nr_reference.json')==spec['reference_sha256']
 os.chdir(str(source));sys.path.insert(0,str(source))
 import torch
 from src.joint_det_dataset import Joint3DDataset,unpickle_data,read_label_mapping
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized();torch.set_num_threads(1)
 scenes=list(unpickle_data(str(points)))[0];assert len(scenes)==1201
 dataset=object.__new__(Joint3DDataset)
 dataset.split='train';dataset.data_path=manifest['data_root'];dataset.scans=scenes
 dataset.use_sacr_source=False;dataset.overfit=False;dataset.skip_missing_superpoints=True
 dataset._scene_graph_parse=lambda annos:None
 dataset.label_map=read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv',label_from='raw_category',label_to='id')
 detection=dataset.load_scannet_annos();assert len(detection)==1199
 for scene in scenes:
  path=Path(manifest['data_root'])/'superpoints/train'/(scene+'_superpoint.pth')
  assert sha(path)==full_input['superpoint_files']['train'][path.name],scene
 salt=reference['split_salt']
 def physical(scene):return scene.split('_')[0]
 def heldout(scene):return int(hashlib.sha256((salt+'\0'+physical(scene)+'_00').encode()).hexdigest()[:8],16)%5==0
 summaries={}
 for name,expected in [('nr3d',32919),('sr3d',65846)]:
  rows=dataset.load_annos(name);assert len(rows)==expected
  assert len({r['scan_id'] for r in rows})==annotation['protocols'][name]['train']['language_scans']
  language_fit=[i for i,r in enumerate(rows) if not heldout(r['scan_id'])]
  language_holdout=[i for i,r in enumerate(rows) if heldout(r['scan_id'])]
  if name=='nr3d':
   assert all(r['scan_id'].endswith('_00') for r in rows)
   assert {'fit':language_fit,'holdout':language_holdout}==reference['row_ids']
   assert len(language_fit)==26747 and len(language_holdout)==6172
  held_physical={physical(rows[i]['scan_id']) for i in language_holdout}
  detection_keep=[i for i,r in enumerate(detection) if physical(r['scan_id']) not in held_physical]
  detection_excluded=[i for i,r in enumerate(detection) if physical(r['scan_id']) in held_physical]
  fit=language_fit+[len(rows)+repeat*len(detection)+i for repeat in range(10) for i in detection_keep]
  detection_ids={physical(detection[i]['scan_id']) for i in detection_keep}
  language_ids={physical(rows[i]['scan_id']) for i in language_fit}
  assert not (detection_ids|language_ids)&held_physical
  formal_scenes=set(annotation['protocols'][name]['val']['language_scan_ids'])
  assert not {physical(s) for s in formal_scenes}&(detection_ids|language_ids|held_physical)
  assert set(fit).isdisjoint(language_holdout)
  assert len(set(fit))==len(fit)
  result=dict(dataset=name,split_salt=salt,physical_rule='sha256(salt+NUL+physical_scene+_00) first8hex modulo5; holdout is0',
   original_language_rows=len(rows),original_detection_base_rows=len(detection),detection_repeats=10,
   language_fit_ids=language_fit,holdout_ids=language_holdout,fit_ids=fit,
   detection_kept_base_ids=detection_keep,detection_excluded_base_ids=detection_excluded,
   language_row_keys_sha256=[hashlib.sha256(json.dumps({k:r[k] for k in ['scan_id','target_id','utterance','dataset']},sort_keys=True,separators=(',',':')).encode()).hexdigest() for r in rows],
   detection_scene_ids=[r['scan_id'] for r in detection],
   fit_physical_scenes=sorted(language_ids|detection_ids),holdout_physical_scenes=sorted(held_physical),
   formal_rows=annotation['protocols'][name]['val']['language_rows'],formal_scenes=sorted(formal_scenes))
  path=root/(name+'_partition.json');path.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
  summaries[name]=dict(language_fit_rows=len(language_fit),holdout_rows=len(language_holdout),
   detection_kept_base_rows=len(detection_keep),detection_excluded_base_rows=len(detection_excluded),
   total_fit_rows=len(fit),batch8_updates_one_pass=(len(fit)+7)//8,
   fit_physical_scenes=len(language_ids|detection_ids),holdout_physical_scenes=len(held_physical),
   formal_rows=result['formal_rows'],partition_sha256=sha(path),partition_bytes=path.stat().st_size,
   train_holdout_physical_overlap=[],train_formal_physical_overlap=[])
 receipt=dict(status='pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  datasets=summaries,script_sha256=sha(__file__),source_manifest_sha256=manifest['source_manifest_sha256'],
  points_sha256=points_digest,annotation_receipt_sha256=sha(selection/'annotation_receipt.json'),
  nr_reference_sha256=spec['reference_sha256'],model_forwards=0,optimizer_steps=0,formal_rows_evaluated=0,
  text_graphs_parsed=False,torch_cuda_initialized=torch.cuda.is_initialized(),
  scope='native annotation and fixed room partition preparation; no training or metric claim',elapsed_seconds=time.time()-started)
 assert not receipt['torch_cuda_initialized']
 (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()