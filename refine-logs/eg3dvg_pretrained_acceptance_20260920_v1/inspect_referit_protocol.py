import ast,csv,hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent;source=r/'source';data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'upstream_commit':'174e34894aea6513442da6b5dfa9b3e2bf8a1efa','model_forwards':0,'training_steps':0,'datasets':{}}
for name in ['nr3d','sr3d']:
 split=source/('data/meta_data/'+name+'_test_scans.txt')
 scenes=set(ast.literal_eval(split.read_text()))
 path=data/('refer_it_3d/'+name+'.csv')
 with path.open() as f:rows=list(csv.DictReader(f))
 field='correct_guess' if name=='nr3d' else 'mentions_target_class'
 chosen=[a for a in rows if a['scan_id'] in scenes and str(a[field]).lower()=='true']
 selected_scenes=sorted({a['scan_id'] for a in chosen})
 classfile=source/'data/cls_results.json';classes=json.loads(classfile.read_text())
 report['datasets'][name]={'rows':len(chosen),'scenes':len(selected_scenes),'csv_sha256':sha(path),'split_sha256':sha(split),'filter':field+'=true','mentions_target_class_filter':name=='sr3d','butd_cls':True,'object_boxes':'scene GT instance boxes','object_classes':'author cls_results predicted classes','cls_results_sha256':sha(classfile),'missing_class_scenes':[s for s in selected_scenes if s not in classes],'missing_superpoint_scenes':[s for s in selected_scenes if not (data/'superpoints/val'/(s+'_superpoint.pth')).exists()],'filter_non_gt_boxes':True,'filter_rule':'native score multiplied by 1 if prediction overlaps any scene input object box at IoU > .25; otherwise 0','native_box_rule':'average of regressed and predicted-mask-derived box; no snapping to GT box'}
report['dedicated_weights']='Pinned public README model table links only ScanRefer; no dedicated Nr3D/Sr3D checkpoint verified.'
(r/'referit_protocol.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
