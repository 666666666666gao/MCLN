import json
from pathlib import Path
r=Path(__file__).resolve().parent;d=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
ann=json.loads((r/'annotation_manifest.json').read_text())
ids=sorted({a['scan_id'] for a in ann})
sp=sorted((d/'superpoints/val').glob('*_superpoint.pth'))
rec={'annotation_rows':len(ann),'annotation_scenes':len(ids),'scanrefer_scene_list':len((d/'ScanRefer/ScanRefer_filtered_val.txt').read_text().splitlines()),'scene_cache_count_recorded':json.loads((r/'data_receipt.json').read_text())['scenes'],'superpoint_files':len(sp),'missing_annotation_sp':[s for s in ids if not (d/'superpoints/val'/(s+'_superpoint.pth')).is_file()],'missing_annotation_detection':[s for s in ids if not (d/'group_free_pred_bboxes/group_free_pred_bboxes_val'/(s+'.npy')).is_file()],'model_forwards':0}
(r/'scene_counts.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
