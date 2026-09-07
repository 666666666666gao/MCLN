"""Export frozen ScanRefer fit detector crops, without model or GT-based cropping."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,type=Path)
    args=p.parse_args()
    config=json.loads(args.manifest.read_text())
    out=args.manifest.parent
    sys.path.insert(0,config['model_source'])
    os.chdir(config['model_source'])
    import numpy as np
    from src.joint_det_dataset import unpickle_data
    source=Path(config['model_source'])
    assert sha(source/'native_source_manifest.json')==config['source_manifest_sha256']
    old=Path(config['fit_archive'])/'rows_000_127.json.gz'
    assert sha(old)==config['fit_shard_sha256']
    # Prespecified first sixteen saved fit expressions. No outcome-based selection.
    rows=json.loads(gzip.decompress(old.read_bytes()))[:16]
    assert [r['row_id'] for r in rows]==config['selected_row_ids']
    scenes=list(unpickle_data(str(Path(config['data_root'])/'train_v3scans.pkl')))[0]
    mean=np.array([109.8,97.2,83.8])/256
    result=[]
    for row in rows:
        scene=scenes[row['scan_id']]
        xyz=np.asarray(scene.orig_pc,dtype=np.float32)
        centered=np.asarray(scene.color-mean,dtype=np.float32)
        native=np.concatenate([xyz,centered],axis=1)
        assert native.shape==(50000,6)
        assert hashlib.sha256(native.tobytes()).hexdigest()==row['point_sha256']
        rgb=np.asarray(scene.color,dtype=np.float32)
        assert np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
        detector=Path(config['data_root'])/'group_free_pred_bboxes/group_free_pred_bboxes_train'/(row['scan_id']+'.npy')
        det=np.load(str(detector),allow_pickle=True).item()
        bounds=np.asarray(det['box'])
        # Match the actual dataset's min/max -> center/size -> float32 slot conversion.
        boxes=np.concatenate([(bounds[:,:3]+bounds[:,3:])*.5,bounds[:,3:]-bounds[:,:3]],1).astype(np.float32)
        assert boxes.shape[0]==len(det['class']) and boxes.shape[1]==6
        values={};objects=[]
        for slot,box in enumerate(boxes):
            inside=((xyz>=box[:3]-box[3:]*.5)&(xyz<=box[:3]+box[3:]*.5)).all(1)
            points=np.concatenate([xyz[inside],rgb[inside]],1)
            values['object_%03d'%slot]=points
            objects.append(dict(slot=slot,points=len(points),box=box.tolist(),detector_class=str(det['class'][slot])))
        filename='row_%05d.npz'%row['row_id']
        np.savez_compressed(str(out/filename),**values)
        record=dict(row_id=row['row_id'],scan_id=row['scan_id'],native_point_sha256=row['point_sha256'],
                    detector_file_sha256=sha(detector),file=filename,file_sha256=sha(out/filename),objects=objects)
        result.append(record)
        print('EXPORTED',row['row_id'],row['scan_id'],len(objects),flush=True)
    objects=[o for r in result for o in r['objects']]
    receipt=dict(status='complete',rows=result,total_objects=len(objects),
                 empty_crops=sum(o['points']==0 for o in objects),
                 crops_below_384=sum(o['points']<384 for o in objects),
                 crops_at_least_384=sum(o['points']>=384 for o in objects),
                 minimum_points=min(o['points'] for o in objects),maximum_points=max(o['points'] for o in objects),
                 native_points_exactly_match_prior=True,gt_used_for_crop=False,
                 formal_rows=0,mcln_forwards=0,optimizer_steps=0,
                 manifest_sha256=sha(args.manifest))
    with (out/'export_receipt.json').open('x') as f:
        json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
