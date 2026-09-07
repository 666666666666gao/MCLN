"""CPU-only native butd_cls region export for frozen appearance preprocessing.

The protocol supplies instance boxes and predicted classes. Crops retain all
points inside those boxes; target IDs and instance masks do not clean crops.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    manifest_path = p.parse_args().manifest.resolve()
    root = manifest_path.parent
    plan = json.loads(manifest_path.read_text())
    source = Path(plan['model_source'])
    source_manifest = source / 'appearance_source_manifest.json'
    assert sha(source_manifest) == plan['source_manifest_sha256']
    for name, digest in json.loads(source_manifest.read_text())['files'].items():
        assert sha(source/name) == digest, name
    for path, digest in plan['input_files'].items():
        assert sha(path) == digest, path
    audits = {k: json.loads(Path(v).read_text()) for k,v in plan['crop_receipts'].items()}
    assert len(audits['nr3d']['rows']) == 511 and len(audits['sr3d']['rows']) == 1018
    by_scene = {}
    for dataset, audit in audits.items():
        for row in audit['rows']:
            scan_id = row['scan_id']
            if scan_id in by_scene:
                for name in ['point_cloud_sha256', 'input_boxes_sha256', 'object_ids', 'crop_point_counts']:
                    assert by_scene[scan_id]['row'][name] == row[name], (scan_id,name)
                by_scene[scan_id]['datasets'].append(dataset)
            else:
                by_scene[scan_id] = dict(row=row, datasets=[dataset])
    assert len(by_scene) == 1061
    os.chdir(str(source));sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset, unpickle_data, read_label_mapping
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    dataset = object.__new__(Joint3DDataset)
    dataset.split, dataset.augment = 'train', False
    dataset.label_map = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='id')
    cls = json.loads((source/'data/cls_results.json').read_text())
    scenes = list(unpickle_data(str(Path(plan['data_root'])/'train_v3scans.pkl')))[0]
    mean_rgb = np.array([109.8,97.2,83.8])/256
    counts_by_dataset = {d:dict(scenes=0,slots=0,available_384=0,noncontiguous_scenes=0) for d in audits}
    records=[];start=time.time()
    for scan_id, entry in sorted(by_scene.items()):
        old=entry['row'];scan=scenes[scan_id]
        scan.pc=np.copy(scan.orig_pc)
        _, boxes, valid = dataset._get_scene_objects(scan)
        boxes=boxes.astype(np.float32)
        slots=np.flatnonzero(valid)
        assert slots.tolist()==old['object_ids']
        assert hashlib.sha256(boxes.tobytes()).hexdigest()==old['input_boxes_sha256']
        xyz=np.asarray(scan.orig_pc,dtype=np.float32)
        rgb=np.asarray(scan.color,dtype=np.float32)
        native=np.concatenate([xyz,np.asarray(scan.color-mean_rgb,dtype=np.float32)],axis=1)
        assert native.shape==(50000,6)
        assert hashlib.sha256(native.tobytes()).hexdigest()==old['point_cloud_sha256']
        assert np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
        predictions=np.asarray(cls[scan_id]);predictions=predictions[predictions>-1]
        assert len(predictions)==len(slots)
        counts=[]
        for slot in slots:
            box=boxes[slot]
            inside=((xyz>=box[:3]-box[3:]*.5)&(xyz<=box[:3]+box[3:]*.5)).all(axis=1)
            counts.append(int(inside.sum()))
        assert counts==old['crop_point_counts'],scan_id
        availability=[v>=384 for v in counts]
        row=dict(scan_id=scan_id,datasets=entry['datasets'],slot_ids=slots.tolist(),
                 padded_slot_count=len(valid),boxes=boxes[slots].tolist(),predicted_class_ids=predictions.tolist(),
                 crop_point_counts=counts,available_384=availability,
                 native_point_sha256=old['point_cloud_sha256'],padded_boxes_sha256=old['input_boxes_sha256'])
        records.append(row)
        for d in entry['datasets']:
            summary=counts_by_dataset[d]
            summary['scenes']+=1;summary['slots']+=len(slots)
            summary['available_384']+=sum(availability)
            summary['noncontiguous_scenes']+=int(slots.tolist()!=list(range(len(slots))))
        if len(records)==1 or len(records)%128==0:
            print('REFERIT SLOT EXPORT',json.dumps(dict(scenes=len(records),total=1061,elapsed_seconds=time.time()-start)),flush=True)
    with (root/'scene_slots.json').open('x') as f:json.dump(records,f,sort_keys=True,allow_nan=False)
    for path,digest in plan['input_files'].items():assert sha(path)==digest,path
    receipt=dict(schema='mcln-referit3d-pretrained-slot-export-v1',status='complete',
        unique_scenes=1061,by_dataset=counts_by_dataset,scene_slots_sha256=sha(root/'scene_slots.json'),
        manifest_sha256=sha(manifest_path),model_source_manifest_sha256=plan['source_manifest_sha256'],
        point_boxes_slots_and_crop_counts_match_prior=True,uses_protocol_instance_boxes=True,
        uses_predicted_classes=True,target_ids_used_for_cropping=False,instance_masks_used_for_cropping=False,
        augmentation=False,appearance_embeddings_computed=False,native_model_forwards=0,gpu_forwards=0,
        optimizer_steps=0,formal_rows=0,elapsed_seconds_excluding_scene_loading=time.time()-start,
        remaining='Encode and bind features by original slot IDs; validate augmented inputs and model loading after Scan qualification.')
    with (root/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
    print('REFERIT SLOT EXPORT COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
