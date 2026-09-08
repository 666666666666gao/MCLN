"""Persist one real CPU Nr3D/joint-detection batch for later native preflight."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();started=time.time()
    spec=json.loads(args.spec.read_bytes());root=args.spec.parent
    selection=Path(spec['selection_root']);runtime=Path(spec['runtime'])
    assert sha(selection/'manifest.json')==spec['selection_manifest_sha256']
    manifest=json.loads((selection/'manifest.json').read_bytes())
    source=Path(manifest['model_source'])
    assert sha(source/'appearance_source_manifest.json')==manifest['source_manifest_sha256']
    for name,digest in json.loads((source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source/name)==digest,name
    for name in ['fixed_selection.py','preflight_rows.json','annotation_receipt.json']:
        assert sha(selection/name)==manifest['input_files'][str(selection/name)],name
    point_archive=Path(manifest['data_root'])/'train_v3scans.pkl'
    assert sha(point_archive)==manifest['input_files'][str(point_archive.resolve())]
    annotation=json.loads((selection/'annotation_receipt.json').read_bytes())
    for path,info in annotation['annotations_and_split_files'].items():
        actual=Path(path) if '/DATA_ROOT/' in path else source/'data/meta_data'/Path(path).name
        assert sha(actual)==info['sha256'],str(actual)
    selections=json.loads((selection/'preflight_rows.json').read_bytes())['nr3d']
    chosen=[0,1,2,3,12,13,14,15]
    for index in chosen:
        name=selections[index]['scan_id']+'_superpoint.pth'
        assert sha(Path(manifest['data_root'])/'superpoints/train'/name)==manifest['superpoints'][name]
    env=json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
    bundle=json.loads((runtime/'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']['files']
    for name in ['prepare_data.py','wandb_config.yaml']:
        assert sha(runtime/'PV-Ground'/name)==bundle[name]['sha256']
    import numpy as np
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed'])
    os.chdir(str(source));sys.path[:0]=[str(source),str(selection)]
    from src.joint_det_dataset import Joint3DDataset
    from fixed_selection import build_probe_dataset
    dataset_args=SimpleNamespace(data_root=manifest['data_root'],use_color=True,use_height=False,
        use_multiview=False,detect_intermediate=True,butd=False,butd_gt=False,butd_cls=True,
        augment_det=False,skip_missing_superpoints=True)
    print('NR_BATCH_LOADING',flush=True)
    dataset=build_probe_dataset(Joint3DDataset,dataset_args,annotation,selections,'nr3d')
    assert dataset.augment and dataset.joint_det
    samples=[];rows=[]
    for index in chosen:
        sample=dataset[index]
        assert sample['point_clouds'].shape==(50000,6)
        assert np.array_equal(sample['all_detected_boxes'],sample['all_bboxes'])
        assert np.array_equal(sample['all_detected_bbox_label_mask'],sample['all_bbox_label_mask'])
        assert np.isin(sample['gt_masks'],[0,1]).all()
        # Exact binary storage; restore float32 before the native loss consumes it.
        sample['gt_masks']=sample['gt_masks'].astype(np.bool_)
        samples.append(sample)
        rows.append(dict(selection_index=index,dataset=selections[index]['dataset'],
                         scan_id=selections[index]['scan_id'],
                         detected_objects=int(sample['all_detected_bbox_label_mask'].sum())))
    batch=torch.utils.data._utils.collate.default_collate(samples)
    assert batch['point_clouds'].shape==(8,50000,6) and batch['gt_masks'].dtype==torch.bool
    sys.path.insert(0,str(runtime/'PV-Ground'))
    from prepare_data import DataProcessor
    from pcdet.config import cfg,cfg_from_yaml_file
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    processor=DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),True,6)
    voxels=processor.collate_batch([processor.forward(dict(points=p.numpy().copy(),use_lead_xyz=True))
                                    for p in batch['point_clouds']])
    assert np.array_equal(voxels['points'][:,0],np.repeat(np.arange(8),50000))
    assert np.array_equal(voxels['points'][:,1:].reshape(8,50000,6),batch['point_clouds'].numpy())
    inputs={k:torch.from_numpy(voxels[k]).float() for k in ['points','voxels','voxel_coords','voxel_num_points']}
    inputs.update(batch_size=8,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
                  det_bbox_label_mask=batch['all_detected_bbox_label_mask'],
                  det_class_ids=batch['all_detected_class_ids'],superpoint=batch['superpoint'])
    assert len(inputs)==10 and not any(k in inputs for k in ['gt_masks','center_label','size_gts','positive_map'])
    path=root/'native_batch.pt'
    torch.save({'inputs':inputs,'native_batch':batch,'rows':rows},str(path))
    # Read back the complete stored payload before accepting the reusable batch.
    loaded=torch.load(str(path),map_location='cpu')
    for key,value in batch.items():
        if torch.is_tensor(value):assert torch.equal(loaded['native_batch'][key],value),key
        else:assert loaded['native_batch'][key]==value,key
    for key,value in inputs.items():
        if torch.is_tensor(value):assert torch.equal(loaded['inputs'][key],value),key
        else:assert loaded['inputs'][key]==value,key
    assert loaded['rows']==rows
    assert not torch.cuda.is_initialized()
    receipt=dict(status='pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        scope='four fixed Nr3D and four joint-detection training rows; CPU integration only, not training-distribution metrics',
        rows=rows,seed=spec['seed'],augmentation=True,batch_size=8,
        selection_manifest_sha256=spec['selection_manifest_sha256'],source_manifest_sha256=manifest['source_manifest_sha256'],
        input_path=str(path),input_bytes=path.stat().st_size,input_sha256=sha(path),script_sha256=sha(__file__),
        gt_masks_storage='exact bool; restore float32 before native loss',
        input_protocol='butd_cls instance boxes plus predicted categories; target labels remain outside model input dict',
        input_shapes={k:list(v.shape) for k,v in inputs.items() if torch.is_tensor(v)},
        model_forwards=0,optimizer_steps=0,formal_rows=0,weights_loaded=0,torch_cuda_initialized=False,
        elapsed_seconds=time.time()-started)
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('NR_NATIVE_BATCH_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':main()
