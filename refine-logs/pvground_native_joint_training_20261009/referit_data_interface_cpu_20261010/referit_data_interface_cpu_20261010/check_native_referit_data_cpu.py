"""Native data items and NLP preprocessing, limited annotations, no PV model or GPU computation."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

root=Path(__file__).resolve().parent
settings=json.loads((root/'CHECK_SPEC.json').read_bytes())
source=Path(settings['warm_source'])
assert os.environ['CUDA_VISIBLE_DEVICES']==''
os.chdir(str(source))
sys.path.insert(0,str(source))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
from src.joint_det_dataset import Joint3DDataset
assert Path(inspect.getfile(Joint3DDataset)).resolve()==source/'src/joint_det_dataset.py'
data=Path(settings['data_root'])
assert all((data/(split+'_v3scans.pkl')).is_file() for split in ('train','val'))

def seed():
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)

def describe(item,anno,dataset,split):
    pc=item['point_clouds']
    masks=item['gt_masks']
    labels=item['point_instance_label']
    sp=item['superpoint']
    gt_valid=item['box_label_mask'].astype(bool)
    assert pc.shape==(50000,6) and pc.dtype==np.float32 and np.isfinite(pc).all()
    assert masks.shape[1]==labels.shape[0]==sp.numel()==pc.shape[0]
    assert np.array_equal(masks[0].astype(bool),labels==0)
    assert gt_valid[0] and (masks[0]>0).any()
    assert item['language_dataset']==dataset and item['sample_dataset']==anno['dataset']
    expected_root=anno['target_id'] if isinstance(anno['target_id'],int) else anno['target_id'][0]
    assert item['target_id']==expected_root
    assert np.array_equal(item['all_detected_boxes'],item['all_bboxes'])
    assert np.array_equal(item['all_detected_bbox_label_mask'],item['all_bbox_label_mask'])
    assert np.isfinite(item['positive_map']).all() and item['positive_map'][0].sum()>0
    valid_proposals=item['all_detected_bbox_label_mask'].astype(bool)
    assert valid_proposals.any()
    classes=np.array(current.cls_results[anno['scan_id']])
    assert np.array_equal(item['all_detected_class_ids'][valid_proposals],classes[classes>-1])
    return dict(split=split,scan_id=anno['scan_id'],sample_dataset=item['sample_dataset'],
        language_dataset=item['language_dataset'],target_id=int(expected_root),
        point_cloud_shape=list(pc.shape),point_cloud_dtype=str(pc.dtype),
        GT_mask_shape=list(masks.shape),root_GT_points=int(masks[0].sum()),
        valid_GT_count=int(gt_valid.sum()),superpoint_count=int(torch.unique(sp).numel()),
        positive_map_shape=list(item['positive_map'].shape),
        valid_proposals=int(valid_proposals.sum()),
        proposals_equal_scene_GT_boxes=True,proposal_classes_equal_cls_results=True,
        utterance_sha256=hashlib.sha256(item['utterances'].encode()).hexdigest())

cases={}
for dataset in ('nr3d','sr3d'):
    started=time.monotonic()
    flags=settings['author_flags'][dataset]
    assert flags['butd_cls'] and not flags['butd'] and not flags['butd_gt']
    assert flags['joint_det'] and flags['detect_intermediate']
    case=dict(author_flags=flags,annotation_cap_per_source=128,items=[])
    for split in ('train','val'):
        seed()
        current=Joint3DDataset(dataset_dict={dataset:1,'scannet':10},test_dataset=dataset,
            split=split,overfit=True,data_path=settings['data_root'],
            use_color=flags['use_color'],use_height=flags['use_height'],use_multiview=flags['use_multiview'],
            detect_intermediate=flags['detect_intermediate'],butd=flags['butd'],
            butd_gt=flags['butd_gt'],butd_cls=flags['butd_cls'],augment_det=flags['augment_det'])
        ref_index=next(i for i,anno in enumerate(current.annos) if anno['dataset']==dataset)
        ref=current[ref_index]
        case['items'].append(describe(ref,current.annos[ref_index],dataset,split))
        if split=='train':
            det_index=next(i for i,anno in enumerate(current.annos) if anno['dataset']=='scannet')
            detection=current[det_index]
            case['items'].append(describe(detection,current.annos[det_index],dataset,split))
            loader=DataLoader(Subset(current,[ref_index,det_index]),batch_size=2,shuffle=False,num_workers=0)
            batch=next(iter(loader))
            assert batch['point_clouds'].shape==(2,50000,6)
            assert batch['sample_dataset']==[dataset,'scannet']
            assert batch['language_dataset']==[dataset,dataset]
            assert all(value.device.type=='cpu' for value in batch.values() if isinstance(value,torch.Tensor))
            case['train_collated_shape']=list(batch['point_clouds'].shape)
            case['train_limited_annotation_count']=len(current)
            del batch,loader,detection
        else:
            case['val_limited_annotation_count']=len(current)
        assert current.augment==(split=='train') and current.joint_det==(split=='train')
        del current,ref
        gc.collect()
    case['elapsed_seconds']=time.monotonic()-started
    cases[dataset]=case
assert not torch.cuda.is_initialized()
result=dict(status='ACTUAL_NATIVE_REFERIT_LIMITED_DATA_CPU_PASS',cases=cases,
    seed=2027,real_item_calls=10,distinct_selected_annotations=6,native_dataset_constructors=4,
    train_collations=2,annotation_limit_per_source=128,
    full_dataset_lengths_verified=False,full_training_budget_verified=False,
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    CUDA_initialized=False,PV_model_constructors=0,PV_forwards=0,criterion_calls=0,
    native_CPU_NLP_preprocessing_used=True,
    optimizer_steps=0,saved_weights=0,saved_candidate_arrays=0,
    current_training_queries=0,formal_accuracy=None,GPU_training_admission=False,full_goal_complete=False)
(root/'REFERIT_DATA_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
