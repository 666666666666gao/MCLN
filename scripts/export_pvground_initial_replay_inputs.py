"""Freeze the actual first batch of the completed initial module-holdout evaluation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    begin=time.time()
    spec=json.loads((args.training_root/'spec.json').read_bytes())
    manifest=json.loads(Path(spec['input_manifest']).read_bytes())
    source=Path(manifest['model_source'])
    assert sha(source/'appearance_source_manifest.json')==manifest['source_manifest_sha256']
    source_manifest=json.loads((source/'appearance_source_manifest.json').read_bytes())
    for name,digest in source_manifest['files'].items():assert sha(source/name)==digest,name
    assert sha(manifest['split_protocol'])==manifest['split_protocol_sha256']
    partitions=json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    ids=partitions['holdout'][:8]
    originals=[json.loads(line) for line in (args.training_root/'initial/rows.jsonl').read_text().splitlines()[:8]]
    assert [r['row_id'] for r in originals]==ids
    import numpy as np
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    os.chdir(str(source));sys.path.insert(0,str(source))
    from src.joint_det_dataset import Joint3DDataset
    class Dataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            assert len(annos)==36665
            super()._scene_graph_parse([annos[i] for i in ids])
    dataset=Dataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='train',
        data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
        detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    dataset.augment=False
    # The original evaluator uses DataLoader worker seeds; no-augmentation getitem
    # is checked against actual per-row point hashes and GT, rather than assumed equal.
    random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
    samples=[];labels=[];records=[]
    for row_id,original in zip(ids,originals):
        row=dataset[row_id]
        inputs=dict(point_clouds=torch.from_numpy(row['point_clouds']),text=row['utterances'],
            det_boxes=torch.from_numpy(row['all_detected_boxes']),
            det_bbox_label_mask=torch.from_numpy(row['all_detected_bbox_label_mask']),
            det_class_ids=torch.from_numpy(row['all_detected_class_ids']),superpoint=row['superpoint'])
        point_sha=hashlib.sha256(row['point_clouds'].tobytes()).hexdigest()
        gt=np.concatenate([row['center_label'][0,:3],row['size_gts'][0]])
        assert original['point_sha256']==point_sha
        assert np.array_equal(gt,np.asarray(original['root_box'],dtype=np.float32))
        maps={k:torch.from_numpy(row[k][0]) for k in ['positive_map','modify_positive_map','pron_positive_map','rel_positive_map','other_entity_map']}
        samples.append(inputs);labels.append(dict(root_box=torch.from_numpy(gt),**maps))
        records.append(dict(row_id=row_id,scan_id=row['scan_ids'],target_id=int(row['target_id']),point_sha256=point_sha,text=row['utterances'],
            input_tensor_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in inputs.items() if torch.is_tensor(v)}))
    batch={k:torch.stack([r[k] for r in samples]) for k in samples[0] if k!='text'}
    batch['text']=[r['text'] for r in samples]
    targets={k:torch.stack([r[k] for r in labels]) for k in labels[0]}
    args.output.mkdir()
    torch.save(dict(inputs=batch,targets=targets),str(args.output/'first_batch.pt'))
    receipt=dict(status='pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        elapsed_seconds=time.time()-begin,rows=records,batch_size=8,source_manifest_sha256=manifest['source_manifest_sha256'],
        data_root=manifest['data_root'],input_file_sha256=sha(args.output/'first_batch.pt'),
        original_initial_receipt_sha256=sha(args.training_root/'initial/receipt.json'),
        script_sha256=sha(__file__),point_and_root_gt_match_actual_initial=True,
        scope='first actual module-holdout batch only; targets stored separately and never enter model',
        model_forwards=0,optimizer_steps=0,formal_rows=0,weights_loaded=0)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('REPLAY_INPUT_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':main()
