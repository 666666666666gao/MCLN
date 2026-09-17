"""Bind actual native ReferIt validation annotations and mesh inputs without inference."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();root=args.spec.parent;spec=json.loads(args.spec.read_bytes());started=time.time()
    assert sha(__file__)==spec['script_sha256']
    assert sha(spec['selection_manifest'])==spec['selection_manifest_sha256']
    manifest=json.loads(Path(spec['selection_manifest']).read_bytes())
    source=Path(manifest['model_source']);data=Path(manifest['data_root'])
    assert sha(source/'appearance_source_manifest.json')==manifest['source_manifest_sha256']
    for name,digest in json.loads((source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source/name)==digest,name
    annotation_path=Path(spec['selection_manifest']).parent/'annotation_receipt.json'
    assert sha(annotation_path)==manifest['input_files'][str(annotation_path)]
    annotation=json.loads(annotation_path.read_bytes())
    for path,item in annotation['annotations_and_split_files'].items():
        actual=Path(path) if '/DATA_ROOT/' in path else source/'data/meta_data'/Path(path).name
        assert sha(actual)==item['sha256'],str(actual)
    assert sha(spec['full_point_manifest'])==spec['full_point_manifest_sha256']
    full=json.loads(Path(spec['full_point_manifest']).read_bytes());assert full['data_root']==manifest['data_root']
    points=data/'val_v3scans.pkl'
    assert points.stat().st_size==annotation['scans']['val']['pickle_metadata']['bytes']
    points_sha=sha(points)
    os.chdir(str(source));sys.path.insert(0,str(source))
    import torch
    from src.joint_det_dataset import Joint3DDataset,unpickle_data
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    dataset=object.__new__(Joint3DDataset)
    dataset.split='val';dataset.data_path=manifest['data_root'];dataset.use_sacr_source=False
    dataset.overfit=False;dataset.skip_missing_superpoints=True;dataset._scene_graph_parse=lambda rows:None
    dataset.scans=list(unpickle_data(str(points)))[0]
    assert len(dataset.scans)==312
    for scene in dataset.scans:
        name=scene+'_superpoint.pth'
        assert sha(data/'superpoints/val'/name)==full['superpoint_files']['val'][name],scene
    outputs={}
    for name,count in [('nr3d',7899),('sr3d',17726)]:
        partition_path=Path(spec['partitions'][name]['path'])
        assert sha(partition_path)==spec['partitions'][name]['sha256']
        partition=json.loads(partition_path.read_bytes())
        rows=dataset.load_annos(name)
        assert len(rows)==count==annotation['protocols'][name]['val']['language_rows']
        scenes=sorted({row['scan_id'] for row in rows})
        assert scenes==partition['formal_scenes']
        assert not {s.split('_')[0] for s in scenes}&(set(partition['fit_physical_scenes'])|set(partition['holdout_physical_scenes']))
        raw_keys=[{key:row[key] for key in ['scan_id','target_id','utterance','dataset']} for row in rows]
        raw_hashes=[hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest() for row in raw_keys]
        identity_digest=hashlib.sha256(json.dumps(raw_keys,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        contract=dict(status='pass',dataset=name,expected_formal_rows=count,data_root=str(data)+'/',
            source_manifest_sha256=manifest['source_manifest_sha256'],dataset_source=str(source),
            annotation_receipt_sha256=sha(annotation_path),input_manifest_sha256=spec['selection_manifest_sha256'],
            full_point_manifest_sha256=spec['full_point_manifest_sha256'],partition_sha256=sha(partition_path),
            points_path=str(points),points_sha256=points_sha,points_bytes=points.stat().st_size,
            val_superpoints_verified=312,formal_scenes=scenes,
            language_row_keys_sha256=raw_hashes,raw_identity_order_sha256=identity_digest,
            input_protocol='native butd_cls instance boxes and predicted categories',
            butd=False,butd_cls=True,butd_gt=False,primary_mode='bbs',mask_gate=False,
            model_forwards=0,formal_rows_executed=0,optimizer_steps=0)
        path=root/(name+'_formal_input_contract.json');path.write_text(json.dumps(contract,indent=2)+'\n')
        outputs[name]=dict(rows=count,scenes=len(scenes),contract_sha256=sha(path),contract_bytes=path.stat().st_size,
                           raw_identity_order_sha256=identity_digest,train_formal_physical_overlap=[])
    assert not torch.cuda.is_initialized()
    receipt=dict(status='pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        datasets=outputs,points_sha256=points_sha,points_bytes=points.stat().st_size,val_superpoints_verified=312,
        script_sha256=sha(__file__),model_forwards=0,optimizer_steps=0,formal_rows_evaluated=0,
        text_graphs_parsed=False,point_samples_constructed=0,torch_cuda_initialized=False,
        elapsed_seconds=time.time()-started,scope='native validation annotation and input contract only; not accuracy')
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
