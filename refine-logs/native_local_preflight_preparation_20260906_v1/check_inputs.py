import datetime, gc, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
root=Path(sys.argv[1])
expected=json.loads((root/'expected.json').read_text())
source=Path(expected['model_source'])
for name,digest in expected['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((source/'native_source_manifest.json').read_bytes()).hexdigest()==expected['source_manifest_sha256']
for name,digest in json.loads((source/'native_source_manifest.json').read_text())['files'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
os.chdir(str(source))
sys.path.insert(0,str(source))
import numpy as np
import torch
from main_utils import parse_option, prepare_source_moe_gate_checkpoint_config
from train_dist_mod import TrainTester
from src.joint_det_dataset import Joint3DDataset
assert not torch.cuda.is_available()
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('native_local_preflight',str(root/'run_native_candidate_local_preflight.py'))
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
selection=json.loads((root/'preflight_rows.json').read_text())
annotation=json.loads((root/'annotation_receipt.json').read_text())
assert hashlib.sha256((root/'preflight_rows.json').read_bytes()).hexdigest()==annotation['preflight_rows_sha256']
for path,item in annotation['annotations_and_split_files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==item['sha256']
contract=json.loads((root/'nr_contract.json').read_text())
def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
results={}
started=time.time()
for dset in ['nr3d','sr3d']:
    argv=list(contract['eval_argv'])
    for key,value in [('--dataset',dset),('--test_dataset',dset),('--expected_eval_sample_count',str(annotation['protocols'][dset]['val']['total_rows']))]:
        argv[argv.index(key)+1]=value
    sys.argv=['native-input-check']+argv+['--use_candidate_local_visual']
    args=prepare_source_moe_gate_checkpoint_config(parse_option())
    assert args.use_color and not args.use_height and not args.use_multiview
    assert args.butd_cls and args.joint_det and args.detect_intermediate
    seed(0)
    dataset=entry.build_probe_dataset(Joint3DDataset,args,annotation,selection[dset],dset)
    records=[]
    for phase,augment,value in [('eval',False,1000),('train',True,2000)]:
        dataset.augment=augment
        seed(value)
        loader=torch.utils.data.DataLoader(dataset,batch_size=12,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(0))
        for batch_index,batch in enumerate(loader):
            inputs=TrainTester._get_inputs(batch)
            size=12 if batch_index==0 else 4
            assert inputs['point_clouds'].shape==(size,50000,6)
            assert inputs['superpoint'].shape==(size,50000)
            assert inputs['det_boxes'].shape==(size,132,6)
            assert inputs['det_class_ids'].shape==(size,132)
            assert torch.isfinite(inputs['point_clouds']).all()
            assert torch.isfinite(inputs['det_boxes']).all()
            assert inputs['det_bbox_label_mask'].any(1).all()
            expected_dataset=dset if batch_index==0 else 'scannet'
            assert list(batch['sample_dataset'])==[expected_dataset]*size
            assert list(batch['scan_ids'])==[r['scan_id'] for r in selection[dset][batch_index*12:batch_index*12+size]]
            assert torch.equal(batch['all_detected_boxes'],batch['all_bboxes'])
            assert torch.equal(batch['all_detected_bbox_label_mask'],batch['all_bbox_label_mask'])
            records.append({'phase':phase,'rows':size,'sample_dataset':expected_dataset,
                            'scan_ids':list(batch['scan_ids']),
                            'point_sha256':[hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in inputs['point_clouds']],
                            'is_view_dep':batch['is_view_dep'].tolist(),
                            'gt_objects':batch['box_label_mask'].sum(1).tolist(),
                            'input_objects':inputs['det_bbox_label_mask'].sum(1).tolist(),
                            'token_map_shape':list(batch['positive_map'].shape),
                            'butd_cls_box_and_valid_mask_identity':True})
            print('NATIVE INPUT',json.dumps({'dataset':dset,'phase':phase,'batch':batch_index,'rows':size,'sample_dataset':expected_dataset}),flush=True)
        assert batch_index==1
    results[dset]={'unique_rows':16,'constructed_point_samples':32,'records':records}
    del dataset,loader,batch,inputs
    gc.collect()
assert not torch.cuda.is_initialized()
result={'schema':'mcln-native-local-real-input-cpu-preparation-v1','status':'actual_cpu_inputs_pass',
        'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'elapsed_seconds':time.time()-started,'protocols':results,'unique_input_rows':32,
        'point_samples_constructed':64,'gpu_forwards':0,'optimizer_steps':0,'checkpoint_writes':0,
        'source_manifest_sha256':expected['source_manifest_sha256'],'files':expected['files'],
        'limits':'Actual native tokenizer, parser, raw point sampling, superpoints, augmentations and butd_cls input path only. No model forward, gradient or native evaluation. Scan formal promotion remains required before GPU preflight.'}
with (root/'receipt.json').open('x') as stream:
    json.dump(result,stream,indent=2,sort_keys=True)
print('NATIVE INPUT COMPLETE',json.dumps({key:result[key] for key in ['status','time_cst','elapsed_seconds','unique_input_rows','point_samples_constructed','gpu_forwards']}),flush=True)
