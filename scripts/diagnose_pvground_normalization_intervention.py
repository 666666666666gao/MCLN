"""Fixed D BatchNorm running-state intervention; no optimizer or saved mixed model."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from evaluate import expanded_parent_state, terminal_state, sha, write_json, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    root = args.spec.parent
    spec = json.loads(args.spec.read_bytes())
    begin = time.time()
    for name, digest in spec['files'].items():
        assert sha(root/name) == digest, name
    training = Path(spec['training_root'])
    assert (training/'controller.exit').read_text().strip() == '0'
    train_spec = json.loads((training/'spec.json').read_bytes())
    receipt = json.loads((training/'receipt.json').read_bytes())
    assert receipt['status'] == 'complete' and receipt['training_steps'] == 3723
    assert sha(training/'spec.json') == spec['training_spec_sha256'] == receipt['spec_sha256']
    assert sha(training/'terminal.pth') == spec['terminal_sha256'] == receipt['terminal_sha256']
    runtime = Path(train_spec['runtime'])
    environment = json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest() == train_spec['env_spec_sha256']
    source = Path(train_spec['model_source'])
    assert sha(train_spec['source_port']) == train_spec['source_port_sha256']
    for name, digest in json.loads(Path(train_spec['source_port']).read_bytes())['files'].items():
        assert sha(source/name) == digest, name
    for filename, key in [('pvground_source_query.py','source_query_module_sha256'),
                          ('pvground_observation_query.py','observation_module_sha256'),
                          ('pvground_task_observation_query.py','task_module_sha256')]:
        assert sha(root/filename) == train_spec[key] == receipt[key]
    fixtures = Path(spec['fixtures'])
    assert sha(fixtures/'receipt.json') == spec['fixture_receipt_sha256']
    fixture_receipt = json.loads((fixtures/'receipt.json').read_bytes())
    assert fixture_receipt['status'] == 'pass' and fixture_receipt['separate_training_labels']
    assert [r['training_row_id'] for r in fixture_receipt['rows']] == [0,173,237,455]
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from types import SimpleNamespace

    def reset_rng():
        random.seed(2027); np.random.seed(2027)
        torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(Path(train_spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source/'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name,digest in json.loads((dataset_source/'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source/name) == digest,name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    os.chdir(str(dataset_source)); sys.path.insert(0,str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == dataset_source/'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(source)); sys.path.insert(0,str(source))
    from models.pv_ground import PVGround
    from pcdet.config import cfg, cfg_from_yaml_file
    from prepare_data import DataProcessor
    from main_utils import BaseTrainTester
    from models.losses import box_cxcyczwhd_to_xyzxyz, _iou3d_par
    assert Path(sys.modules['models.pv_ground'].__file__).resolve() == source/'models/pv_ground.py'
    checkpoint = environment['weight_dirs']['scanrefer']
    assert sha(checkpoint['path']) == checkpoint['sha256'] == receipt['checkpoint_sha256']
    saved = torch.load(checkpoint['path'],map_location='cpu')
    config = saved['config']
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert all(name.startswith('module.') for name in saved['model'])
    parent = {name[7:]:value for name,value in saved['model'].items()}
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    model = PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=fixture_receipt['data_root'],self_attend=config.self_attend)
    initial = expanded_parent_state(model,parent)
    saved_delta = torch.load(str(training/'terminal.pth'),map_location='cpu')
    manifest = json.loads(Path(train_spec['input_manifest']).read_bytes())
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert saved_delta['step'] == 3723 and Counter(saved_delta['row_ids']) == Counter(partitions['fit'])
    assert saved_delta['parent_checkpoint_sha256'] == checkpoint['sha256']
    assert saved_delta['spec_sha256'] == sha(training/'spec.json')
    for key in ['source_query_read','source_query_module_sha256','source_port_sha256',
                'observation_state','observation_module_sha256','task_read','task_module_sha256']:
        assert saved_delta[key] == train_spec[key] == receipt[key], key
    terminal = terminal_state(model,initial,saved_delta['state_delta'])
    model.load_state_dict(terminal,strict=True)
    assert len(terminal) == 1271
    assert all(torch.equal(value,terminal[name]) for name,value in model.state_dict().items())
    model.cuda().eval()
    reader = model.decoder[-1].source_query_read
    assert reader.enabled and model.decoder[-1].task_read

    reference_result = json.loads((root/'reference_diagnostic.json').read_bytes())
    reference_rows = json.loads((root/'reference_rows.json').read_bytes())
    reference_selection = json.loads((root/'reference_selection.json').read_bytes())
    assert reference_result['status']=='complete' and reference_result['terminal_sha256']==spec['terminal_sha256']
    chosen = [r['row_id'] for r in reference_selection['rows']]
    old_by_id = {r['row_id']:r for r in reference_rows}
    assert len(chosen)==len(set(chosen))==128 and chosen==[r['row_id'] for r in reference_rows]
    processor = DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),False,6)
    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            assert len(annos)==36665
            actual={'fit':[],'holdout':[]}
            for index,row in enumerate(annos):
                row['_local_training_id']=index
                code=(manifest['split_salt']+chr(0)+row['scan_id'].split('_')[0]).encode()
                fold=int(hashlib.sha256(code).hexdigest()[:8],16)%5
                actual['holdout' if fold==0 else 'fit'].append(index)
            assert actual==partitions
            assert not set(chosen).intersection(partitions['holdout'])
            assert len({annos[i]['scan_id'].split('_')[0] for i in chosen})==128
            super()._scene_graph_parse([annos[i] for i in chosen])
        def __getitem__(self,index):
            result=super().__getitem__(index)
            result['local_training_id']=self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'],[0,1]).all()
            result['gt_masks']=result['gt_masks'].astype(np.bool_)
            return result
    os.chdir(str(dataset_source))
    verify_scanrefer_superpoints(manifest['data_root'],'train',manifest['superpoint_files']['train'])
    dataset=FitDataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='train',
        data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
        detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    dataset.augment=False
    assert [r['_local_training_id'] for r in dataset.annos]==list(range(36665))
    for expected,index in zip(reference_selection['rows'],chosen):
        assert dataset.annos[index]['scan_id']==expected['scan_id']
        assert dataset.annos[index]['utterance']==expected['text']
    write_json(root/'input_selection.json',reference_selection)
    reset_rng()
    loader=DataLoader(Subset(dataset,chosen),batch_size=8,shuffle=False,num_workers=2,
        generator=torch.Generator().manual_seed(2027),pin_memory=True,drop_last=False)
    bn_keys=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            assert module.track_running_stats and not module.training
            for suffix in ['running_mean','running_var','num_batches_tracked']:
                key=name+'.'+suffix
                assert key in initial and key in terminal
                bn_keys.append(key)
    census=json.loads((root/'census.json').read_bytes())
    assert census['terminal_sha256']==spec['terminal_sha256']
    assert set(bn_keys)=={r['name'] for r in census['rows']} and len(bn_keys)==252
    buffers=dict(model.named_buffers())
    parent_bn={n:initial[n].cuda() for n in bn_keys}
    terminal_bn={n:terminal[n].cuda() for n in bn_keys}
    def put_bn(values):
        for name in bn_keys:
            buffers[name].copy_(values[name])
    def rng_get():
        return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def rng_put(state):
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])
    records=[];batch_records=[];arrays={}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for raw in loader:
            batch_start=time.time()
            voxel_rows=[processor.forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in raw['point_clouds']]
            voxels=processor.collate_batch(voxel_rows)
            bs=len(raw['utterances']);assert bs==8
            assert np.array_equal(voxels['points'][:,1:].reshape(bs,50000,6),raw['point_clouds'].numpy())
            ids=[int(i) for i in raw['local_training_id']]
            hashes=[hashlib.sha256(pc.numpy().tobytes()).hexdigest() for pc in raw['point_clouds']]
            for row_id,digest in zip(ids,hashes):assert digest==old_by_id[row_id]['point_sha256'],row_id
            state=rng_get();outputs={}
            for arm in ['normal','repeat','parent_bn']:
                rng_put(state);put_bn(parent_bn if arm=='parent_bn' else terminal_bn)
                batch={k:v.clone().cuda() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in raw.items()}
                inputs={k:torch.from_numpy(voxels[k].copy()).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
                inputs.update(batch_size=bs,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
                    det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
                    superpoint=batch['superpoint'],train=False)
                prediction=model(inputs)
                if arm=='normal':next_state=rng_get()
                probabilities=prediction['last_sem_cls_scores'].softmax(-1)
                scores=(probabilities*(batch['positive_map'][:,0]>0).float().unsqueeze(1)).sum(-1)
                for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                    scores=scores+(probabilities*batch[key][:,0].unsqueeze(1)).sum(-1)
                scores=scores-(probabilities*batch['other_entity_map'][:,0].unsqueeze(1)).sum(-1)
                boxes=torch.cat([prediction['last_center'],prediction['last_pred_size']],-1)
                gt=torch.cat([batch['center_label'][:,:,:3],batch['size_gts']],-1)[:,0]
                assert bool(batch['box_label_mask'][:,0].all())
                assert bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(scores).all())
                selected=scores.argsort(-1,descending=True)[:,0]
                values=[]
                for b,row_id in enumerate(ids):
                    clamped=boxes[b].clone();clamped[:,3:]=clamped[:,3:].clamp(min=1e-6)
                    iou,_=_iou3d_par(box_cxcyczwhd_to_xyzxyz(clamped),box_cxcyczwhd_to_xyzxyz(gt[b:b+1]))
                    iou=iou[:,0];choice=int(selected[b])
                    values.append(dict(selected=choice,selected_iou=float(iou[choice]),raw_oracle_iou=float(iou.max()),selected_score=float(scores[b,choice])))
                    arrays[str(row_id)+'_'+arm]=torch.cat([boxes[b],scores[b,:,None],iou[:,None]],-1).cpu().numpy()
                outputs[arm]=values
                del prediction,probabilities,inputs,batch,scores,boxes
            put_bn(terminal_bn);rng_put(next_state)
            for b,row_id in enumerate(ids):
                arrays[str(row_id)+'_gt']=gt[b].cpu().numpy()
                records.append(dict(row_id=row_id,scan_id=dataset.annos[row_id]['scan_id'],point_sha256=hashes[b],
                    root_box=gt[b].cpu().tolist(),**{a:outputs[a][b] for a in outputs}))
            rec=dict(batch=len(batch_records)+1,rows=ids,seconds=time.time()-batch_start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            batch_records.append(rec);print('PVG_BN_INTERVENTION_BATCH '+json.dumps(rec),flush=True)
    assert len(records)==128 and len(batch_records)==16
    model.cpu()
    assert all(torch.equal(v,terminal[n]) for n,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    summary={}
    for arm in ['normal','repeat','parent_bn']:
        summary[arm]={str(t):dict(hits=sum(r[arm]['selected_iou']>t for r in records),
            fixes=sum(r['normal']['selected_iou']<=t<r[arm]['selected_iou'] for r in records),
            breaks=sum(r[arm]['selected_iou']<=t<r['normal']['selected_iou'] for r in records)) for t in [.25,.5]}
    diffs={arm:dict(selection_changes=sum(r[arm]['selected']!=r['normal']['selected'] for r in records),
        box_max_abs=max(float(np.max(np.abs(arrays[str(r['row_id'])+'_'+arm][:,:6]-arrays[str(r['row_id'])+'_normal'][:,:6]))) for r in records),
        score_max_abs=max(float(np.max(np.abs(arrays[str(r['row_id'])+'_'+arm][:,6]-arrays[str(r['row_id'])+'_normal'][:,6]))) for r in records)) for arm in ['repeat','parent_bn']}
    write_json(root/'rows.json',records);write_json(root/'batches.json',batch_records)
    np.savez_compressed(str(root/'candidate_values.npz'),**arrays)
    result=dict(status='complete',time_cst=now(),elapsed_seconds=time.time()-begin,training_rows=128,physical_scenes=128,
        model_forwards=48,optimizer_steps=0,formal_rows=0,new_checkpoints=0,backwards=0,
        summary=summary,differences=diffs,bn_buffer_tensors=len(bn_keys),state_unchanged=True,all_model_gradients_none=True,
        terminal_sha256=spec['terminal_sha256'],training_spec_sha256=spec['training_spec_sha256'],
        input_selection_sha256=sha(root/'input_selection.json'),rows_sha256=sha(root/'rows.json'),batches_sha256=sha(root/'batches.json'),
        arrays_sha256=sha(root/'candidate_values.npz'),script_sha256=sha(__file__),
        scope='Fixed train-input eval intervention; no training benefit, formal accuracy, Mask or promotion claim')
    write_json(root/'diagnostic.json',result)
    print('PVG_BN_INTERVENTION_COMPLETE '+json.dumps(result),flush=True)

if __name__=='__main__':
    main()
