"""Fixed D shared-parameter gradient directions; no optimizer or model update."""
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

    criterion, set_criterion = BaseTrainTester.get_criterion(config)
    assert set_criterion.matcher.cost_class == 1 and set_criterion.matcher.cost_bbox == 0
    assert set_criterion.matcher.cost_giou == 2 and set_criterion.matcher.cost_masks == .0002
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
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    names=[n for n,p in named];parameters=tuple(p for n,p in named)
    assert len(parameters)==820 and sum(p.numel() for p in parameters)==28883227
    group_of=[('.'.join(n.split('.')[:2]) if n.startswith(('decoder.','prediction_heads.')) else n.split('.')[0]) for n in names]
    group_names=sorted(set(group_of))
    write_json(root/'parameter_groups.json',dict(trainable_tensors=len(parameters),trainable_parameters=sum(p.numel() for p in parameters),
        groups={g:[n for n,k in zip(names,group_of) if k==g] for g in group_names}))

    def dot_groups(left,right):
        sums={g:torch.zeros((),device='cuda',dtype=torch.float64) for g in group_names}
        for a,b,g in zip(left,right,group_of):
            if a is not None and b is not None:
                sums[g]+=torch.sum(a.detach().double()*b.detach().double())
        return {g:float(v) for g,v in sums.items()}

    def finite(grads):
        for n,g in zip(names,grads):
            if g is not None:assert bool(torch.isfinite(g).all()),n

    records=[];batch_records=[];arrays={};margin_backwards=0
    torch.cuda.reset_peak_memory_stats()
    for raw in loader:
        batch_start=time.time()
        voxel_rows=[processor.forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in raw['point_clouds']]
        voxels=processor.collate_batch(voxel_rows)
        bs=len(raw['utterances']);assert bs==8
        assert np.array_equal(voxels['points'][:,1:].reshape(bs,50000,6),raw['point_clouds'].numpy())
        ids=[int(i) for i in raw['local_training_id']]
        hashes=[hashlib.sha256(pc.numpy().tobytes()).hexdigest() for pc in raw['point_clouds']]
        for row_id,digest in zip(ids,hashes):assert digest==old_by_id[row_id]['point_sha256'],row_id
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in raw.items()}
        inputs={k:torch.from_numpy(voxels[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs.update(batch_size=bs,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
            det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
            superpoint=batch['superpoint'],train=False)
        prediction=model(inputs)
        logits=prediction['last_sem_cls_scores']
        assert logits.requires_grad
        probabilities=logits.softmax(-1)
        scores=(probabilities*(batch['positive_map'][:,0]>0).float().unsqueeze(1)).sum(-1)
        for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
            scores=scores+(probabilities*batch[key][:,0].unsqueeze(1)).sum(-1)
        scores=scores-(probabilities*batch['other_entity_map'][:,0].unsqueeze(1)).sum(-1)
        selected=scores.detach().argsort(-1,descending=True)[:,0]
        assert not set(prediction).intersection(batch)
        prediction.update(batch)
        matching=[]
        def capture_match(module,inputs,output):
            matching.append([(q.clone(),t.clone()) for q,t in output])
        hook=set_criterion.matcher.register_forward_hook(capture_match)
        total_loss,prediction=criterion(prediction,6,set_criterion,query_points_obj_topk=config.query_points_obj_topk)
        hook.remove()
        assert len(matching)==7 and bool(torch.isfinite(total_loss))
        indices=matching[1]
        last_ce=prediction['last__loss_ce']*(.5/7.)
        full_grad=torch.autograd.grad(total_loss,parameters,retain_graph=True,allow_unused=True)
        ce_all=torch.autograd.grad(last_ce,parameters+(logits,),retain_graph=True,allow_unused=True)
        ce_grad=ce_all[:-1];ce_logits=ce_all[-1]
        finite(full_grad);finite(ce_grad)
        assert ce_logits is not None and bool(torch.isfinite(ce_logits).all())
        full_norm=dot_groups(full_grad,full_grad);ce_norm=dot_groups(ce_grad,ce_grad)
        cross=dot_groups(full_grad,ce_grad)
        assert all(p.grad is None for p in model.parameters())
        gt_boxes=torch.cat([batch['center_label'][:,:,:3],batch['size_gts']],-1)
        boxes=torch.cat([prediction['last_center'],prediction['last_pred_size'].clamp(min=1e-6)],-1).detach()
        for b,(query_ids,target_ids) in enumerate(indices):
            assert int((target_ids==0).sum())==1 and bool(batch['box_label_mask'][b,0])
            matched=int(query_ids[target_ids==0][0]);choice=int(selected[b]);row_id=ids[b]
            iou,_=_iou3d_par(box_cxcyczwhd_to_xyzxyz(boxes[b]),box_cxcyczwhd_to_xyzxyz(gt_boxes[b,0:1]))
            iou=iou[:,0].detach();best=int(iou.argmax())
            margin=scores[b,matched]-scores[b,choice]
            if matched==choice:
                full_dot={g:0. for g in group_names};ce_dot=dict(full_dot);logit_dot=0.
            else:
                m_all=torch.autograd.grad(margin,parameters+(logits,),retain_graph=True,allow_unused=True)
                m_grad=m_all[:-1];m_logits=m_all[-1]
                finite(m_grad);assert m_logits is not None and bool(torch.isfinite(m_logits).all())
                full_dot=dot_groups(m_grad,full_grad);ce_dot=dot_groups(m_grad,ce_grad)
                logit_dot=float(torch.sum(m_logits.detach().double()*ce_logits.detach().double()))
                margin_backwards+=1
                del m_all,m_grad,m_logits
            old=old_by_id[row_id]
            record=dict(row_id=row_id,scan_id=dataset.annos[row_id]['scan_id'],point_sha256=hashes[b],
                selected=choice,matched_root=matched,best_iou_query=best,selected_iou=float(iou[choice]),
                matched_iou=float(iou[matched]),best_iou=float(iou[best]),selected_score=float(scores[b,choice]),
                matched_score=float(scores[b,matched]),margin=float(margin),root_box=gt_boxes[b,0].cpu().tolist(),
                parameter_full_velocity=-sum(full_dot.values()),parameter_last_ce_velocity=-sum(ce_dot.values()),
                parameter_remainder_velocity=-sum(full_dot.values())+sum(ce_dot.values()),
                logit_last_ce_velocity=-logit_dot,
                full_velocity_by_group={g:-v for g,v in full_dot.items()},last_ce_velocity_by_group={g:-v for g,v in ce_dot.items()},
                reference_selected=old['selected'],reference_matched_root=old['matched_root'],
                reference_selected_iou=old['selected_iou'],reference_matched_iou=old['matched_iou'],
                selected_score_reference_diff=float(scores[b,choice])-old['selected_score'])
            records.append(record)
            arrays[str(row_id)+'_values']=torch.stack([scores[b].detach(),iou],-1).cpu().numpy()
        batch_record=dict(batch=len(batch_records)+1,rows=ids,seconds=time.time()-batch_start,
            loss_total=float(total_loss),loss_last_ce_weighted=float(last_ce),
            loss_components={k:float(prediction[k]) for k in ['loss_ce','loss_bbox','loss_giou','loss_sem_align','loss_mask','loss_dice',
                'sp_loss_mask','sp_loss_dice','corresponding_loss_mask','corresponding_loss_dice','adaptive_weight_loss_mask','adaptive_weight_loss_dice']},
            full_grad_norm2_by_group=full_norm,last_ce_grad_norm2_by_group=ce_norm,full_last_ce_dot_by_group=cross,
            full_grad_nonempty=sum(g is not None for g in full_grad),last_ce_grad_nonempty=sum(g is not None for g in ce_grad),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        batch_records.append(batch_record)
        print('PVG_PARAMETER_DIRECTION_BATCH '+json.dumps(batch_record),flush=True)
        del prediction,logits,probabilities,scores,total_loss,last_ce,full_grad,ce_all,ce_grad,ce_logits,batch,inputs,margin
    assert [r['row_id'] for r in records]==chosen and len(batch_records)==16
    model.cpu()
    assert all(torch.equal(v,terminal[n]) for n,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    summary={}
    for threshold in [.25,.5]:
        errors=[r for r in records if r['selected_iou']<=threshold<r['matched_iou']]
        summary[str(threshold)]=dict(selected_hits=sum(r['selected_iou']>threshold for r in records),
            root_matched_hits=sum(r['matched_iou']>threshold for r in records),root_covered_errors=len(errors),
            logit_ce_positive=sum(r['logit_last_ce_velocity']>0 for r in errors),
            parameter_ce_positive=sum(r['parameter_last_ce_velocity']>0 for r in errors),
            parameter_full_positive=sum(r['parameter_full_velocity']>0 for r in errors),
            parameter_full_negative=sum(r['parameter_full_velocity']<0 for r in errors),
            ce_positive_full_negative=sum(r['parameter_last_ce_velocity']>0 and r['parameter_full_velocity']<0 for r in errors),
            full_negative_row_ids=[r['row_id'] for r in errors if r['parameter_full_velocity']<0])
    write_json(root/'rows.json',records);write_json(root/'batches.json',batch_records)
    np.savez_compressed(str(root/'candidate_values.npz'),**arrays)
    result=dict(status='complete',time_cst=now(),elapsed_seconds=time.time()-begin,training_rows=128,physical_scenes=128,
        model_forwards=16,loss_parameter_backwards=32,margin_backwards=margin_backwards,optimizer_steps=0,formal_rows=0,new_checkpoints=0,
        summary=summary,strict_terminal_restore=True,state_unchanged=True,all_model_gradients_none=True,
        selection_differences=sum(r['selected']!=r['reference_selected'] for r in records),
        root_match_differences=sum(r['matched_root']!=r['reference_matched_root'] for r in records),
        selected_iou_reference_max_diff=max(abs(r['selected_iou']-r['reference_selected_iou']) for r in records),
        selected_score_reference_max_diff=max(abs(r['selected_score_reference_diff']) for r in records),
        terminal_sha256=spec['terminal_sha256'],training_spec_sha256=spec['training_spec_sha256'],
        loss_source_sha256=sha(source/'models/losses.py'),source_port_sha256=train_spec['source_port_sha256'],
        input_selection_sha256=sha(root/'input_selection.json'),rows_sha256=sha(root/'rows.json'),batches_sha256=sha(root/'batches.json'),
        arrays_sha256=sha(root/'candidate_values.npz'),script_sha256=sha(__file__),
        scope='eval-mode fixed-input first-order parameter direction; no optimizer, train-mode or finite-step performance claim')
    write_json(root/'diagnostic.json',result)
    print('PVG_PARAMETER_DIRECTION_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
