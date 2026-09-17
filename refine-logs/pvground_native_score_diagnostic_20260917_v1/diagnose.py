"""Fixed D native loss/score contract on 128 training scenes; no parameter updates."""
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

    _, set_criterion = BaseTrainTester.get_criterion(config)
    assert set_criterion.matcher.cost_class == 1 and set_criterion.matcher.cost_bbox == 0
    assert set_criterion.matcher.cost_giou == 2 and set_criterion.matcher.cost_masks == .0002
    processor = DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),False,6)
    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            assert len(annos) == 36665
            actual = {'fit':[],'holdout':[]}
            for index,row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + chr(0) + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8],16)%5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)
        def __getitem__(self,index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'],[0,1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result
    os.chdir(str(dataset_source))
    verify_scanrefer_superpoints(manifest['data_root'],'train',manifest['superpoint_files']['train'])
    dataset = FitDataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='train',
        data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,
        detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    dataset.augment = False
    assert [r['_local_training_id'] for r in dataset.annos] == list(range(36665))
    chosen = [];seen = set()
    for index in partitions['fit']:
        physical = dataset.annos[index]['scan_id'].split('_')[0]
        if physical not in seen:
            seen.add(physical);chosen.append(index)
        if len(chosen) == 128:
            break
    assert len(chosen) == len(seen) == 128
    assert not set(chosen).intersection(partitions['holdout'])
    assert not seen.intersection({dataset.annos[i]['scan_id'].split('_')[0] for i in partitions['holdout']})
    write_json(root/'input_selection.json',dict(selection='first expression of first 128 different fit physical scenes, before inference',
        rows=[dict(row_id=i,scan_id=dataset.annos[i]['scan_id'],text=dataset.annos[i]['utterance']) for i in chosen],
        split_protocol_sha256=manifest['split_protocol_sha256']))
    reset_rng()
    loader = DataLoader(Subset(dataset,chosen),batch_size=8,shuffle=False,num_workers=2,
        generator=torch.Generator().manual_seed(2027),pin_memory=True,drop_last=False)
    map_keys = ['positive_map','modify_positive_map','pron_positive_map','rel_positive_map','other_entity_map']
    records = [];arrays = {};forward_count = 0;max_gradient_error = 0.
    for raw in loader:
        voxel_rows = [processor.forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in raw['point_clouds']]
        voxels = processor.collate_batch(voxel_rows)
        bs = len(raw['utterances'])
        assert bs == 8
        assert np.array_equal(voxels['points'][:,1:].reshape(bs,50000,6),raw['point_clouds'].numpy())
        point_hashes = [hashlib.sha256(pc.numpy().tobytes()).hexdigest() for pc in raw['point_clouds']]
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in raw.items()}
        inputs = {k:torch.from_numpy(voxels[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs.update(batch_size=bs,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
            det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
            superpoint=batch['superpoint'],train=False)
        with torch.no_grad():
            prediction = model(inputs)
        forward_count += 1
        assert bool(batch['box_label_mask'][:,0].all())
        gt_boxes = torch.cat([batch['center_label'][:,:,:3],batch['size_gts']],-1)
        targets = []
        for b in range(bs):
            valid = batch['box_label_mask'][b].bool()
            target = dict(labels=batch['sem_cls_label'][b,valid],boxes=gt_boxes[b,valid],masks=batch['gt_masks'][b,valid])
            target.update({k:batch[k][b,valid] for k in map_keys})
            targets.append(target)
        logits = prediction['last_sem_cls_scores'].detach().double().requires_grad_()
        # Match using the actual float32 prediction, before any score or size transformation.
        native = dict(pred_logits=prediction['last_sem_cls_scores'],
            pred_boxes=torch.cat([prediction['last_center'],prediction['last_pred_size']],-1),
            pred_masks=prediction['last_pred_masks'],superpoints=prediction['superpoints'],
            language_dataset=batch['language_dataset'])
        indices = set_criterion.matcher(native,targets)
        assert all(len(j)>0 and int((j==0).sum())==1 for i,j in indices)
        native['pred_logits'] = logits
        num_boxes = sum(len(j) for i,j in indices)
        ce = set_criterion.loss_pos_align(native,targets,indices,num_boxes,None)['loss_ce']
        gradient = torch.autograd.grad(ce,logits)[0]
        probabilities = logits.softmax(-1)
        text_weights = (batch['positive_map'][:,0]>0).double()
        for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
            text_weights += batch[key][:,0].double()
        text_weights -= batch['other_entity_map'][:,0].double()
        scores = (probabilities*text_weights.unsqueeze(1)).sum(-1)
        score_gradient = torch.autograd.grad(scores.sum(),logits)[0]
        analytic_score_gradient = probabilities.detach()*(text_weights.unsqueeze(1)-scores.detach().unsqueeze(-1))
        error = float((score_gradient-analytic_score_gradient).abs().max())
        assert error < 1e-12,error
        target_sim = torch.zeros_like(logits);target_sim[:,:,-1] = 1
        # Match the native loss's float32 EOS constant even with double analysis logits.
        eos = torch.full(logits.shape[:2],set_criterion.eos_coef,device=logits.device)
        for b,(query_ids,target_ids) in enumerate(indices):
            t = targets[b]
            weighted = .6*t['positive_map']+.2*t['modify_positive_map']+.2*t['pron_positive_map']+.1*t['rel_positive_map']
            assert batch['language_dataset'][b] == 'scanrefer'
            target_sim[b,query_ids] = weighted[target_ids].double()
            eos[b,query_ids] = 1
        analytic_ce_gradient = (probabilities.detach()*target_sim.sum(-1,keepdim=True)-target_sim)*eos.unsqueeze(-1)/num_boxes
        error = max(error,float((gradient-analytic_ce_gradient).abs().max()))
        assert error < 1e-12,error
        max_gradient_error = max(max_gradient_error,error)
        velocity = -(gradient*score_gradient).sum(-1).detach()
        scores = scores.detach()
        # Confirm double-precision score analysis has not changed the deployed float32 selection.
        deployed_probs = prediction['last_sem_cls_scores'].softmax(-1)
        deployed_scores = (deployed_probs*(batch['positive_map'][:,0]>0).float().unsqueeze(1)).sum(-1)
        for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
            deployed_scores += (deployed_probs*batch[key][:,0].unsqueeze(1)).sum(-1)
        deployed_scores -= (deployed_probs*batch['other_entity_map'][:,0].unsqueeze(1)).sum(-1)
        selected = deployed_scores.argsort(-1,descending=True)[:,0]
        selection_double = scores.argsort(-1,descending=True)[:,0]
        boxes = torch.cat([prediction['last_center'],prediction['last_pred_size'].clamp(min=1e-6)],-1)
        for b,(query_ids,target_ids) in enumerate(indices):
            row_id = int(batch['local_training_id'][b])
            matched = int(query_ids[target_ids==0][0]);choice = int(selected[b])
            ious,_ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(boxes[b]),box_cxcyczwhd_to_xyzxyz(gt_boxes[b,0:1]))
            ious = ious[:,0];best = int(ious.argmax())
            match_ids = torch.full((256,),-1,dtype=torch.int64)
            match_ids[query_ids] = target_ids
            record = dict(row_id=row_id,scan_id=dataset.annos[row_id]['scan_id'],point_sha256=point_hashes[b],
                matched_root=matched,selected=choice,best_iou_query=best,double_selection=int(selection_double[b]),
                selected_iou=float(ious[choice]),matched_iou=float(ious[matched]),best_iou=float(ious[best]),
                selected_score=float(deployed_scores[b,choice]),matched_score=float(deployed_scores[b,matched]),
                best_score=float(deployed_scores[b,best]),best_matched_target=int(match_ids[best]),
                selected_velocity=float(velocity[b,choice]),matched_velocity=float(velocity[b,matched]),
                best_velocity=float(velocity[b,best]),best_margin_velocity=float(velocity[b,best]-velocity[b,choice]),
                matched_margin_velocity=float(velocity[b,matched]-velocity[b,choice]),
                loss_ce=float(ce),num_matched_targets=num_boxes)
            for threshold in [.25,.5]:
                good = ious.cpu()>threshold
                unmatched = match_ids == -1
                label = str(threshold)
                record['good_unmatched_'+label] = int((good&unmatched).sum())
                record['good_unmatched_score_decreasing_'+label] = int((good&unmatched&(velocity[b].cpu()<0)).sum())
            records.append(record)
            arrays[str(row_id)+'_values'] = torch.stack([deployed_scores[b].double(),scores[b],ious.double(),velocity[b]],-1).cpu().numpy()
            arrays[str(row_id)+'_matched_target'] = match_ids.numpy()
        print('NATIVE_SCORE_BATCH '+json.dumps(dict(forwards=forward_count,rows=len(records),gradient_max_error=error)),flush=True)
        del prediction,native,logits,gradient,probabilities,score_gradient,targets,batch,inputs
    assert forward_count == 16 and [r['row_id'] for r in records] == chosen
    model.cpu()
    assert all(torch.equal(value,terminal[name]) for name,value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    np.savez_compressed(str(root/'candidate_values.npz'),**arrays)
    summary = {}
    for threshold in [.25,.5]:
        errors = [r for r in records if r['selected_iou']<=threshold]
        covered_errors = [r for r in errors if r['best_iou']>threshold]
        summary[str(threshold)] = dict(selected_hits=sum(r['selected_iou']>threshold for r in records),
            matched_root_hits=sum(r['matched_iou']>threshold for r in records),raw_oracle_hits=sum(r['best_iou']>threshold for r in records),
            errors_with_good_candidate=len(covered_errors),
            covered_error_best_margin_increasing=sum(r['best_margin_velocity']>0 for r in covered_errors),
            covered_error_matched_margin_increasing=sum(r['matched_margin_velocity']>0 for r in covered_errors),
            covered_error_matched_root_good=sum(r['matched_iou']>threshold for r in covered_errors),
            good_unmatched_candidates=sum(r['good_unmatched_'+str(threshold)] for r in records),
            good_unmatched_score_decreasing=sum(r['good_unmatched_score_decreasing_'+str(threshold)] for r in records))
    write_json(root/'rows.json',records)
    result = dict(status='complete',time_cst=now(),elapsed_seconds=time.time()-begin,training_rows=128,physical_scenes=128,
        model_forwards=forward_count,optimizer_steps=0,formal_rows=0,new_checkpoints=0,summary=summary,
        score_direction='negative native CE gradient at fixed final logits and matching; not shared-parameter or total-loss update',
        double_selection_differences=sum(r['selected']!=r['double_selection'] for r in records),
        gradient_analytic_max_error=max_gradient_error,strict_terminal_restore=True,all_model_gradients_none=True,
        state_unchanged=True,terminal_sha256=spec['terminal_sha256'],training_spec_sha256=spec['training_spec_sha256'],
        loss_source_sha256=sha(source/'models/losses.py'),source_port_sha256=train_spec['source_port_sha256'],
        input_selection_sha256=sha(root/'input_selection.json'),rows_sha256=sha(root/'rows.json'),
        arrays_sha256=sha(root/'candidate_values.npz'),script_sha256=sha(__file__))
    write_json(root/'diagnostic.json',result)
    print('PVG_NATIVE_SCORE_DIAGNOSTIC_COMPLETE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
