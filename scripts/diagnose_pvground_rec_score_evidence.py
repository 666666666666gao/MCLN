"""Fixed initial/F token score and target matching diagnostic; no optimizer or inference change."""
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
    assert sha(root/'pvground_rec_competition.py') == train_spec['competition_module_sha256']
    from pvground_rec_competition import bbs_scores, competition_loss
    reference_result = json.loads((root/'reference_diagnostic.json').read_bytes())
    reference_rows = json.loads((root/'reference_rows.json').read_bytes())
    reference_selection = json.loads((root/'reference_selection.json').read_bytes())
    assert reference_result['status']=='complete'
    assert reference_result['input_selection_sha256']==sha(root/'reference_selection.json')
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
    def rng_get():
        return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def rng_put(state):
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])
    records=[];batch_records=[];arrays={}
    term_names=['root','modifier','pronoun','relation','other_entity']
    map_names=['positive_map','modify_positive_map','pron_positive_map','rel_positive_map','other_entity_map']
    states={'initial':initial,'terminal':terminal}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for raw in loader:
            start=time.time();bs=len(raw['utterances']);assert bs==8
            voxel_rows=[processor.forward({'points':pc.numpy().copy(),'use_lead_xyz':True}) for pc in raw['point_clouds']]
            voxels=processor.collate_batch(voxel_rows)
            assert np.array_equal(voxels['points'][:,1:].reshape(bs,50000,6),raw['point_clouds'].numpy())
            ids=[int(i) for i in raw['local_training_id']]
            hashes=[hashlib.sha256(pc.numpy().tobytes()).hexdigest() for pc in raw['point_clouds']]
            for row_id,digest in zip(ids,hashes):assert digest==old_by_id[row_id]['point_sha256']
            state=rng_get();arm_rows={};arm_losses={}
            for arm in ['initial','terminal']:
                model.load_state_dict(states[arm],strict=True);model.eval();rng_put(state)
                batch={k:v.clone().cuda() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in raw.items()}
                inputs={k:torch.from_numpy(voxels[k].copy()).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
                inputs.update(batch_size=bs,text=batch['utterances'],det_boxes=batch['all_detected_boxes'],
                    det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'],
                    superpoint=batch['superpoint'],train=False)
                prediction=model(inputs)
                if arm=='initial':next_state=rng_get()
                probabilities=prediction['last_sem_cls_scores'].softmax(-1)
                terms=[]
                for index,key in enumerate(map_names):
                    weights=(batch[key][:,0]>0).float() if index==0 else batch[key][:,0]
                    terms.append((probabilities*weights[:,None]).sum(-1))
                parts=torch.stack(terms,-1)
                scores=parts[...,:4].sum(-1)-parts[...,4]
                native_scores=bbs_scores(prediction['last_sem_cls_scores'],batch)
                assert torch.allclose(scores,native_scores,atol=2e-7,rtol=0)
                # Preserve the original operation order for actual ranking.
                scores=native_scores
                selected=scores.argsort(-1,descending=True)[:,0]
                matches=[]
                def capture_match(module,inputs,output):
                    matches.append([(q.clone(),t.clone()) for q,t in output])
                hook=set_criterion.matcher.register_forward_hook(capture_match)
                assert not set(prediction).intersection(batch)
                prediction.update(batch)
                native_loss,prediction=criterion(prediction,6,set_criterion,query_points_obj_topk=config.query_points_obj_topk)
                hook.remove();assert len(matches)==7 and bool(torch.isfinite(native_loss))
                extra,counts=competition_loss(prediction,batch,matches[1])
                arm_losses[arm]=dict(native=float(native_loss),competition=float(extra),**counts)
                boxes=torch.cat([prediction['last_center'],prediction['last_pred_size']],-1)
                gt=torch.cat([batch['center_label'][:,:,:3],batch['size_gts']],-1)[:,0]
                assert bool(batch['box_label_mask'][:,0].all())
                values=[]
                for b,row_id in enumerate(ids):
                    q,t=matches[1][b];assert int((t==0).sum())==1
                    matched=int(q[t==0][0]);choice=int(selected[b])
                    clamped=boxes[b].clone();clamped[:,3:]=clamped[:,3:].clamp(min=1e-6)
                    iou,_=_iou3d_par(box_cxcyczwhd_to_xyzxyz(clamped),box_cxcyczwhd_to_xyzxyz(gt[b:b+1]))
                    iou=iou[:,0];assert bool(torch.isfinite(iou).all())
                    roles={'selected':choice,'matched_root':matched,'best_iou':int(iou.argmax())}
                    comparisons={}
                    for threshold in [.25,.5]:
                        good=iou>threshold;eligible=bool(iou[matched]>threshold)
                        entry=dict(eligible=eligible,has_good=bool(good.any()))
                        if bool(good.any()):
                            good_ids=torch.nonzero(good,as_tuple=False).flatten()
                            best=int(good_ids[scores[b,good_ids].argmax()]);roles['best_good_'+str(threshold)]=best
                            entry.update(rank_min=1+int((scores[b]>scores[b,best]).sum()),
                                score_gap_to_selected=float(scores[b,best]-scores[b,choice]),
                                matched_is_best_good=matched==best,
                                matched_to_best_good_iou_gap=float(iou[matched]-iou[best]))
                        negatives=torch.nonzero(~good,as_tuple=False).flatten()
                        if eligible and len(negatives):
                            negative=int(negatives[scores[b,negatives].argmax()])
                            entry.update(negative_query=negative,negative_iou=float(iou[negative]),
                                actual_margin_violation=float((iou[matched]-iou[negative]+scores[b,negative]-scores[b,matched]).relu()))
                        comparisons[str(threshold)]=entry
                    token_ids=prediction['tokenized']['input_ids'][b].cpu().tolist()
                    tokens=model.tokenizer.convert_ids_to_tokens(token_ids)
                    role_values={}
                    for role,index in roles.items():
                        weights=probabilities[b,index];largest=weights.argsort(descending=True)[:8].cpu().tolist()
                        role_values[role]=dict(query=index,iou=float(iou[index]),score=float(scores[b,index]),
                            components={name:float(parts[b,index,k]) for k,name in enumerate(term_names)},
                            no_object_probability=float(weights[-1]),top_probabilities=[dict(position=k,
                                token='<no-object>' if k==255 else tokens[k] if k<len(tokens) else '<unused-position>',
                                probability=float(weights[k])) for k in largest])
                    values.append(dict(roles=role_values,comparisons=comparisons))
                    arrays[str(row_id)+'_'+arm]=torch.cat([boxes[b],scores[b,:,None],iou[:,None],parts[b],probabilities[b,:,-1:]],-1).cpu().numpy()
                    arrays[str(row_id)+'_'+arm+'_role_probs']=torch.stack([probabilities[b,index] for index in roles.values()]).cpu().numpy()
                arm_rows[arm]=values
                del prediction,probabilities,inputs,batch,scores,boxes,parts,native_loss,extra
            rng_put(next_state)
            for b,row_id in enumerate(ids):
                records.append(dict(row_id=row_id,scan_id=dataset.annos[row_id]['scan_id'],text=raw['utterances'][b],
                    point_sha256=hashes[b],root_box=gt[b].cpu().tolist(),**{a:arm_rows[a][b] for a in arm_rows}))
            info=dict(batch=len(batch_records)+1,rows=ids,seconds=time.time()-start,losses=arm_losses,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            batch_records.append(info);print('PVG_SCORE_EVIDENCE_BATCH '+json.dumps(info),flush=True)
    assert [r['row_id'] for r in records]==chosen and len(batch_records)==16
    model.cpu()
    assert all(torch.equal(v,terminal[n]) for n,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    summary={}
    for arm in ['initial','terminal']:
        summary[arm]={}
        for threshold in [.25,.5]:
            key=str(threshold)
            failed=[r[arm] for r in records if r[arm]['roles']['selected']['iou']<=threshold]
            covered=[r for r in failed if r['comparisons'][key]['has_good']]
            summary[arm][key]=dict(selected_hits=128-len(failed),
                root_matched_hits=sum(r[arm]['roles']['matched_root']['iou']>threshold for r in records),
                raw256_hits=sum(r[arm]['roles']['best_iou']['iou']>threshold for r in records),
                covered_errors=len(covered),covered_errors_matched_eligible=sum(r['comparisons'][key]['eligible'] for r in covered),
                covered_errors_not_matched_eligible=sum(not r['comparisons'][key]['eligible'] for r in covered))
    write_json(root/'rows.json',records);write_json(root/'batches.json',batch_records)
    np.savez_compressed(str(root/'candidate_values.npz'),**arrays)
    result=dict(status='complete',time_cst=now(),elapsed_seconds=time.time()-begin,training_rows=128,physical_scenes=128,
        model_forwards=32,optimizer_steps=0,formal_rows=0,new_checkpoints=0,backwards=0,
        summary=summary,strict_terminal_restore=True,state_unchanged=True,all_model_gradients_none=True,
        terminal_sha256=spec['terminal_sha256'],training_spec_sha256=spec['training_spec_sha256'],
        input_selection_sha256=sha(root/'input_selection.json'),rows_sha256=sha(root/'rows.json'),
        batches_sha256=sha(root/'batches.json'),arrays_sha256=sha(root/'candidate_values.npz'),script_sha256=sha(__file__),
        array_columns=['cx','cy','cz','sx','sy','sz','bbs','iou']+term_names+['no_object_probability'],
        scope='Preselected128 fit physical scenes in eval mode; initial/F same input and RNG. '
              'Token evidence and matching analysis, not new accuracy or causal branch attribution. '
              'No hypothesis is tested by changing deployment scoring, and no permanent query correspondence is assumed.')
    write_json(root/'diagnostic.json',result)
    print('PVG_SCORE_EVIDENCE_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
