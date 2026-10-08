"""Actual B8 support-identity sanity check; not a formal accuracy evaluation."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = Path(spec['root'])
    assert not (root/'preflight.json').exists()
    assert spec['seed'] == 2027 and spec['batch_size'] == 8
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['starting_hits'] == [5599,4859] and spec['target_hits'] == [5658,4850]
    for name,digest in spec['new_files'].items():assert sha(root/name) == digest,name
    parent_spec = json.loads(Path(spec['parent_spec']).read_bytes())
    assert sha(spec['parent_spec']) == spec['parent_spec_sha256']
    env = json.loads((Path(parent_spec['runtime'])/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest() == parent_spec['env_spec_sha256']
    manifest = json.loads(Path(parent_spec['input_manifest']).read_bytes())
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    source = Path(manifest['model_source']); model_source = Path(parent_spec['model_source'])
    port = json.loads(Path(parent_spec['source_port']).read_bytes())
    assert sha(parent_spec['source_port']) == parent_spec['source_port_sha256']
    for name,digest in port['files'].items():assert sha(model_source/name) == digest,name
    for name,digest in parent_spec['runner_files'].items():assert sha(Path(parent_spec['helper_root'])/name) == digest,name
    for name in ('mask_support_model_factory.py','selected_mask_reference_factory.py','mask_support_corrector.py','mask_reference.py'):
        assert sha(Path(parent_spec['root'])/name) == parent_spec['new_runner_files'][name],name
    assert sha(spec['support_terminal']) == spec['support_terminal_sha256']
    official = env['weight_dirs']['scanrefer']['path']
    assert sha(official) == parent_spec['checkpoint_sha256']
    assert sha(parent_spec['base_terminal']) == parent_spec['base_terminal_sha256']
    assert sha(parent_spec['selected_terminal']) == parent_spec['selected_terminal_sha256']

    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Subset

    def reset_rng():
        random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    os.chdir(str(source));sys.path.insert(0,str(source))
    from src.joint_det_dataset import Joint3DDataset
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == source/'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    for path in (parent_spec['helper_root'],parent_spec['root'],str(root),str(model_source)):
        sys.path.insert(0,path)
    from pcdet.config import cfg,cfg_from_yaml_file
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from mask_support_model_factory import build_support_model
    from support_identity_readout import SupportIdentityReadout,install_support_identity_readout
    from readback_preflight_checks import observed_readback_forward,native_bbs_witness
    from pvground_semantic_assignment import semantic_assignment_correction
    from whole_model_preflight_checks import optimizer_restore_exact
    cfg_from_yaml_file(str(Path(parent_spec['runtime'])/'PV-Ground/wandb_config.yaml'),cfg)
    official_payload = torch.load(official,map_location='cpu')
    g_payload = torch.load(parent_spec['base_terminal'],map_location='cpu')
    reference_payload = torch.load(parent_spec['selected_terminal'],map_location='cpu')
    support_payload = torch.load(spec['support_terminal'],map_location='cpu')
    assert support_payload['arm'] == 'content' and support_payload['total_support_updates'] == 7446
    model,config,load = build_support_model(cfg,official_payload,g_payload,reference_payload,manifest['data_root'],support_payload)
    assert load['full_state_tensors'] == 1314
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.cuda().eval()
    frozen = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    heads = {'shared_text':SupportIdentityReadout(False).cuda()}
    heads['candidate_fused'] = copy.deepcopy(heads['shared_text'])
    heads['candidate_fused'].candidate_specific = True
    optimizers = {name:torch.optim.AdamW(head.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay']) for name,head in heads.items()}
    assert all(torch.equal(v,heads['candidate_fused'].state_dict()[k]) for k,v in heads['shared_text'].state_dict().items())
    training = copy.copy(config);training.frozen=False;training.small_lr=False
    criterion,set_criterion = BaseTrainTester.get_criterion(training)
    processor = DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),True,6)

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self,annos):
            assert len(annos) == 36665
            actual = {'fit':[],'holdout':[]}
            for index,row in enumerate(annos):
                row['_identity_training_id'] = index
                key = (manifest['split_salt']+'\0'+row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(key).hexdigest()[:8],16)%5
                actual['holdout' if fold==0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)

        def __getitem__(self,index):
            sample = super().__getitem__(index)
            sample['local_training_id'] = self.annos[index]['_identity_training_id']
            assert np.isin(sample['gt_masks'],[0,1]).all()
            sample['gt_masks'] = sample['gt_masks'].astype(np.bool_)
            return sample

    os.chdir(str(source));started=time.monotonic()
    dataset = FitDataset(dataset_dict={'scanrefer':1},test_dataset='scanrefer',split='train',data_path=manifest['data_root'],use_color=True,use_height=False,use_multiview=False,detect_intermediate=True,butd=True,butd_cls=False,butd_gt=False,augment_det=False,skip_missing_superpoints=True)
    assert len(dataset) == 36665
    dataset.augment=True;dataset.augment_det=False
    reset_rng()
    loader = DataLoader(Subset(dataset,partitions['fit'][8:16]),batch_size=8,shuffle=False,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(2027))
    raw_batch = next(iter(loader))
    os.chdir(str(model_source))

    def prepare():
        voxel = processor.collate_batch([processor.forward(dict(points=p.numpy().copy(),use_lead_xyz=True)) for p in raw_batch['point_clouds']])
        assert np.array_equal(voxel['points'][:,1:].reshape(8,50000,6),raw_batch['point_clouds'].numpy())
        batch = {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in raw_batch.items()}
        inputs = {k:torch.from_numpy(voxel[k]).float().cuda() for k in ('points','voxels','voxel_coords','voxel_num_points')}
        inputs.update(batch_size=8,text=batch['utterances'],superpoint=batch['superpoint'],train=False,det_boxes=batch['all_detected_boxes'],det_bbox_label_mask=batch['all_detected_bbox_label_mask'],det_class_ids=batch['all_detected_class_ids'])
        return inputs,batch

    witnesses = []
    for step in range(2):
        inputs,batch = prepare()
        with torch.no_grad():parent,parent_call = observed_readback_forward(model,inputs)
        q = parent['last_semantic_query_before_readback'].detach()
        raw_points = inputs['points'][:,1:].reshape(8,50000,6)
        assert torch.equal(q,parent['last_semantic_query_after_readback'])
        for arm,head in heads.items():
            other = 'candidate_fused' if arm=='shared_text' else 'shared_text'
            before_other = {k:v.detach().clone() for k,v in heads[other].state_dict().items()}
            predictions = dict(parent)
            after = head(q,None,None,raw_points,predictions)
            logits = model.prediction_heads[-1].sem_cls_scores_head(after.transpose(1,2).contiguous()).transpose(2,1)
            if step==0:
                assert torch.equal(q,after) and torch.equal(logits,parent['last_sem_cls_scores'])
            predictions['last_sem_cls_scores'] = logits
            assert not set(predictions).intersection(batch)
            predictions.update(batch)
            matching = []
            hook = set_criterion.matcher.register_forward_hook(lambda module,arguments,result:matching.append([(a.clone(),b.clone()) for a,b in result]))
            native,predictions = criterion(predictions,6,set_criterion,query_points_obj_topk=training.query_points_obj_topk)
            hook.remove();assert len(matching)==7
            correction,assignment = semantic_assignment_correction(predictions,batch,matching[1],set_criterion.eos_coef)
            loss = native+correction
            score_check = native_bbs_witness(logits,batch)
            isolated = predictions['last__loss_ce']*(.5/7)+correction
            parameters = tuple(head.parameters())
            full_gradients = torch.autograd.grad(loss,parameters,retain_graph=True)
            semantic_gradients = torch.autograd.grad(isolated,parameters,retain_graph=True)
            assert all(torch.allclose(a,b,rtol=1e-5,atol=1e-7) for a,b in zip(full_gradients,semantic_gradients))
            optimizers[arm].zero_grad(set_to_none=True)
            loss.backward()
            norms = {name:float(p.grad.norm()) for name,p in head.named_parameters()}
            assert norms['output.weight']>0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            if step:
                assert sum(v for name,v in norms.items() if not name.startswith('output.'))>0
            assert all(p.grad is None for p in model.parameters())
            assert all(p.grad is None for p in heads[other].parameters())
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(),spec['clip_norm'])
            assert torch.isfinite(gradient_norm)
            optimizers[arm].step();optimizers[arm].zero_grad(set_to_none=True)
            assert all(torch.equal(v,heads[other].state_dict()[k]) for k,v in before_other.items())
            witnesses.append(dict(step=step+1,arm=arm,loss=float(loss),assignment=assignment,gradients=norms,gradient_norm=float(gradient_norm),native_score=score_check,parent_call=parent_call,zero_output_exact=step==0,gt_count=int(batch['box_label_mask'].sum()),support_mass_zero_queries=int((predictions['identity_support_mass']==0).sum()),only_native_last_ce_and_g_gradient_route=True))
    assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in frozen.items())
    restores = {}
    for arm,head in heads.items():
        memory = io.BytesIO();torch.save(dict(head=head.state_dict(),optimizer=optimizers[arm].state_dict()),memory)
        length=memory.tell();memory.seek(0);saved=torch.load(memory,map_location='cpu')
        restored,cfg_restored,receipt = build_support_model(cfg,official_payload,g_payload,reference_payload,manifest['data_root'],support_payload)
        installed = install_support_identity_readout(restored,arm=='candidate_fused',saved['head'])
        expected = {k:v for k,v in frozen.items() if not k.startswith('boundary_evidence_readback.')}
        expected.update({'boundary_evidence_readback.'+k:v for k,v in saved['head'].items()})
        assert set(expected)==set(restored.state_dict())
        assert all(torch.equal(v,restored.state_dict()[k].detach().cpu()) for k,v in expected.items())
        restored.cuda().eval()
        optimizer = torch.optim.AdamW(installed.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
        optimizer.load_state_dict(saved['optimizer'])
        optimizer_check = optimizer_restore_exact(optimizer,saved['optimizer'])
        assert all(int(v['step'])==2 for v in optimizer.state.values())
        inputs,_ = prepare()
        with torch.no_grad():integrated,call = observed_readback_forward(restored,inputs)
        with torch.no_grad():
            replay = installed(integrated['last_semantic_query_before_readback'],None,None,None,dict(integrated))
        assert torch.equal(replay,integrated['last_semantic_query_after_readback'])
        restores[arm] = dict(strict_cpu_full_states=len(expected),head_state_tensors=8,optimizer_states=8,optimizer_check=optimizer_check,actual_integrated_native_head_calls=call['final_semantic_head_calls'],same_frame_masks_geometry_exact=call['fixed_geometry'],same_cached_query_replay_exact=True,serialized_bytes=length,serialization_storage='in_memory_BytesIO',independent_full_forward_identity_claim=False)
        del restored,optimizer,installed
    report = dict(status='PASS_ACTUAL_B8_TWO_STEP_SUPPORT_IDENTITY_PREFLIGHT',seed=2027,training_rows=raw_batch['local_training_id'].tolist(),batch_size=8,optimizer_steps_per_arm=2,head_parameters=107040,head_state_tensors=8,parent_state_tensors=1314,parent_state_unchanged=True,arms=list(heads),restores=restores,witnesses=witnesses,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-started,formal_accuracy_result=False,weight_files_created=0,no_multiseed=True)
    with (root/'preflight.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
