"""Exercise the pinned upstream evaluator and full native loss/backward on four fit rows."""
import argparse
from collections import defaultdict
import copy
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    root = args.runtime.resolve()
    model_source = Path('/root/autodl-tmp/mcln_pvground_observation_source_20260909_v1/PV-Ground')
    spec = json.loads((root / 'env_spec.json').read_bytes())
    build = json.loads((root / 'build_receipt.json').read_bytes())
    assert build['status'] == 'pass'
    assert build['spec_sha256'] == hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert json.loads((root / 'agent_kernel_receipt.json').read_bytes())['status'] == 'pass'
    source = json.loads((root / 'source_bundle_receipt.json').read_bytes())
    port = json.loads((model_source.parent/'source_port.json').read_bytes())
    assert sha(model_source.parent/'source_port.json')=='83c712907348c1eeb829bed9db805f32aab6efad87a37f041d5876adcfe84306'
    for name,digest in port['files'].items():assert sha(model_source/name)==digest,name
    fixture_receipt = json.loads((args.fixtures / 'receipt.json').read_bytes())
    assert fixture_receipt['status'] == 'pass' and fixture_receipt['separate_training_labels']
    assert [r['training_row_id'] for r in fixture_receipt['rows']] == [0, 173, 237, 455]
    checkpoint = spec['weight_dirs']['scanrefer']
    assert sha(checkpoint['path']) == checkpoint['sha256']
    args.output.mkdir()
    os.chdir(str(model_source))
    sys.path.insert(0, str(model_source))
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from pcdet.config import cfg, cfg_from_yaml_file
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from train_dist_mod import TrainTester
    from prepare_data import DataProcessor
    from src.grounding_evaluator import GroundingEvaluator

    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(checkpoint['path'], map_location='cpu')
    config = payload['config']
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_color and not config.use_height and not config.use_multiview
    assert config.num_target == 256 and config.num_decoder_layers == 6
    assert config.use_soft_token_loss and config.use_contrastive_align
    assert all(k.startswith('module.') for k in payload['model'])
    state = {k[7:]:v for k,v in payload['model'].items()}
    assert len(state) == 1234
    cfg_from_yaml_file('wandb_config.yaml', cfg)
    model = PVGround(cfg, num_class=256, num_queries=config.num_target,
        num_decoder_layers=config.num_decoder_layers, self_position_embedding=config.self_position_embedding,
        contrastive_align_loss=config.use_contrastive_align, butd=config.butd,
        pointnet_ckpt=None, data_path=fixture_receipt['data_root'], self_attend=config.self_attend)
    embeddings = model.text_encoder.embeddings
    position_ids = torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1))
    assert set(model.state_dict()) - set(state) == {'text_encoder.embeddings.position_ids'}
    assert not set(state) - set(model.state_dict())
    assert torch.equal(embeddings.position_ids, position_ids)
    embeddings.register_buffer('position_ids', embeddings.position_ids, persistent=False)
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    from pvground_observation_query import install_observation_query_read
    install_observation_query_read(model)
    native_state_count=len(state)
    added=set(model.state_dict())-set(state)
    assert added and all(k.startswith('decoder.5.source_query_read.') for k in added)
    state.update({k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in added})
    reader=model.decoder[-1].source_query_read
    model.cuda()
    criterion, set_criterion = BaseTrainTester.get_criterion(config)
    processor_args = (cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE))
    eval_owner = SimpleNamespace(data_processor=DataProcessor(*processor_args, False, 6))
    train_owner = SimpleNamespace(data_processor=DataProcessor(*processor_args, True, 6))

    def batch_at(start, owner):
        fixtures = []
        for row in fixture_receipt['rows'][start:start+2]:
            path = args.fixtures / row['file']
            assert sha(path) == row['file_sha256']
            value = torch.load(str(path), map_location='cpu')
            assert set(value) == {'inputs', 'labels'}
            fixtures.append(value)
        batch = default_collate([v['labels'] for v in fixtures])
        mapped = {'point_clouds':'point_clouds', 'det_boxes':'all_detected_boxes',
            'det_bbox_label_mask':'all_detected_bbox_label_mask', 'det_class_ids':'all_detected_class_ids',
            'superpoint':'superpoint'}
        for old, new in mapped.items():
            batch[new] = torch.cat([v['inputs'][old] for v in fixtures], 0)
        batch['utterances'] = [v['inputs']['text'][0] for v in fixtures]
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        inputs = TrainTester._get_inputs(owner, batch)
        assert set(inputs) == {'voxels','points','voxel_coords','voxel_num_points','batch_size',
            'text','det_boxes','det_bbox_label_mask','det_class_ids','superpoint'}
        assert torch.equal(inputs['points'][:,1:].reshape(2,50000,6), batch['point_clouds'])
        return inputs, batch

    def loss_after_forward(output, batch):
        assert not set(output).intersection(batch)
        output.update(batch)
        loss, output = criterion(output, config.num_decoder_layers, set_criterion,
            query_points_obj_topk=config.query_points_obj_topk)
        assert torch.isfinite(loss), 'native total loss'
        components = {k:float(v) for k,v in output.items() if ('loss' in k) and
            (isinstance(v, (int,float)) or (torch.is_tensor(v) and v.numel()==1))}
        assert all(np.isfinite(v) for v in components.values())
        return loss, output, components

    replay_records=[]
    def rng_state():
        return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def restore_rng(saved):
        random.setstate(saved[0]);np.random.set_state(saved[1])
        torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])
    def same_rng(a,b):
        return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1])
                and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])
                and all(torch.equal(x,y) for x,y in zip(a[3],b[3])))
    def paired_replay(inputs,label):
        rng=rng_state()
        buffers={k:v.detach().clone() for k,v in model.named_buffers()}
        source_outputs=[]
        handle=reader.register_forward_hook(lambda module,args,value:source_outputs.append(float(value.detach().abs().max())))
        with torch.no_grad():
            reader.enabled=False
            native=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
            native_end=rng_state()
            for k,v in model.named_buffers():v.copy_(buffers[k])
            restore_rng(rng);reader.enabled=True
            output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
        handle.remove()
        assert source_outputs==[0.0,0.0]
        assert same_rng(native_end,rng_state())
        rec_keys=['source_features','seed_features','query_points_xyz','query_points_feature',
                  'last_center','last_pred_size','last_sem_cls_scores','last_proj_queries','proj_tokens']
        for key in rec_keys:assert torch.equal(output[key],native[key]),(label,key)
        # The disabled/disabled diagnostic already measured pre-existing superpoint
        # and Mask numeric variation. Record it; do not claim full-Mask bitwise replay.
        states=output['source_observations']
        assert [s.shape[-1] for s in states]==[10,13,13,13,13,13]
        assert all(torch.isfinite(s).all() and not s.requires_grad for s in states)
        state_summary=[dict(shape=list(s.shape),first_radius_empty=int((s[:,:,0]==0).sum()),second_radius_empty=int((s[:,:,6]==0).sum())) for s in states[1:]]
        mask_max={key:max(float((a-b).abs().max()) for a,b in zip(output[key],native[key]))
                  for key in ['sp_last_pred_masks','last_pred_masks','super_xyz_list']}
        replay_records.append(dict(mode=label,rec_keys_exact=rec_keys,rng_end_equal=True,
            source_output_max=source_outputs,mask_max_abs=mask_max,spatial_observation_summary=state_summary,bev_observation_shape=list(states[0].shape)))
        return output

    evaluator = GroundingEvaluator(only_root=True, thresholds=[0.25,0.5], topks=[1,5,10],
        prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround')
    eval_rows = []
    model.eval()
    with torch.no_grad():
        for start in (0,2):
            inputs, batch = batch_at(start, eval_owner)
            inputs['train'] = False
            output = paired_replay(inputs,'eval_'+str(start))
            raw_sizes = output['last_pred_size'].clone()
            _, output, components = loss_after_forward(output, batch)
            # This is the author's _main_eval_branch rule, not a new candidate filter.
            for key in output:
                if 'pred_size' in key:
                    output[key] = torch.clamp(output[key], min=1e-6)
            evaluator.evaluate(output, 'last_')
            semantic = output['last_sem_cls_scores'].softmax(-1)
            contrastive_raw = (torch.matmul(output['last_proj_queries'], output['proj_tokens'].transpose(-1,-2))/0.07).softmax(-1)
            contrastive = torch.zeros_like(semantic)
            contrastive[:,:,:contrastive_raw.shape[-1]] = contrastive_raw
            for bid in range(2):
                row = fixture_receipt['rows'][start+bid]
                details = {'training_row_id':row['training_row_id'], 'scan_id':row['scan_id'],
                    'raw_nonpositive_size_boxes':int((raw_sizes[bid]<=0).any(-1).sum()),
                    'postprocess_positive_size_boxes':int((output['last_pred_size'][bid]>0).all(-1).sum())}
                for mode, probabilities in [('bbs',semantic), ('bbf',contrastive)]:
                    main_map = (batch['positive_map'][bid,0]>0).float()
                    score = (probabilities[bid]*main_map).sum(-1)
                    for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                        score = score + (probabilities[bid]*batch[key][bid,0]).sum(-1)
                    score = score - (probabilities[bid]*batch['other_entity_map'][bid,0]).sum(-1)
                    ranked = score.argsort(descending=True)
                    selected = int(ranked[0])
                    boxes = torch.cat([output['last_center'][bid],output['last_pred_size'][bid]],-1).cpu().numpy()
                    gt = torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]],-1).cpu().numpy()
                    lo = np.maximum(boxes[:,:3]-boxes[:,3:]/2,gt[:3]-gt[3:]/2)
                    hi = np.minimum(boxes[:,:3]+boxes[:,3:]/2,gt[:3]+gt[3:]/2)
                    intersection = np.maximum(hi-lo,0).prod(-1)
                    iou = intersection / (boxes[:,3:].prod(-1)+gt[3:].prod()-intersection)
                    assert np.isfinite(iou).all()
                    logits = output['adaptive_weights'][bid]*output['last_pred_masks'][bid][0,selected]
                    logits = logits + (1-output['adaptive_weights'][bid])*output['sp_last_pred_masks'][bid][selected]
                    mask = (logits.sigmoid()>0.5)[output['superpoints'][bid]]
                    truth = batch['gt_masks'][bid,0].bool()
                    mask_iou = float((mask & truth).sum().float() / (mask | truth).sum())
                    details[mode] = {'query':selected,'score':float(score[selected]),'box':boxes[selected].tolist(),
                        'iou':float(iou[selected]),'mask_iou':mask_iou,
                        'selected_raw_nonpositive_size':bool((raw_sizes[bid,selected]<=0).any()),
                        'top10_ious':iou[ranked[:10].cpu().numpy()].tolist()}
                eval_rows.append(details)
            print('PVG_NATIVE_EVAL_BATCH '+json.dumps({'rows':eval_rows[-2:],'loss':components['loss']}),flush=True)
            del output, inputs, batch
    for mode in ['bbs','bbf']:
        for threshold in [.25,.5]:
            for k in [1,5,10]:
                manual = sum(any(v>threshold for v in row[mode]['top10_ious'][:k]) for row in eval_rows)
                assert evaluator.dets[('last_',threshold,k,mode)] == manual
                assert evaluator.gts[('last_',threshold,k,mode)] == 4
        mask_key = 'mask_pos' if mode=='bbs' else 'mask_sem'
        assert abs(float(evaluator.dets[mask_key])-sum(r[mode]['mask_iou'] for r in eval_rows)) < 1e-5
    assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
    eval_receipt = {'status':'pass','rows':eval_rows,'evaluator_dets':{str(k):float(v) for k,v in evaluator.dets.items()},
        'evaluator_gts':{str(k):float(v) for k,v in evaluator.gts.items()},'formal_rows':0,'checkpoint_state_unchanged':True}
    (args.output/'evaluation_interface.json').write_text(json.dumps(eval_receipt,indent=2)+'\n')

    training_rng=rng_state()
    model.train()
    replay_inputs,replay_batch=batch_at(0,train_owner)
    before_rng=rng_state()
    layer=model.decoder[-1];original_forward=layer.forward
    captured={}
    def capture_forward(*args,**kwargs):
        captured['args']=tuple(v.detach().clone() if torch.is_tensor(v) else v for v in args)
        captured['kwargs']={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in kwargs.items()}
        return original_forward(*args,**kwargs)
    layer.forward=capture_forward
    # First hold the reader disabled twice: quantify upstream training-mode repeat
    # differences before claiming anything about the new last-layer computation.
    upstream=[];upstream_rng=[]
    with torch.no_grad():
        for repeat in range(2):
            model.load_state_dict(state,strict=True);restore_rng(before_rng);reader.enabled=False
            replay_output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in replay_inputs.items()})
            upstream.append(replay_output['source_features'].detach().clone())
            upstream_rng.append(rng_state())
            del replay_output
    layer.forward=original_forward
    assert same_rng(upstream_rng[0],upstream_rng[1])
    upstream_difference=float((upstream[0]-upstream[1]).abs().max())
    del upstream
    # The same real layer inputs remove upstream sparse/BN numerical differences
    # from the causal question of inserting a zero residual before native dropout.
    layer_rng=rng_state();layer_outputs=[];layer_end_rng=[];source_outputs=[]
    handle=reader.register_forward_hook(lambda module,args,value:source_outputs.append(float(value.detach().abs().max())))
    with torch.no_grad():
        for enabled in [False,True]:
            model.load_state_dict(state,strict=True);restore_rng(layer_rng);reader.enabled=enabled
            layer_outputs.append(original_forward(*captured['args'],**captured['kwargs']).detach().clone())
            layer_end_rng.append(rng_state())
    handle.remove()
    assert torch.equal(layer_outputs[0],layer_outputs[1])
    assert same_rng(layer_end_rng[0],layer_end_rng[1])
    assert source_outputs==[0.0,0.0]
    train_layer_replay=dict(same_actual_inputs=True,output_exact=True,rng_end_equal=True,
        disabled_upstream_repeat_max_abs=upstream_difference,full_train_model_bitwise_claim=False,
        full_model_forwards=2,extra_last_layer_forwards=2,source_output_max=source_outputs)
    del captured,layer_outputs,replay_inputs,replay_batch
    model.load_state_dict(state,strict=True);reader.enabled=True
    restore_rng(training_rng)
    training = copy.copy(config)
    training.frozen = False
    training.small_lr = False
    training.lr = training.lr_backbone = 1e-5
    optimizer = BaseTrainTester.get_optimizer(training, model)
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen = {n:p for n,p in model.named_parameters() if not p.requires_grad}
    assert frozen and all(n.startswith('text_encoder.') for n in frozen)
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in trainable.values()}
    train_steps = []
    model.train()
    for step, start in enumerate((0,2),1):
        begin = time.time()
        inputs, batch = batch_at(start, train_owner)
        torch.cuda.reset_peak_memory_stats()
        output = model(inputs)
        loss, output, components = loss_after_forward(output, batch)
        optimizer.zero_grad()
        loss.backward()
        grads = {}
        for name, parameter in trainable.items():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                grads[name] = float(parameter.grad.norm())
        source_grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in reader.named_parameters()}
        assert source_grads['output.weight']>0
        if step==2:
            assert source_grads['attention.in_proj_weight']>0
            assert all(source_grads['projections.%d.weight'%i]>0 for i in range(6))
        if step==2:
            assert all(source_grads['observation_keys.%d'%i]>0 and source_grads['observation_values.%d'%i]>0 for i in range(6))
        groups = defaultdict(lambda:{'tensors':0,'nonzero_tensors':0,'norm_squared_sum':0.0})
        for name, norm in grads.items():
            group = groups[name.split('.')[0]]
            group['tensors'] += 1
            group['nonzero_tensors'] += norm > 0
            group['norm_squared_sum'] += norm**2
        for name in ['backbone_net','cross_encoder','gumbel','decoder','prediction_heads','x_mask','x_query']:
            assert groups[name]['norm_squared_sum']>0, name
        assert all(p.grad is None for p in frozen.values())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.clip_norm)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in trainable.values())
        torch.cuda.synchronize()
        record = {'step':step,'training_row_ids':[r['training_row_id'] for r in fixture_receipt['rows'][start:start+2]],
            'losses':components,'grad_norm_before_clip':float(grad_norm), 'gradient_groups':dict(groups),
            'missing_gradients':[n for n in trainable if n not in grads],
            'source_gradient_norms':source_grads,'all_present_gradients_finite':True,'seconds':time.time()-begin,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        train_steps.append(record)
        print('PVG_FULL_BACKWARD_STEP '+json.dumps(record),flush=True)
        del output, loss, inputs, batch
    changed = [n for n,p in trainable.items() if not torch.equal(p.detach().cpu(),state[n])]
    assert changed
    assert all(torch.equal(p.detach().cpu(),state[n]) for n,p in frozen.items())
    assert torch.equal(model.text_encoder.embeddings.position_ids.cpu(),position_ids)
    changed_buffers = [n for n,p in model.named_buffers() if n in state and not torch.equal(p.detach().cpu(),state[n])]
    assert sha(checkpoint['path']) == checkpoint['sha256']
    receipt = {'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'strict_state_tensors':native_state_count,'added_state_tensors':len(added),'env_spec_sha256':build['spec_sha256'],'checkpoint_sha256':checkpoint['sha256'],
        'fixture_receipt_sha256':sha(args.fixtures/'receipt.json'),'script_sha256':sha(__file__),
        'training_rows':4,'formal_rows':0,'optimizer_steps':2,'new_checkpoint_files':0,
        'eval_forwards':4,'train_forwards':4,'trainable_tensors':len(trainable),
        'trainable_parameters':sum(p.numel() for p in trainable.values()),'frozen_tensors':len(frozen),
        'changed_parameters':changed,'changed_buffers':changed_buffers,'steps':train_steps,
        'optimizer':{'class':type(optimizer).__name__,'lr':training.lr,'backbone_lr':training.lr_backbone,
            'weight_decay':training.weight_decay,'clip_norm':training.clip_norm},
        'frozen_parameters_unchanged':True,'metric_claim':False,'elapsed_seconds':time.time()-started}
    receipt.update(source_query_read=True,observation_state=True,module_sha256=sha(Path(__file__).parent/'pvground_observation_query.py'),model_source=str(model_source),source_port_sha256=sha(model_source.parent/'source_port.json'),zero_residual_eval_rec_replay_exact=True,train_layer_replay=train_layer_replay,replay_records=replay_records,mask_bitwise_equality_claim=False,new_parameters=sum(p.numel() for p in reader.parameters()))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('PVG_SOURCE_QUERY_INTERFACE_PASS '+json.dumps({k:v for k,v in receipt.items() if k not in ['steps','changed_parameters','changed_buffers']}),flush=True)


if __name__ == '__main__':
    main()
