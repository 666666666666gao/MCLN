"""Trace two seeded replays and one advancing-RNG control of a frozen PV-Ground batch."""
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
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run=json.loads(args.spec.read_bytes())
    runtime=Path(run['runtime'])
    env=json.loads((runtime/'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==run['env_spec_sha256']
    bundle=json.loads((runtime/'source_bundle_receipt.json').read_bytes())
    model_source=Path(run['model_source'])
    port=json.loads(Path(run['source_port']).read_bytes())
    for name,info in bundle['sources']['PV-Ground']['files'].items():
        expected=port['after_sha256'] if 'PV-Ground/'+name==port['file'] else info['sha256']
        assert sha(model_source/name)==expected,name
    checkpoint=env['weight_dirs']['scanrefer']
    assert sha(checkpoint['path'])==run['checkpoint_sha256']==checkpoint['sha256']
    fixtures=args.spec.parent/'inputs'
    receipt=json.loads((fixtures/'receipt.json').read_bytes())
    assert receipt['status']=='pass' and receipt['batch_size']==8
    assert sha(fixtures/'first_batch.pt')==receipt['input_file_sha256']
    os.chdir(str(model_source));sys.path.insert(0,str(model_source))
    import numpy as np
    import torch
    from pcdet.config import cfg,cfg_from_yaml_file
    from models.pv_ground import PVGround
    from prepare_data import DataProcessor
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

    def seed():
        random.seed(run['seed']);np.random.seed(run['seed'])
        torch.manual_seed(run['seed']);torch.cuda.manual_seed_all(run['seed'])

    seed()
    payload=torch.load(checkpoint['path'],map_location='cpu')
    config=payload['config']
    state={k[7:]:v for k,v in payload['model'].items()}
    assert len(state)==1234 and all(k.startswith('module.') for k in payload['model'])
    cfg_from_yaml_file('wandb_config.yaml',cfg)
    model=PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
        pointnet_ckpt=None,data_path=receipt['data_root'],self_attend=config.self_attend)
    assert set(model.state_dict())-set(state)=={'text_encoder.embeddings.position_ids'}
    expected=torch.arange(model.text_encoder.config.max_position_embeddings).expand((1,-1))
    assert torch.equal(model.text_encoder.embeddings.position_ids,expected)
    model.text_encoder.embeddings.register_buffer('position_ids',model.text_encoder.embeddings.position_ids,persistent=False)
    model.load_state_dict(state,strict=True);model.cuda().eval().requires_grad_(False)
    frozen=torch.load(str(fixtures/'first_batch.pt'),map_location='cpu')
    batch=frozen['inputs'];targets=frozen['targets']
    processor=DataProcessor(cfg.DATA_PROCESSOR,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),False,6)
    voxel_rows=[processor.forward(dict(points=p.numpy().copy(),use_lead_xyz=True)) for p in batch['point_clouds']]
    voxels=processor.collate_batch(voxel_rows)
    assert np.array_equal(voxels['points'][:,1:].reshape(8,50000,6),batch['point_clouds'].numpy())
    trace={}

    def record(name,value):
        trace[name]=value.detach().cpu().clone()

    def backbone_hook(module,inputs,output):
        for key in ['inds','point_coords','point_features']:record('backbone.'+key,output[key])

    def cross_hook(module,inputs,output):
        record('cross_encoder.visual',output[0]);record('cross_encoder.text',output[1])

    def sampler_hook(module,inputs,output):
        record('gumbel.logits',output)
        record('gumbel.cuda_rng_before_noise',torch.cuda.get_rng_state())

    hooks=[model.backbone_net.register_forward_hook(backbone_hook),
        model.text_projector.register_forward_hook(lambda m,i,o:record('text_projector',o)),
        model.box_embeddings.register_forward_hook(lambda m,i,o:record('box_embeddings',o)),
        model.cross_encoder.register_forward_hook(cross_hook),
        model.gumbel.sampler.register_forward_hook(sampler_hook),
        model.gumbel.register_forward_hook(lambda m,i,o:record('gumbel.attention',o)),
        model.decoder[-1].register_forward_hook(lambda m,i,o:record('last_decoder',o))]

    def tensor_hash(value):
        return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()

    def compare(reference,current):
        assert list(reference)==list(current)
        result={}
        for key in reference:
            left,right=reference[key],current[key]
            assert left.shape==right.shape and left.dtype==right.dtype,key
            delta=(left.double()-right.double()).abs()
            result[key]=dict(exact=torch.equal(left,right),max_abs=float(delta.max()) if delta.numel() else 0.,mean_abs=float(delta.mean()) if delta.numel() else 0.)
        return result

    args.output.mkdir()
    results=[];reference=None
    for index,name in enumerate(['reset_seed_first','reset_seed_repeat','advance_rng_control']):
        model.load_state_dict(state,strict=True)
        inputs={k:torch.from_numpy(voxels[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs.update({k:batch[k].cuda().clone() for k in ['det_boxes','det_bbox_label_mask','det_class_ids','superpoint']})
        inputs.update(batch_size=8,text=list(batch['text']),train=False)
        if index<2:seed()
        trace={}
        record('cpu_rng_before',torch.get_rng_state());record('cuda_rng_before',torch.cuda.get_rng_state())
        begin=time.time()
        with torch.no_grad():output=model(inputs)
        torch.cuda.synchronize()
        for key in ['seed_xyz','seed_features','query_points_xyz','last_center','last_pred_size','last_sem_cls_scores','last_proj_queries','proj_tokens']:
            assert torch.isfinite(output[key]).all(),key
            record('output.'+key,output[key])
        record('cpu_rng_after',torch.get_rng_state());record('cuda_rng_after',torch.cuda.get_rng_state())
        # Scoring is read only and happens after the forward; no target enters inputs.
        probs=output['last_sem_cls_scores'].softmax(-1)
        target=targets['positive_map'].cuda().clone()
        target[target>0]=1
        score=(probs*target.unsqueeze(1)).sum(-1)
        for key in ['modify_positive_map','pron_positive_map','rel_positive_map']:
            score=score+(probs*targets[key].cuda().unsqueeze(1)).sum(-1)
        score=score-(probs*targets['other_entity_map'].cuda().unsqueeze(1)).sum(-1)
        record('output.raw_bbs_score',score);record('output.raw_bbs_query',score.argmax(-1))
        assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
        item=dict(case=name,seconds=time.time()-begin,shapes={k:list(v.shape) for k,v in trace.items()},
            hashes={k:tensor_hash(v) for k,v in trace.items()},raw_bbs_query=trace['output.raw_bbs_query'].tolist(),state_unchanged=True)
        if reference is None:
            reference=trace
            torch.save(reference,str(args.output/'reference_trace.pt'))
        else:item['compared_to_first']=compare(reference,trace)
        results.append(item)
        print('REPLAY_CASE '+json.dumps(item),flush=True)
        del output,inputs,score,probs
    for hook in hooks:hook.remove()
    result=dict(status='complete',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        input_sha256=receipt['input_file_sha256'],checkpoint_sha256=checkpoint['sha256'],seed=run['seed'],batch_size=8,
        cases=results,reference_trace_sha256=sha(args.output/'reference_trace.pt'),
        script_sha256=sha(__file__),formal_rows=0,optimizer_steps=0,full_forwards=3,
        scope='same frozen first batch; no original capacity backward or whole6887 replay; hooks are observational',
        repeated_seed_exact=all(v['exact'] for v in results[1]['compared_to_first'].values()))
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REPLAY_COMPLETE '+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
