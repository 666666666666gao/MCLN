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
    port=json.loads((runtime/'source_port.json').read_bytes())
    for name,info in bundle['sources']['PV-Ground']['files'].items():
        expected=port['after_sha256'] if 'PV-Ground/'+name==port['file'] else info['sha256']
        assert sha(runtime/'PV-Ground'/name)==expected,name
    checkpoint=env['weight_dirs']['scanrefer']
    assert sha(checkpoint['path'])==run['checkpoint_sha256']==checkpoint['sha256']
    fixtures=args.spec.parent/'inputs'
    receipt=json.loads((fixtures/'receipt.json').read_bytes())
    assert receipt['status']=='pass' and receipt['batch_size']==8
    assert sha(fixtures/'first_batch.pt')==receipt['input_file_sha256']
    os.chdir(str(runtime/'PV-Ground'));sys.path.insert(0,str(runtime/'PV-Ground'))
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
    args.output.mkdir()
    reference={};differences={};observations=[];current={}
    def capture(name,tensor):
        value=tensor.detach().cpu().clone()
        current[name]=value
        if name in reference:
            left=reference[name]
            assert left.shape==value.shape and left.dtype==value.dtype,name
            delta=(left.double()-value.double()).abs()
            differences[name]=dict(exact=torch.equal(left,value),max_abs=float(delta.max()),mean_abs=float(delta.mean()))
    def sparse_hook(name):
        def hook(module,inputs,output):
            coords=output.indices.detach().cpu().clone()
            features=output.features.detach().cpu().clone()
            order=np.lexsort(tuple(coords[:,i].numpy() for i in [3,2,1,0]))
            order=torch.from_numpy(order.copy()).long()
            capture(name+'.ordered_indices',coords)
            capture(name+'.canonical_indices',coords[order])
            capture(name+'.canonical_features',features[order])
            counts=torch.bincount(coords[:,0].long(),minlength=8)
            expected=torch.repeat_interleave(torch.arange(8),counts)
            observations.append(dict(case=case,stage=name,rows=len(coords),batch_counts=counts.tolist(),
                batch_segment_mismatches=int((coords[:,0]!=expected).sum()),
                batch_transitions=int((coords[1:,0]!=coords[:-1,0]).sum())))
        return hook
    b=model.backbone_net
    hooks=[b.vfe.register_forward_hook(lambda m,i,o:capture('vfe',o['voxel_features'])),
           b.to_bev.register_forward_hook(lambda m,i,o:capture('bev',o['spatial_features']))]
    for name in ['conv_input','conv1','conv2','conv3','conv4','conv_out']:
        hooks.append(getattr(b.backbone_3d,name).register_forward_hook(sparse_hook(name)))
    for index,module in enumerate(b.vsa.SA_layers):
        name=b.vsa.SA_layer_names[index]
        hooks.append(module.register_forward_hook(lambda m,i,o,n=name:capture('vsa.'+n,o[1])))
    hooks.append(b.vsa.SA_rawpoints.register_forward_hook(lambda m,i,o:capture('vsa.raw_points',o[1])))
    begin=time.time()
    for case in ['seeded_first','seeded_repeat']:
        seed();current={}
        inputs={k:torch.from_numpy(voxels[k]).float().cuda() for k in ['points','voxels','voxel_coords','voxel_num_points']}
        inputs['batch_size']=8
        with torch.no_grad():result=b(inputs)
        capture('backbone.point_features',result['point_features'])
        capture('backbone.inds',result['inds'])
        assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
        if case=='seeded_first':reference=current
        print('BACKBONE_CASE '+case,flush=True)
        del result,inputs
    for h in hooks:h.remove()
    result=dict(status='complete',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        seconds=time.time()-begin,observations=observations,differences=differences,
        first_observed_difference=next((k for k,v in differences.items() if not v['exact']),None),
        input_sha256=receipt['input_file_sha256'],checkpoint_sha256=checkpoint['sha256'],
        script_sha256=sha(__file__),backbone_forwards=2,full_model_forwards=0,optimizer_steps=0,formal_rows=0,
        scope='one frozen scene batch; CPU canonical comparison only; no sorting of model inputs or outputs')
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BACKBONE_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
