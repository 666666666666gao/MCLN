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
    model_source = Path('/root/autodl-tmp/mcln_pvground_source_query_source_20260908_v1/PV-Ground')
    spec = json.loads((root / 'env_spec.json').read_bytes())
    build = json.loads((root / 'build_receipt.json').read_bytes())
    assert build['status'] == 'pass'
    assert build['spec_sha256'] == hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert json.loads((root / 'agent_kernel_receipt.json').read_bytes())['status'] == 'pass'
    source = json.loads((root / 'source_bundle_receipt.json').read_bytes())
    port = json.loads((model_source.parent/'source_port.json').read_bytes())
    assert sha(model_source.parent/'source_port.json')=='834861df76493163580ea66ede20fdcea53e0a1fc024b5be97ab23a0844bc6f7'
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
    from pvground_source_query import install_source_query_read
    install_source_query_read(model)
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

    model.eval()
    inputs,batch=batch_at(0,eval_owner)
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def restore():
        random.setstate(states[0]);np.random.set_state(states[1])
        torch.set_rng_state(states[2]);torch.cuda.set_rng_state_all(states[3])
    traces=[];rngs=[];source_outputs=[]
    handle=reader.register_forward_hook(lambda module,args,value:source_outputs.append(value.detach().cpu().clone()))
    with torch.no_grad():
        for enabled in [False,False,True]:
            restore();reader.enabled=enabled
            output=model({k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in inputs.items()})
            trace={}
            for key,value in output.items():
                if torch.is_tensor(value):trace[key]=value.detach().cpu().clone()
                elif isinstance(value,list) and value and all(torch.is_tensor(v) for v in value):
                    for index,item in enumerate(value):trace[key+'.'+str(index)]=item.detach().cpu().clone()
            traces.append(trace)
            rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
            del output
    handle.remove()
    differences=[]
    for trace in traces[1:]:
        pairs={}
        for key,base in traces[0].items():
            value=trace[key]
            pairs[key]=dict(exact=torch.equal(base,value),shape=list(base.shape),
                max_abs=float((base.double()-value.double()).abs().max()) if base.numel() else 0.0)
        differences.append(pairs)
    receipt=dict(status='diagnosed',cases=['disabled','disabled_repeat','enabled'],
        differences=differences,source_output_max=[float(v.abs().max()) for v in source_outputs],
        rng_end_equal=[torch.equal(rngs[0][0],r[0]) and all(torch.equal(a,b) for a,b in zip(rngs[0][1],r[1])) for r in rngs[1:]],
        model_forwards=3,optimizer_steps=0,formal_rows=0,module_sha256=sha(Path(__file__).parent/'pvground_source_query.py'),
        script_sha256=sha(__file__),time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(nonexact=[{k:v for k,v in d.items() if not v['exact']} for d in differences],
        source_output_max=receipt['source_output_max'],rng_end_equal=receipt['rng_end_equal'])),flush=True)


if __name__ == '__main__':
    main()
