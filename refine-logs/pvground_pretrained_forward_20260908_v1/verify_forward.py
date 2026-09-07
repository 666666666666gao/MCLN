"""Strict-load the published ScanRefer checkpoint and run four fixed train inputs."""
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
    option = parser.parse_args()
    started = time.time()
    root = option.runtime.resolve()
    spec = json.loads((root / 'env_spec.json').read_bytes())
    build = json.loads((root / 'build_receipt.json').read_bytes())
    assert build['status'] == 'pass'
    assert build['spec_sha256'] == hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert json.loads((root / 'agent_kernel_receipt.json').read_bytes())['status'] == 'pass'
    source_manifest = json.loads((root / 'source_bundle_receipt.json').read_bytes())
    port = json.loads((root / 'source_port.json').read_bytes())
    for name, info in source_manifest['sources']['PV-Ground']['files'].items():
        expected = port['after_sha256'] if 'PV-Ground/' + name == port['file'] else info['sha256']
        assert sha(root / 'PV-Ground' / name) == expected, name
    fixture_receipt = json.loads((option.fixtures / 'receipt.json').read_bytes())
    assert fixture_receipt['status'] == 'pass' and fixture_receipt['training_fixture_rows'] == 4
    checkpoint = spec['weight_dirs']['scanrefer']
    assert sha(checkpoint['path']) == checkpoint['sha256']
    option.output.mkdir()
    os.chdir(str(root / 'PV-Ground'))
    sys.path.insert(0, str(root / 'PV-Ground'))
    import numpy as np
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from models.pv_ground import PVGround
    from prepare_data import DataProcessor
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
    assert all(name.startswith('module.') for name in payload['model'])
    state = {name[7:]: value for name, value in payload['model'].items()}
    assert len(state) == 1234
    cfg_from_yaml_file('wandb_config.yaml', cfg)
    model = PVGround(cfg, num_class=256, num_queries=config.num_target,
        num_decoder_layers=config.num_decoder_layers, self_position_embedding=config.self_position_embedding,
        contrastive_align_loss=config.use_contrastive_align, butd=config.butd,
        pointnet_ckpt=None, data_path=fixture_receipt['data_root'], self_attend=config.self_attend)
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    strict_receipt = {'status':'pass', 'checkpoint_sha256':checkpoint['sha256'], 'state_tensors':len(state),
        'checkpoint_epoch':int(payload['epoch']), 'saved_config_model_name':config.model,
        'actual_model_class':model.__class__.__name__, 'missing_keys':[], 'unexpected_keys':[],
        'parameter_count':sum(p.numel() for p in model.parameters()), 'model_forwards':0}
    (option.output / 'strict_load.json').write_text(json.dumps(strict_receipt, indent=2)+'\n')
    print('PVG_STRICT_LOAD_PASS '+json.dumps(strict_receipt), flush=True)
    model.cuda().eval().requires_grad_(False)
    processor = DataProcessor(cfg.DATA_PROCESSOR, np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), False, 6)
    records = []
    for start in (0, 2):
        fixture_rows = fixture_receipt['rows'][start:start+2]
        fixture_inputs = []
        for row in fixture_rows:
            path = option.fixtures / row['file']
            assert sha(path) == row['file_sha256']
            value = torch.load(str(path), map_location='cpu')
            assert sorted(value) == ['det_bbox_label_mask','det_boxes','det_class_ids','point_clouds','superpoint','text']
            fixture_inputs.append(value)
        batch = {k:torch.cat([v[k] for v in fixture_inputs], dim=0) for k in fixture_inputs[0] if k != 'text'}
        batch['text'] = [v['text'][0] for v in fixture_inputs]
        voxel_rows = [processor.forward({'points':pc.numpy().copy(), 'use_lead_xyz':True}) for pc in batch['point_clouds']]
        voxels = processor.collate_batch(voxel_rows)
        assert np.array_equal(voxels['points'][:, 1:].reshape(2, 50000, 6), batch['point_clouds'].numpy())
        # Match the author's TrainTester._get_inputs conversion, including integer-valued arrays stored as float.
        inputs = {k:torch.from_numpy(voxels[k]).float().cuda() for k in ('points','voxels','voxel_coords','voxel_num_points')}
        inputs.update({k:batch[k].cuda() for k in ('det_boxes','det_bbox_label_mask','det_class_ids','superpoint')})
        inputs.update(batch_size=2, text=batch['text'])
        torch.cuda.reset_peak_memory_stats()
        begin = time.time()
        with torch.no_grad():
            output = model(inputs)
        torch.cuda.synchronize()
        forward_seconds = time.time() - begin
        for key in ('last_center','last_pred_size','last_sem_cls_scores','last_proj_queries','proj_tokens','seed_xyz','seed_features'):
            assert torch.isfinite(output[key]).all(), key
        assert output['last_center'].shape == output['last_pred_size'].shape == (2,256,3)
        assert output['seed_xyz'].shape == (2,1024,3)
        for index, row in enumerate(fixture_rows):
            for key in ('sp_last_pred_masks','last_pred_masks'):
                assert torch.isfinite(output[key][index]).all(), key
            boxes = torch.cat([output['last_center'][index],output['last_pred_size'][index]], dim=-1).cpu().numpy()
            np.save(str(option.output / ('row_%05d_boxes.npy' % row['training_row_id'])), boxes)
            records.append({'training_row_id':row['training_row_id'],'scan_id':row['scan_id'],
                'box_shape':list(boxes.shape),'positive_size_boxes':int((boxes[:,3:]>0).all(-1).sum()),
                'query_mask_shape':list(output['sp_last_pred_masks'][index].shape),
                'text_mask_shape':list(output['last_pred_masks'][index].shape),
                'all_output_checks_finite':True,'raw_point_order_preserved':True,
                'voxel_count':int((voxels['voxel_coords'][:,0]==index).sum()),
                'batch_forward_seconds':forward_seconds,'batch_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        print('PVG_FORWARD_BATCH '+json.dumps(records[-2:]), flush=True)
        del output, inputs
    model.cpu()
    assert all(torch.equal(value, state[name]) for name,value in model.state_dict().items())
    assert sha(checkpoint['path']) == checkpoint['sha256']
    receipt = {'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'strict_load':strict_receipt,'rows':records,'fixture_receipt_sha256':sha(option.fixtures/'receipt.json'),
        'env_spec_sha256':build['spec_sha256'],'source_commits':spec['source_commits'],'source_port':port,
        'full_model_forwards':2,'training_fixture_rows':4,'formal_rows':0,'optimizer_steps':0,
        'loaded_state_unchanged':True,'metric_claim':False,'script_sha256':sha(__file__),'elapsed_seconds':time.time()-started}
    (option.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('PVG_PRETRAINED_FORWARD_PASS '+json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
