"""Frozen Scan parent: observed VSA support and hypothetical center support on fit scenes."""
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
    root = Path(__file__).resolve().parent
    spec = json.loads((root / 'spec.json').read_bytes())
    started = time.time()
    runtime = Path(spec['runtime'])
    env = json.loads((runtime / 'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
    source = Path(spec['model_source'])
    port = json.loads(Path(spec['source_port']).read_bytes())
    bundle = json.loads((runtime / 'source_bundle_receipt.json').read_bytes())
    for name, entry in bundle['sources']['PV-Ground']['files'].items():
        expected = port['after_sha256'] if 'PV-Ground/' + name == port['file'] else entry['sha256']
        assert sha(source / name) == expected, name
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((dataset_source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source / name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    os.chdir(str(dataset_source))
    sys.path.insert(0, str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])

    class Dataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            first = {}
            for index, row in enumerate(annos):
                physical = row['scan_id'].split('_')[0]
                fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + physical).encode()).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
                if fold != 0 and physical not in first:
                    first[physical] = index
            assert actual == partitions
            scenes = sorted(first)
            indices = np.linspace(0, len(scenes) - 1, spec['scene_count'], dtype=np.int64)
            self.selected_ids = [first[scenes[i]] for i in indices]
            assert len(set(self.selected_ids)) == spec['scene_count']
            super()._scene_graph_parse([annos[i] for i in self.selected_ids])

    print('FIT_SUPPORT_DATA_LOADING', flush=True)
    dataset = Dataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
                      data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
                      detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
                      augment_det=False, skip_missing_superpoints=True)
    assert not set(dataset.selected_ids).intersection(partitions['holdout'])
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    from models.pv_ground import PVGround
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
    from prepare_data import DataProcessor
    from pvground_support_observation import stack_indices_to_global
    assert Path(sys.modules['models.pv_ground'].__file__).resolve() == source / 'models/pv_ground.py'
    parent = env['weight_dirs']['scanrefer']
    assert sha(parent['path']) == parent['sha256'] == spec['checkpoint_sha256']
    seed(spec['seed'])
    payload = torch.load(parent['path'], map_location='cpu')
    config = payload['config']
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert all(k.startswith('module.') for k in payload['model'])
    state = {k[7:]: v for k, v in payload['model'].items()}
    cfg_from_yaml_file(str(source / 'wandb_config.yaml'), cfg)
    model = PVGround(cfg, num_class=256, num_queries=256, num_decoder_layers=6,
                     self_position_embedding=config.self_position_embedding, contrastive_align_loss=True,
                     butd=True, pointnet_ckpt=None, data_path=manifest['data_root'], self_attend=config.self_attend)
    assert set(model.state_dict()) - set(state) == {'text_encoder.embeddings.position_ids'}
    assert torch.equal(model.text_encoder.embeddings.position_ids,
                       torch.arange(model.text_encoder.config.max_position_embeddings).expand(1, -1))
    model.text_encoder.embeddings.register_buffer('position_ids', model.text_encoder.embeddings.position_ids, persistent=False)
    model.load_state_dict(state, strict=True)
    model.cuda().eval().requires_grad_(False)
    processors = {flag: DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), flag, 6)
                  for flag in [False, True]}
    bounds = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    vsa = model.backbone_net.vsa
    names = {id(vsa.SA_rawpoints): 'raw_points'}
    names.update({id(layer): name for layer, name in zip(vsa.SA_layers, vsa.SA_layer_names)})
    original_aggregate = vsa.aggregate_keypoint_features_from_one_source
    original_ball = pointnet2_utils.ball_query
    context, packets = {}, []

    def aggregate(**kwargs):
        assert not kwargs['filter_neighbors_with_roi']
        context['source'] = names[id(kwargs['aggregate_func'])]
        labels = kwargs['xyz_bs_idxs'].detach().cpu().numpy()
        context['batch'] = labels.astype(np.int64)
        assert np.array_equal(labels, context['batch'])
        return original_aggregate(**kwargs)

    def ball(radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt):
        indices, empty = original_ball(radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        counts = xyz_batch_cnt.detach().cpu().numpy()
        query_counts = new_xyz_batch_cnt.detach().cpu().numpy()
        assert np.array_equal(context['batch'], np.repeat(np.arange(len(counts)), counts))
        global_indices, valid = stack_indices_to_global(indices.detach().cpu().numpy(), empty.detach().cpu().numpy(), counts, query_counts)
        selected = np.concatenate([np.arange(sum(query_counts[:b]), sum(query_counts[:b + 1]), 32) for b in range(len(query_counts))])
        packets.append(dict(source=context['source'], radius=float(radius), cap=int(nsample),
                            xyz=xyz.detach().cpu().numpy().copy(), batch=context['batch'].copy(),
                            query_batch=np.repeat(np.arange(len(query_counts)), query_counts), valid=valid,
                            selected=global_indices[selected], selected_batch=np.repeat(np.arange(len(query_counts)), 32)))
        return indices, empty

    records, forwards = [], 0
    selected_queries = np.arange(0, 256, 8)
    vsa.aggregate_keypoint_features_from_one_source = aggregate
    pointnet2_utils.ball_query = ball
    # Restore observational bindings after a failed forward as well as success.
    try:
        for augmented in [False, True]:
            dataset.augment = augmented
            dataset.augment_det = augmented
            for start in range(0, spec['scene_count'], 8):
                cases, raw = [], []
                ids = dataset.selected_ids[start:start + 8]
                for row_id in ids:
                    seed(spec['seed'] + row_id)
                    row = dataset[row_id]
                    pc = row['point_clouds']
                    root_mask = np.asarray(row['gt_masks'][0], dtype=np.bool_)
                    assert root_mask.shape == (50000,) and pc.shape == (50000, 6)
                    inside = ((pc[:, :3] >= bounds[:3]) & (pc[:, :3] < bounds[3:])).all(axis=1)
                    raw.append(dict(points=pc, text=row['utterances'], det_boxes=row['all_detected_boxes'],
                                    det_bbox_label_mask=row['all_detected_bbox_label_mask'], det_class_ids=row['all_detected_class_ids'],
                                    superpoint=row['superpoint'], root_box=np.concatenate([row['center_label'][0, :3], row['size_gts'][0]])))
                    cases.append(dict(row_id=row_id, scan_id=row['scan_ids'], augmented=augmented,
                                      point_sha256=hashlib.sha256(pc.tobytes()).hexdigest(), root_points=int(root_mask.sum()),
                                      points_outside_voxel_range=int((~inside).sum()), root_points_outside_voxel_range=int((root_mask & ~inside).sum())))
                voxels = processors[augmented].collate_batch([processors[augmented].forward(dict(points=r['points'].copy(), use_lead_xyz=True)) for r in raw])
                assert np.array_equal(voxels['points'][:, 1:].reshape(8, 50000, 6), np.stack([r['points'] for r in raw]))
                inputs = {k: torch.from_numpy(voxels[k]).float().cuda() for k in ['points', 'voxels', 'voxel_coords', 'voxel_num_points']}
                for key in ['det_boxes', 'det_bbox_label_mask', 'det_class_ids']:
                    inputs[key] = torch.from_numpy(np.stack([r[key] for r in raw])).cuda()
                inputs.update(superpoint=torch.stack([r['superpoint'] for r in raw]).cuda(), batch_size=8,
                              text=[r['text'] for r in raw], train=False)
                packets.clear()
                seed(spec['seed'] + start)
                with torch.no_grad():
                    output = model(inputs)
                forwards += 1
                assert len(packets) == 10
                centers = output['last_center'].detach().cpu().numpy()
                sizes = output['last_pred_size'].detach().cpu().numpy()
                assert np.isfinite(centers).all() and np.isfinite(sizes).all() and (sizes > 0).all()
                for b, case in enumerate(cases):
                    gt = raw[b]['root_box'].astype(np.float64)
                    lower, upper = centers[b].astype(np.float64) - sizes[b] / 2., centers[b].astype(np.float64) + sizes[b] / 2.
                    intersection = np.maximum(0., np.minimum(upper, gt[:3] + gt[3:] / 2) - np.maximum(lower, gt[:3] - gt[3:] / 2)).prod(axis=1)
                    iou = intersection / (sizes[b].astype(np.float64).prod(axis=1) + gt[3:].prod() - intersection)
                    case.update(query_indices=selected_queries.tolist(), sampled_root_iou=iou[selected_queries].tolist(),
                                full256_root_oracle25=bool((iou > .25).any()), full256_root_oracle50=bool((iou > .5).any()), sources=[])
                    for packet in packets:
                        xyz = packet['xyz'][packet['batch'] == b].astype(np.float64)
                        queries = centers[b, selected_queries].astype(np.float64)
                        counts, nearest = [], []
                        for center in queries:
                            distances = np.linalg.norm(xyz - center, axis=1)
                            counts.append(int((distances < packet['radius']).sum()))
                            nearest.append(float(distances.min()))
                        chosen = packet['selected'][packet['selected_batch'] == b]
                        unique = [int(len(np.unique(index[index >= 0]))) for index in chosen]
                        case['sources'].append(dict(source=packet['source'], radius_m=packet['radius'], cap=packet['cap'],
                            actual_vsa_queries=int((packet['query_batch'] == b).sum()),
                            actual_vsa_empty=int((~packet['valid'][packet['query_batch'] == b]).sum()),
                            sampled_vsa_unique_support_rows=unique,
                            hypothetical_center_support_counts=counts, hypothetical_center_nearest_m=nearest,
                            hypothetical_gt_center_support=int((np.linalg.norm(xyz - gt[:3], axis=1) < packet['radius']).sum())))
                    records.append(case)
                with (root / 'rows.jsonl').open('a') as stream:
                    for case in cases:
                        stream.write(json.dumps(case, allow_nan=False) + '\n')
                print('FIT_SUPPORT_BATCH ' + json.dumps(dict(augmented=augmented, start=start, forwards=forwards, rows=len(records))), flush=True)
                del output, inputs, raw, voxels
    finally:
        del vsa.aggregate_keypoint_features_from_one_source
        pointnet2_utils.ball_query = original_ball
    assert forwards == 4 and len(records) == 32
    assert len({r['scan_id'].split('_')[0] for r in records}) == 16
    assert all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
    receipt = dict(status='complete', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   elapsed_seconds=time.time() - started, selected_fit_row_ids=dataset.selected_ids,
                   selected_physical_scenes=sorted({r['scan_id'].split('_')[0] for r in records}),
                   rows=32, forwards=forwards, optimizer_steps=0, formal_rows=0, checkpoint_unchanged=True,
                   checkpoint_sha256=spec['checkpoint_sha256'], env_spec_sha256=spec['env_spec_sha256'],
                   rows_sha256=sha(root / 'rows.jsonl'), script_sha256=sha(__file__),
                   scope='16 fit scenes, native unaugmented/augmented inputs, frozen eval parent; not accuracy evaluation or matched-transform causal comparison')
    (root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('FIT_SUPPORT_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
