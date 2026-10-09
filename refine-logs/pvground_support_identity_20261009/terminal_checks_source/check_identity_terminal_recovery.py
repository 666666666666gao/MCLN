"""Restore both actual formal deltas, full model states and Adam after the original job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = json.loads(args.spec.read_bytes())
    root = Path(spec['root'])
    assert int((root / 'campaign.exit').read_text()) == 0
    campaign = json.loads((root / 'campaign/receipt.json').read_bytes())
    assert campaign['status'] == 'complete' and campaign['optimizer_steps_per_arm'] == 3723
    assert campaign['parent_state_exact'] and campaign['preflight_updates_not_carried']
    for name, digest in spec['files'].items():
        assert sha(root / name) == digest, name
    parent_spec = json.loads(Path(spec['parent_spec']).read_bytes())
    assert sha(spec['parent_spec']) == spec['parent_spec_sha256']
    env = json.loads((Path(parent_spec['runtime']) / 'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == parent_spec['env_spec_sha256']
    manifest = json.loads(Path(parent_spec['input_manifest']).read_bytes())
    source, model_source = Path(manifest['model_source']), Path(parent_spec['model_source'])
    assert sha(source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source / name) == digest, name
    assert sha(parent_spec['source_port']) == parent_spec['source_port_sha256']
    for name, digest in json.loads(Path(parent_spec['source_port']).read_bytes())['files'].items():
        assert sha(model_source / name) == digest, name
    for name, digest in parent_spec['runner_files'].items():
        assert sha(Path(parent_spec['helper_root']) / name) == digest, name
    for name in ('mask_support_model_factory.py', 'selected_mask_reference_factory.py', 'mask_support_corrector.py', 'mask_reference.py'):
        assert sha(Path(parent_spec['root']) / name) == parent_spec['new_runner_files'][name]
    official = env['weight_dirs']['scanrefer']['path']
    parents = {official: parent_spec['checkpoint_sha256'], parent_spec['base_terminal']: parent_spec['base_terminal_sha256'], parent_spec['selected_terminal']: parent_spec['selected_terminal_sha256'], spec['support_terminal']: spec['support_terminal_sha256']}
    assert all(sha(path) == digest for path, digest in parents.items())

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    def reset_rng():
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == source / 'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    for path in (parent_spec['helper_root'], parent_spec['root'], str(root), str(model_source)):
        sys.path.insert(0, path)
    from pcdet.config import cfg, cfg_from_yaml_file
    from prepare_data import DataProcessor
    from mask_support_model_factory import build_support_model
    from support_identity_readout import install_support_identity_readout
    from readback_preflight_checks import observed_readback_forward
    from whole_model_preflight_checks import optimizer_restore_exact
    from native_root_bbs import native_root_bbs
    for name in ('models.pv_ground', 'prepare_data'):
        assert model_source in Path(sys.modules[name].__file__).resolve().parents
    cfg_from_yaml_file(str(Path(parent_spec['runtime']) / 'PV-Ground/wandb_config.yaml'), cfg)
    official_payload = torch.load(official, map_location='cpu')
    g_payload = torch.load(parent_spec['base_terminal'], map_location='cpu')
    reference_payload = torch.load(parent_spec['selected_terminal'], map_location='cpu')
    support_payload = torch.load(spec['support_terminal'], map_location='cpu')
    assert support_payload['arm'] == 'content' and support_payload['total_support_updates'] == 7446
    verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['superpoint_files']['val'])
    os.chdir(str(source))
    dataset = Joint3DDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='val', data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False, detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False, augment_det=False, skip_missing_superpoints=True)
    dataset.augment = False
    dataset.augment_det = False
    assert len(dataset) == 9508
    reset_rng()
    raw = next(iter(DataLoader(Subset(dataset, list(range(8))), batch_size=8, shuffle=False, num_workers=2, pin_memory=True, generator=torch.Generator().manual_seed(2027))))
    os.chdir(str(model_source))
    processor = DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), False, 6)
    voxel = processor.collate_batch([processor.forward(dict(points=p.numpy().copy(), use_lead_xyz=True)) for p in raw['point_clouds']])
    assert np.array_equal(voxel['points'][:, 1:].reshape(8, 50000, 6), raw['point_clouds'].numpy())
    batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in raw.items()}
    inputs = {k: torch.from_numpy(voxel[k]).float().cuda() for k in ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
    inputs.update(batch_size=8, text=batch['utterances'], superpoint=batch['superpoint'], train=False, det_boxes=batch['all_detected_boxes'], det_bbox_label_mask=batch['all_detected_bbox_label_mask'], det_class_ids=batch['all_detected_class_ids'])
    prefix = 'boundary_evidence_readback.'
    checks = {}
    for arm in ('shared_text', 'candidate_fused'):
        terminal = root / 'campaign' / arm / 'terminal.pth'
        terminal_digest = sha(terminal)
        saved = torch.load(terminal, map_location='cpu')
        assert saved['arm'] == arm and saved['candidate_specific'] == (arm == 'candidate_fused')
        assert saved['step'] == saved['new_identity_updates'] == 3723
        assert saved['parent_support_updates'] == 7446 and saved['seed'] == 2027
        assert saved['parent_support_terminal_sha256'] == spec['support_terminal_sha256']
        assert saved['spec_sha256'] == sha(args.spec)
        assert saved['optimizer_reinitialized'] and not saved['preflight_state_carried']
        assert saved['parent_state_tensors'] == 1314 and saved['deployed_full_state_tensors'] == 1299
        delta = saved['state_delta']
        assert len(delta) == 8 and all(name.startswith(prefix) for name in delta)
        head_state = {name[len(prefix):]: value for name, value in delta.items()}
        model, config, load = build_support_model(cfg, official_payload, g_payload, reference_payload, manifest['data_root'], support_payload)
        assert load['full_state_tensors'] == 1314
        expected = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if not name.startswith(prefix)}
        expected.update(delta)
        head = install_support_identity_readout(model, arm == 'candidate_fused', head_state)
        assert len(expected) == len(model.state_dict()) == 1299
        assert set(expected) == set(model.state_dict())
        assert all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in expected.items())
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 107040
        model.cuda().eval()
        optimizer = torch.optim.AdamW(head.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
        optimizer.load_state_dict(saved['optimizer'])
        optimizer_check = optimizer_restore_exact(optimizer, saved['optimizer'])
        assert len(optimizer.state) == 8 and all(int(value['step']) == 3723 for value in optimizer.state.values())
        reset_rng()
        with torch.no_grad():
            predictions, call = observed_readback_forward(model, inputs)
            replay = head(predictions['last_semantic_query_before_readback'], None, None, None, dict(predictions))
            scores = native_root_bbs(predictions['last_sem_cls_scores'], batch)
        assert torch.equal(replay, predictions['last_semantic_query_after_readback'])
        assert scores.shape == (8, 256) and torch.isfinite(scores).all()
        assert all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in expected.items())
        assert sha(terminal) == terminal_digest
        checks[arm] = dict(checkpoint_sha256=terminal_digest, checkpoint_bytes=terminal.stat().st_size,
            full_state_tensors=1299, strict_state_reconstruction=True, optimizer=optimizer_check,
            optimizer_step=3723, actual_deployment_forward=call,
            cached_readout_replay_exact=True, native_score_shape=list(scores.shape),
            independent_formal_9508_restore_replay=False)
        del model, head, optimizer, predictions, replay, scores, expected, saved
    assert all(sha(path) == digest for path, digest in parents.items())
    result = dict(status='FORMAL_TERMINAL_FULL_MODEL_AND_ADAM_RECOVERY_COMPLETE', arms=checks,
        actual_validation_rows=list(range(8)), scope='Strict complete state restoration, Adam restoration and one actual integrated B8 deployment forward per arm; not a repeated 9508 metric run.',
        parent_dependencies_unchanged=True, no_optimizer_update=True, no_weight_files_written=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.monotonic() - started, source_sha256=sha(__file__), best_updated=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
