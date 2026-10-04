"""Construct the frozen4506 provider and test two real readback updates.

This entry point performs CPU construction or a real-batch preflight only.
It does not run formal training/evaluation or create a disk weight file.
"""
import argparse
import copy
import datetime
import hashlib
import io
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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    begin = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--mode', choices=['cpu', 'preflight'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    output = Path(spec['root'])
    runtime = Path(spec['runtime'])
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    assert spec['lr'] == 1e-5 and spec['weight_decay'] == .0005 and spec['clip_norm'] == .1
    assert spec['geometry_hits50'] == 4506 and spec['geometry_frozen']
    assert isinstance(spec['use_geometry_evidence'], bool)
    for name, digest in spec['runner_files'].items():
        assert sha(output / name) == digest, name
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    dataset_source = Path(manifest['model_source'])
    assert sha(dataset_source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((dataset_source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(dataset_source / name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    env = json.loads((runtime / 'env_spec.json').read_bytes())
    env_sha = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert env_sha == spec['env_spec_sha256']
    interface = json.loads(Path(spec['training_interface_receipt']).read_bytes())
    assert interface['status'] == 'pass' and interface['optimizer_steps'] == 2
    assert interface['env_spec_sha256'] == env_sha and interface['strict_cpu_restore']
    assert interface['direct_routing_verified'] and interface['added_state_tensors'] == 37
    assert interface['new_parameters'] == 923616
    assert interface['source_port_sha256'] == sha(spec['parent_source_port'])
    assert interface['module_sha256'] == spec['runner_files']['pvground_task_observation_query.py']
    model_source = Path(spec['model_source'])
    assert sha(spec['source_port']) == spec['source_port_sha256']
    port = json.loads(Path(spec['source_port']).read_bytes())
    assert port['boundary_evidence_readback'] and port['native_semantic_head_deferred']
    assert port['call_position'] == 'after native Mask generation'
    for name, digest in port['files'].items():
        assert sha(model_source / name) == digest, name
    official_weight = env['weight_dirs']['scanrefer']
    assert sha(official_weight['path']) == official_weight['sha256'] == spec['checkpoint_sha256']
    assert sha(spec['base_terminal']) == spec['base_terminal_sha256']
    assert sha(spec['geometry_terminal']) == spec['geometry_terminal_sha256']

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    if args.mode == 'preflight':
        torch.cuda.reset_peak_memory_stats()

    def reset_rng():
        random.seed(spec['seed'])
        np.random.seed(spec['seed'])
        torch.manual_seed(spec['seed'])
        if args.mode == 'preflight':
            torch.cuda.manual_seed_all(spec['seed'])

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(str(dataset_source))
    sys.path.insert(0, str(dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == dataset_source / 'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    sys.path.insert(0, str(model_source))
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    from readback_model_factory import build_readback_model
    from readback_preflight_checks import (observed_readback_forward, repeated_forward_differences,
        zero_readback_cached_native_head, native_bbs_witness, readback_semantic_route)
    from pvground_semantic_assignment import semantic_assignment_correction
    from whole_model_preflight_checks import optimizer_restore_exact
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ('src.joint_det_dataset', 'models.pv_ground', 'models.losses', 'main_utils', 'prepare_data')}
    for name in ('models.pv_ground', 'models.losses', 'main_utils', 'prepare_data'):
        assert model_source in Path(imported[name]).parents
    write_json(output / 'imports.json', dict(files=imported, sha256={k: sha(v) for k, v in imported.items()}))
    cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
    model, config, initial, load = build_readback_model(cfg,
        torch.load(official_weight['path'], map_location='cpu'),
        torch.load(spec['base_terminal'], map_location='cpu'),
        torch.load(spec['geometry_terminal'], map_location='cpu'), manifest['data_root'],
        spec['use_geometry_evidence'])
    load.update(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
                spec_sha256=sha(args.spec), geometry_terminal=spec['geometry_terminal'])
    write_json(output / 'load.json', load)
    if args.mode == 'cpu':
        print('READBACK_CPU_FACTORY_PASS ' + json.dumps(load), flush=True)
        return
    model.cuda()
    readback = model.boundary_evidence_readback
    training = copy.copy(config)
    training.frozen = False
    training.small_lr = False
    training.lr = spec['lr']
    training.lr_backbone = spec['lr']
    assert training.weight_decay == spec['weight_decay'] and training.clip_norm == spec['clip_norm']
    criterion, set_criterion = BaseTrainTester.get_criterion(training)
    processor = DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), False, 6)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert len(trainable) == 23 and all(name.startswith('boundary_evidence_readback.') for name in trainable)
    core_names = set(initial)
    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_local_training_id']
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    os.chdir(str(dataset_source))
    dataset = FitDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
        detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
        augment_det=False, skip_missing_superpoints=True)
    dataset.augment = False
    reference = json.loads((Path(spec['reference_fixtures']) / 'receipt.json').read_bytes())
    reset_rng()
    for row in reference['rows']:
        sample = dataset[row['training_row_id']]
        checks = dict(point_clouds=sample['point_clouds'], det_boxes=sample['all_detected_boxes'],
            det_bbox_label_mask=sample['all_detected_bbox_label_mask'],
            det_class_ids=sample['all_detected_class_ids'], superpoint=sample['superpoint'].numpy())
        assert sample['utterances'] == row['text']
        for name, array in checks.items():
            assert hashlib.sha256(array.tobytes()).hexdigest() == row['tensor_sha256'][name], name
    reset_rng()
    loader = DataLoader(Subset(dataset, partitions['fit']), batch_size=8, shuffle=True, num_workers=2,
        generator=torch.Generator().manual_seed(spec['seed']), pin_memory=True, drop_last=False)
    batch = next(iter(loader))
    voxel_rows = [processor.forward(dict(points=pc.numpy().copy(), use_lead_xyz=True)) for pc in batch['point_clouds']]
    voxels = processor.collate_batch(voxel_rows)
    assert np.array_equal(voxels['points'][:, 1:].reshape(8, 50000, 6), batch['point_clouds'].numpy())
    batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    inputs = {k: torch.from_numpy(voxels[k]).float().cuda() for k in
        ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
    inputs.update(batch_size=8, text=batch['utterances'], superpoint=batch['superpoint'], train=False,
        det_boxes=batch['all_detected_boxes'], det_bbox_label_mask=batch['all_detected_bbox_label_mask'],
        det_class_ids=batch['all_detected_class_ids'])
    input_keys = set(inputs)
    input_tensors = {key: value.detach().clone() for key, value in inputs.items() if torch.is_tensor(value)}
    input_text = list(inputs['text'])
    def unchanged_input():
        assert set(inputs) == input_keys and inputs['text'] == input_text
        assert all(torch.equal(inputs[key], value) for key, value in input_tensors.items())
        return dict(original_dictionary_keys_exact=True, original_tensor_leaves_exact=True,
                    original_text_exact=True, tensor_keys=list(input_tensors))
    model.eval()
    model.boundary_evidence_readback = None
    reset_rng()
    with torch.no_grad():
        reference_predictions = model(dict(inputs))
    unchanged_input()
    reset_rng()
    with torch.no_grad():
        disabled_repeat_predictions = model(dict(inputs))
    unchanged_input()
    disabled_repeat = repeated_forward_differences(reference_predictions, disabled_repeat_predictions)
    write_json(output / 'disabled_repeat_differences.json', disabled_repeat)
    del disabled_repeat_predictions
    model.boundary_evidence_readback = readback
    reset_rng()
    with torch.no_grad():
        zero_predictions, zero_call = observed_readback_forward(model, inputs)
    unchanged_input()
    zero_cross_forward = repeated_forward_differences(reference_predictions, zero_predictions)
    write_json(output / 'zero_cross_forward_differences.json', zero_cross_forward)
    zero_native = zero_readback_cached_native_head(model, zero_predictions)
    fixed_zero = zero_call['fixed_geometry']
    del zero_predictions, reference_predictions
    readback.train()
    optimizer = torch.optim.AdamW(tuple(trainable.values()), lr=spec['lr'], weight_decay=spec['weight_decay'])
    steps = []
    for index in range(2):
        reset_rng()
        step_begin = time.perf_counter()
        predictions, call = observed_readback_forward(model, inputs)
        input_check = unchanged_input()
        fixed = call['fixed_geometry']
        score = native_bbs_witness(predictions['last_sem_cls_scores'], batch)
        matching = []
        hook = set_criterion.matcher.register_forward_hook(
            lambda module, arguments, result: matching.append([(q.clone(), t.clone()) for q, t in result]))
        assert not set(predictions).intersection(batch)
        predictions.update(batch)
        native, predictions = criterion(predictions, 6, set_criterion,
            query_points_obj_topk=training.query_points_obj_topk)
        hook.remove()
        assert len(matching) == 7 and torch.isfinite(native)
        correction, counts = semantic_assignment_correction(predictions, batch, matching[1], set_criterion.eos_coef)
        route = readback_semantic_route(predictions, correction, readback, index)
        loss = native + correction
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
        assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in trainable)
        norm = torch.nn.utils.clip_grad_norm_(tuple(trainable.values()), spec['clip_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        steps.append(dict(update=index + 1, loss=float(loss), correction=float(correction),
            assignment=counts, semantic_route=route, native_bbs=score, fixed_geometry=fixed,
            call_order=call, gradient_norm=float(norm), seconds=time.perf_counter() - step_begin))
        steps[-1]['unchanged_inputs'] = input_check
    assert not model.training
    assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)
    stream = io.BytesIO()
    selected = set(trainable)
    torch.save(dict(delta={n: v.detach().cpu() for n, v in model.state_dict().items() if n in selected},
                    optimizer=optimizer.state_dict()), stream)
    stream.seek(0)
    reloaded = torch.load(stream, map_location='cpu')
    restored = dict(initial)
    restored.update(reloaded['delta'])
    model.load_state_dict(restored, strict=True)
    optimizer.load_state_dict(reloaded['optimizer'])
    assert all(torch.equal(value.detach().cpu(), restored[name]) for name, value in model.state_dict().items())
    assert all(int(state['step']) == 2 for state in optimizer.state.values())
    optimizer_check = optimizer_restore_exact(optimizer, reloaded['optimizer'])
    receipt = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
        batch_size=8, optimizer_steps=2, use_geometry_evidence=spec['use_geometry_evidence'],
        readback_parameters=load['readback_parameters'], readback_state_tensors=23,
        zero_residual_native_semantic_exact=True, zero_cached_native_head=zero_native,
        disabled_repeat_differences=disabled_repeat, zero_cross_forward_differences=zero_cross_forward,
        zero_fixed_geometry=fixed_zero, zero_call_order=zero_call,
        geometry_provider_and_g_states_exact=True, upstream_eval=True, steps=steps,
        original_inputs_unchanged=unchanged_input(),
        serialization_bytes=stream.getbuffer().nbytes, optimizer_exact_check=optimizer_check,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_seconds=time.perf_counter() - begin, weight_files_created=0, formal_rows=0, accuracy_result=False)
    write_json(output / 'preflight.json', receipt)
    print('READBACK_REAL_PREFLIGHT_PASS ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
