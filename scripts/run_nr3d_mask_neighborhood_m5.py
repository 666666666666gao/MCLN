"""Matched short training of existing Mask projections with two groupers."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    addon = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(addon / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(addon / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_mask_neighborhood_probe import NearestTwoGroup
    from scripts.run_nr3d_view_pair_role import read_train_rows

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    partitions = {'fit': [], 'holdout': []}
    for i, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        partitions['holdout' if fold == 0 else 'fit'].append(i)
    assert len(partitions['fit']) == 26747 and len(partitions['holdout']) == 6172
    partitions['fit'] = partitions['fit'][:2048]
    assert partitions == manifest['row_ids']
    scenes = {name: {raw_rows[i]['scan_id'] for i in ids} for name, ids in partitions.items()}
    assert not scenes['fit'].intersection(scenes['holdout'])
    assert len(scenes['holdout']) == 98
    selected_ids = sorted(partitions['fit'] + partitions['holdout'])

    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert checkpoint['evaluation_only'] and 'optimizer' not in checkpoint
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint['config']))
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    assert args.mask_loss_scale == args.consistency_loss_scale == 1 and args.clip_norm > 0
    initial_state = {name[7:]: value for name, value in checkpoint['model'].items()}
    del checkpoint
    models, optimizers, trainable = {}, {}, {}
    prefixes = ('x_query.', 'x_mask.', 'rel_encoder.')
    for arm in ['native', 'nearest']:
        model = TrainTester.get_model(args).cuda().eval()
        model.load_state_dict(initial_state, strict=True)
        assert model.decoder_query_adapter is None
        assert model.super_grouper.radius == .2 and model.super_grouper.nsample == 2
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert len(parameters) == 16
        if arm == 'nearest':
            model.super_grouper = NearestTwoGroup()
        models[arm] = model
        trainable[arm] = parameters
        optimizers[arm] = torch.optim.AdamW(parameters.values(), lr=1e-5, weight_decay=args.weight_decay)
    assert list(trainable['native']) == list(trainable['nearest'])
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)

    def verify_state(before_training=False):
        changed = {}
        for arm, model in models.items():
            changed[arm] = []
            for name, value in model.state_dict().items():
                same = torch.equal(value.detach().cpu(), initial_state[name])
                if before_training or name not in trainable[arm]:
                    assert same, (arm, name)
                elif not same:
                    changed[arm].append(name)
            assert all(parameter.grad is None for name, parameter in model.named_parameters()
                       if name not in trainable[arm])
        return changed

    class SelectedDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in selected_ids]
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['m5_row_id'] = np.int64(self.m5_ids[index])
            return item

    base = SelectedDataset(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                           data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
                           detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                           skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(base.annos) == len(selected_ids)
    by_id = {}
    for row_id, anno in zip(selected_ids, base.annos):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id'] and anno['target_id'] == int(raw_rows[row_id]['target_id'])
        by_id[row_id] = anno
    datasets = {}
    for name, ids in partitions.items():
        dataset = copy.copy(base)
        dataset.annos = [by_id[i] for i in ids]
        dataset.m5_ids = ids
        dataset.augment = False
        datasets[name] = dataset

    def loader(name, seed, batch_size, shuffle):
        return torch.utils.data.DataLoader(datasets[name], batch_size=batch_size, shuffle=shuffle,
                 num_workers=0, generator=torch.Generator().manual_seed(seed))

    def inputs_for(raw):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        return inputs, batch

    def grounding_snapshot(outputs):
        return {key: outputs[key].detach().clone() for key in ['last_center', 'last_pred_size', 'selected_source_scores']}

    def require_same_grounding(first, second):
        for key in first:
            assert torch.equal(first[key], second[key]), key

    started = time.time()
    preflight_inputs, preflight_batch = inputs_for(next(iter(loader('fit', 0, 4, False))))
    preflight = {}
    first_grounding = None
    for arm, model in models.items():
        outputs = model(preflight_inputs)
        outputs.update(preflight_batch)
        loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        snapshot = grounding_snapshot(outputs)
        if first_grounding is None:
            first_grounding = snapshot
        else:
            require_same_grounding(first_grounding, snapshot)
        loss.backward()
        norms = {}
        for name, parameter in trainable[arm].items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            norms[name] = float(parameter.grad.norm())
        assert all(value > 0 for value in norms.values())
        optimizers[arm].zero_grad(set_to_none=True)
        preflight[arm] = {'loss': float(loss), 'gradient_norms': norms}
    verify_state(before_training=True)
    preflight.update(optimizer_steps=0, trainable_parameters=sum(p.numel() for p in trainable['native'].values()),
                     learning_rate=1e-5, weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                     fit_rows=2048, fit_scenes=len(scenes['fit']), holdout_rows=6172, holdout_scenes=98,
                     identical_grounding=True, frozen_state_unchanged=True)
    write_json(addon / 'preflight.json', preflight)
    print('M5 PREFLIGHT', json.dumps(preflight), flush=True)
    del outputs, loss, preflight_inputs, preflight_batch, first_grounding, snapshot

    def evaluate(stage):
        seed_everything(1000)
        records = []
        with torch.no_grad():
            for index, raw in enumerate(loader('holdout', 1000, 16, False)):
                inputs, batch = inputs_for(raw)
                observations, snapshots = {}, {}
                for arm, model in models.items():
                    outputs = model(inputs)
                    outputs.update(batch)
                    observations[arm] = diagnose_root_candidates(outputs, evaluator)
                    snapshots[arm] = grounding_snapshot(outputs)
                require_same_grounding(snapshots['native'], snapshots['nearest'])
                for bid, row_id in enumerate(batch['m5_row_id'].tolist()):
                    digest = hashlib.sha256()
                    for tensor in snapshots['native'].values():
                        digest.update(tensor[bid].cpu().numpy().tobytes())
                    record = {'id': row_id, 'scan_id': raw_rows[row_id]['scan_id'], 'grounding_sha256': digest.hexdigest(),
                              'input_point_sha256': hashlib.sha256(inputs['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()}
                    for arm in models:
                        observed = observations[arm][bid]
                        rec = observed['rec_selection']
                        record[arm] = {'rec_query': None if rec is None else rec['query'],
                                       'rec_box_iou': None if rec is None else rec['box_iou'],
                                       'mask_query': observed['mask_selection']['query'],
                                       'mask_iou': observed['mask_selection']['mask_iou']}
                    records.append(record)
                if (index + 1) % 50 == 0:
                    print('M5 EVAL', json.dumps({'stage': stage, 'batches': index + 1, 'rows': len(records),
                          'elapsed_seconds': time.time() - started}), flush=True)
        assert [row['id'] for row in records] == partitions['holdout']
        write_json(addon / (stage + '_rows.json'), records)
        return records

    baseline = evaluate('baseline')
    seen_ids = []
    step = 0
    for epoch in range(2):
        seed_everything(epoch)
        for raw in loader('fit', epoch, 4, True):
            inputs, batch = inputs_for(raw)
            snapshots, statistics = {}, {}
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                outputs.update(batch)
                loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
                assert torch.isfinite(loss)
                snapshots[arm] = grounding_snapshot(outputs)
                loss.backward()
                for name, parameter in trainable[arm].items():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (arm, name)
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable[arm].values(), args.clip_norm)
                assert torch.isfinite(gradient_norm)
                optimizer.step()
                statistics[arm] = {'loss': float(loss), 'gradient_norm_before_clip': float(gradient_norm)}
            require_same_grounding(snapshots['native'], snapshots['nearest'])
            step += 1
            seen_ids.extend(batch['m5_row_id'].tolist())
            if step % 64 == 0:
                print('M5 TRAIN', json.dumps({'step': step, 'epoch': epoch, 'arms': statistics,
                      'elapsed_seconds': time.time() - started}), flush=True)
    assert step == 1024
    for epoch in range(2):
        assert sorted(seen_ids[epoch * 2048:(epoch + 1) * 2048]) == partitions['fit']
    changed = verify_state()
    assert all(len(names) == 16 for names in changed.values())
    terminal = evaluate('terminal')
    for original, updated in zip(baseline, terminal):
        for key in ['id', 'scan_id', 'grounding_sha256', 'input_point_sha256']:
            assert original[key] == updated[key], (original['id'], key)
        for arm in models:
            for key in ['rec_query', 'rec_box_iou', 'mask_query']:
                assert original[arm][key] == updated[arm][key], (original['id'], arm, key)
    changed = verify_state()
    artifacts = {}
    for arm in models:
        optimizer = optimizers[arm]
        assert len(optimizer.state) == 16
        assert all(float(value['step']) == 1024 for value in optimizer.state.values())
        assert all(torch.isfinite(value[key]).all() for value in optimizer.state.values() for key in ['exp_avg', 'exp_avg_sq'])
        path = addon / (arm + '_mask_projection_state.pt')
        assert not path.exists()
        torch.save({'mask_projection_state': {name: parameter.detach().cpu() for name, parameter in trainable[arm].items()},
                    'optimizer': optimizer.state_dict(), 'steps': step, 'arm': arm,
                    'parent_checkpoint_sha256': manifest['checkpoint_sha256']}, path)
        artifacts[arm] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    verify_inputs()
    receipt = {'schema': 'mcln-nr3d-mask-neighborhood-m5-v1', 'status': 'complete', 'optimizer_steps_per_arm': step,
               'fit_rows': 2048, 'epochs': 2, 'holdout_rows': 6172, 'formal_rows': 0, 'formal_promotion': False,
               'fit_order_sha256': hashlib.sha256(json.dumps(seen_ids).encode()).hexdigest(), 'fit_order_ids': seen_ids,
               'frozen_parameters_and_buffers_unchanged': True, 'source_data_and_parent_checkpoint_unchanged': True,
               'grounding_and_query_selection_exactly_equal_to_start': True, 'changed_parameter_names': changed,
               'baseline_rows_sha256': file_sha(addon / 'baseline_rows.json'), 'terminal_rows_sha256': file_sha(addon / 'terminal_rows.json'),
               'artifacts': artifacts, 'manifest_sha256': file_sha(options.manifest), 'elapsed_seconds': time.time() - started,
               'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(addon / 'receipt.json', receipt)
    print('M5 COMPLETE', json.dumps({key: value for key, value in receipt.items() if key != 'fit_order_ids'}), flush=True)


if __name__ == '__main__':
    main()
