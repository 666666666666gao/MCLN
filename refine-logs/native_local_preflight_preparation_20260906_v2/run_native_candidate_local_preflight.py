"""Run a bounded native Nr/Sr GPU probe after ScanRefer formal promotion."""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_probe_data_inputs(root):
    data = json.loads((root / 'data_inputs.json').read_text())
    data_root = data['data_root'].rstrip('/') + '/'
    for split, count in [('train', 1201), ('val', 312)]:
        files = data['superpoint_files'][split]
        assert len(files) == count
        for name, digest in files.items():
            assert file_sha(Path(data_root) / 'superpoints' / split / name) == digest, (split, name)
    for name, original in data['non_superpoint_entries_share_original_inode'].items():
        assert os.path.samefile(str(Path(data_root) / name), original), name
    return data_root


def build_probe_argv(eval_argv, dset, val_rows, data_root):
    argv = list(eval_argv)
    for key, value in [('--dataset', dset), ('--test_dataset', dset),
                       ('--expected_eval_sample_count', str(val_rows)), ('--data_root', data_root)]:
        argv[argv.index(key) + 1] = value
    return argv


def build_probe_dataset(dataset_class, args, annotation, rows, dset):
    language = [row for row in rows if row['dataset'] == dset]
    detection = [row for row in rows if row['dataset'] == 'scannet']
    assert len(language) == 12 and len(detection) == 4
    assert len({r['scan_id'] for r in language}) == 12

    class ProbeDataset(dataset_class):
        def _scene_graph_parse(self, annos):
            assert len(annos) == annotation['protocols'][dset]['train']['language_rows']
            chosen = []
            for item in language:
                row = annos[item['raw_language_row_id']]
                for key in ['dataset', 'scan_id', 'target_id', 'target', 'utterance', 'anchor_ids']:
                    assert row[key] == item[key], (dset, key)
                chosen.append(row)
            annos[:] = chosen
            super()._scene_graph_parse(annos)

        def load_scannet_annos(self):
            annos = super().load_scannet_annos()
            assert len(annos) == 1199
            chosen = [annos[item['raw_detection_row_id']] for item in detection]
            assert [r['scan_id'] for r in chosen] == [r['scan_id'] for r in detection]
            return chosen

    dataset = ProbeDataset(dataset_dict={dset: 1, 'scannet': 10}, test_dataset=dset,
        split='train', data_path=args.data_root, use_color=args.use_color,
        use_height=args.use_height, use_multiview=args.use_multiview,
        detect_intermediate=args.detect_intermediate, butd=args.butd, butd_gt=args.butd_gt,
        butd_cls=args.butd_cls, augment_det=args.augment_det,
        skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(dataset) == 52  # twelve language rows plus four detection rows repeated ten times
    dataset.annos = dataset.annos[:16]
    assert [r['scan_id'] for r in dataset.annos] == [r['scan_id'] for r in rows]
    assert dataset.joint_det and dataset.augment and not dataset.use_sacr_source
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    root = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    source = Path(manifest['model_source'])
    assert file_sha(source / 'native_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'native_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(root / name) == digest, name
    data_root = verify_probe_data_inputs(root)
    assert data_root == manifest['data_root']
    assert file_sha(manifest['checkpoint']) == manifest['checkpoint_sha256']
    assert file_sha(manifest['scan_formal_receipt']) == manifest['scan_formal_receipt_sha256']
    formal = json.loads(Path(manifest['scan_formal_receipt']).read_text())
    assert formal['schema'] == 'mcln-scanrefer-local-visual-official-v2'
    assert formal['status'] == 'complete' and formal['data_root'] == data_root
    assert formal['promotion']['advance_to_nr3d_sr3d_rec']
    assert manifest['rows_per_dataset'] == 16 and manifest['optimizer_steps_per_dataset'] == 2
    assert manifest['core_learning_rate'] == 1e-6 and manifest['local_learning_rate'] == 1e-4
    selection = json.loads((root / 'preflight_rows.json').read_text())
    annotation = json.loads((root / 'annotation_receipt.json').read_text())
    assert annotation['preflight_rows_sha256'] == file_sha(root / 'preflight_rows.json')
    assert annotation['source_manifest_sha256'] == manifest['source_manifest_sha256']
    for path, item in annotation['annotations_and_split_files'].items():
        assert file_sha(path) == item['sha256'], path
    for split in ['train', 'val']:
        path = Path('/root/autodl-tmp/DATA_ROOT') / (split + '_v3scans.pkl')
        identity = annotation['scans'][split]['pickle_metadata']
        assert path.stat().st_size == identity['bytes'] and path.stat().st_mtime_ns == identity['mtime_ns']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import src.grounding_evaluator as grounding
    from main_utils import BaseTrainTester, load_checkpoint, parse_option, prepare_source_moe_gate_checkpoint_config
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset

    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    output_keys = ['last_center', 'last_pred_size', 'last_sem_cls_scores',
                   'selected_source_scores', 'sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']

    def equal_outputs(reference, candidate):
        for key in output_keys:
            left, right = reference[key], candidate[key]
            if torch.is_tensor(left):
                assert torch.equal(left, right), key
            else:
                assert len(left) == len(right), key
                assert all(torch.equal(a, b) for a, b in zip(left, right)), key

    tester = object.__new__(TrainTester)
    tester.logger = logging.getLogger('native-local-preflight')
    contract = json.loads((root / 'nr_contract.json').read_text())
    results = {}
    started = time.time()
    for dset in ['nr3d', 'sr3d']:
        rows = selection[dset]
        argv = build_probe_argv(contract['eval_argv'], dset,
                               annotation['protocols'][dset]['val']['total_rows'], data_root)
        sys.argv = ['native-local-preflight'] + argv + ['--use_candidate_local_visual']
        args = prepare_source_moe_gate_checkpoint_config(parse_option())
        assert args.data_root == data_root
        args.eval = False
        args.checkpoint_path = manifest['checkpoint']
        args.model_only_initialization = True
        args.checkpoint_start_epoch = 1
        args.max_epoch = 1
        args.lr = args.lr_backbone = args.source_choice_selector_lr = manifest['core_learning_rate']
        args.candidate_local_visual_lr = manifest['local_learning_rate']
        args.checkpoint_metric_retention = True
        args.checkpoint_retention_metrics = ['rec_acc025', 'rec_acc050']
        assert args.butd_cls and not args.butd and not args.butd_gt
        assert args.joint_det and args.detect_intermediate and args.use_color
        assert not args.use_height and not args.use_multiview
        assert args.use_source_choice_selector and args.eval_use_selector_choice_scores
        assert not args.eval_use_rec_reranker_scores and not args.eval_use_rec_geometry_reranker_scores

        seed(0)
        dataset = build_probe_dataset(Joint3DDataset, args, annotation, rows, dset)
        seed(0)
        model = TrainTester.get_model(args).cuda()
        wrapped = torch.nn.Module()
        wrapped.module = model
        optimizer = BaseTrainTester.get_optimizer(args, wrapped)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=.1)
        load_checkpoint(args, wrapped, optimizer, scheduler)
        assert args.start_epoch == 1 and not optimizer.state
        reference = copy.deepcopy(model)
        reference.decoder[-1].local_visual = None
        criterion, set_criterion = TrainTester.get_criterion(args)
        initial_local = {name: value.detach().cpu().clone()
                         for name, value in model.decoder[-1].local_visual.state_dict().items()}
        initial_text = {name: value.detach().cpu().clone() for name, value in model.text_encoder.state_dict().items()}
        records = []
        forwards = 0
        torch.cuda.reset_peak_memory_stats()

        def loader():
            return torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0,
                                                generator=torch.Generator().manual_seed(0))

        def native_decisions(outputs, batch):
            evaluator = tester._build_grounding_evaluator(args, ['last_'])
            assert evaluator.filter_non_gt_boxes and evaluator.eval_use_selector_choice_scores
            decisions = []
            original_filter = grounding.build_detector_overlap_valid
            original_top = evaluator._position_top_indices

            def record_filter(*values, **keywords):
                valid = original_filter(*values, **keywords)
                assert valid.shape == (1, 256)
                decisions.append({'valid': valid[0].cpu().tolist(), 'selected_query': None})
                return valid

            def record_top(scores, valid, axis_mode, count):
                assert axis_mode == 'default_query_axis'
                bid = len(decisions) - 1
                assert torch.equal(scores[0], outputs['selected_source_scores'][bid])
                selected = original_top(scores, valid, axis_mode, count)
                decisions[bid]['selected_query'] = int(selected[0, 0])
                return selected

            grounding.build_detector_overlap_valid = record_filter
            evaluator._position_top_indices = record_top
            view = dict(outputs)
            assert not set(view).intersection(batch)
            view.update(batch)
            evaluator.evaluate_bbox_by_pos_align(view, 'last_')
            grounding.build_detector_overlap_valid = original_filter
            assert len(decisions) == 12
            assert all(evaluator.gts[('last_', threshold, 1, 'bbs')] == 12 for threshold in [.25, .5])
            return decisions

        # Real points, no augmentation: before any optimizer update, both model paths must agree.
        dataset.augment = False
        seed(1000)
        model.eval()
        reference.eval()
        with torch.no_grad():
            for batch_index, raw in enumerate(loader()):
                batch = TrainTester._to_gpu(raw)
                inputs = TrainTester._get_inputs(batch)
                before = reference(inputs)
                after = model(inputs)
                forwards += 2
                equal_outputs(before, after)
                record = {'phase': 'eval', 'batch': batch_index, 'rows': len(raw['scan_ids']),
                          'zero_output_parity': True,
                          'point_sha256': [hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in inputs['point_clouds']]}
                if batch_index == 0:
                    reference_decisions = native_decisions(before, batch)
                    decisions = native_decisions(after, batch)
                    assert reference_decisions == decisions
                    record['native_filter_and_selector_parity'] = True
                    record['decisions'] = decisions
                records.append(record)
                del before, after
        # The original trainer uses model.train(), native augmentations and the full optimizer.
        dataset.augment = True
        seed(2000)
        TrainTester._set_source_moe_train_mode(wrapped, args)
        reference.train()
        assert model.training and model.backbone_net.training and model.decoder[-1].training
        assert model.decoder[-1].local_visual.training
        for batch_index, raw in enumerate(loader()):
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            optimizer.zero_grad()
            seed(3000 + batch_index)
            if batch_index == 0:
                with torch.no_grad():
                    before = reference(inputs)
                forwards += 1
                seed(3000 + batch_index)
            after = model(inputs)
            forwards += 1
            if batch_index == 0:
                equal_outputs(before, after)
                del before
            assert not set(after).intersection(batch)
            after.update(batch)
            loss, after = TrainTester._compute_loss(after, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            reader = model.decoder[-1].local_visual
            norms = {name: float(p.grad.norm()) for name, p in reader.named_parameters() if p.grad is not None}
            assert norms['output_projection.weight'] > 0
            assert all(math.isfinite(value) for value in norms.values())
            if batch_index == 1:
                for name in ['point_encoder.0.weight', 'query_projection.weight',
                             'key_projection.weight', 'value_projection.weight']:
                    assert norms[name] > 0, name
            trainable = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, .1)
            assert torch.isfinite(grad_norm)
            optimizer.step()
            records.append({'phase': 'train', 'batch': batch_index, 'rows': len(raw['scan_ids']),
                            'native_train_mode': True, 'dataset_augmentation_enabled': dataset.augment,
                            'zero_output_parity_before_first_update': batch_index == 0,
                            'loss': float(loss.detach()), 'total_gradient_norm': float(grad_norm),
                            'local_gradient_norms': norms,
                            'point_sha256': [hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in inputs['point_clouds']]})
            print('NATIVE LOCAL PREFLIGHT', json.dumps({'dataset': dset, 'step': batch_index + 1,
                  'rows': len(raw['scan_ids']), 'loss': float(loss.detach())}), flush=True)
            del after, loss
        assert batch_index == 1 and forwards == 7
        assert all(torch.equal(model.text_encoder.state_dict()[name].cpu(), value) for name, value in initial_text.items())
        changed = [name for name, value in model.decoder[-1].local_visual.state_dict().items()
                   if not torch.equal(value.cpu(), initial_local[name])]
        assert 'output_projection.weight' in changed
        results[dset] = {'rows': 16, 'point_samples_constructed': 32, 'forwards': forwards,
                         'disposable_optimizer_steps': 2, 'frozen_text_state_equal': True,
                         'changed_local_tensors': changed, 'records': records,
                         'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'optimizer_groups': [{'name': g['name'], 'lr': g['lr'],
                            'parameter_tensors': len(g['params'])} for g in optimizer.param_groups]}
        del model, reference, wrapped, optimizer, scheduler, dataset, initial_text, initial_local
        del reader, trainable, batch, raw, inputs, criterion, set_criterion
        gc.collect()
        torch.cuda.empty_cache()
    result = {'schema': 'mcln-native-candidate-local-gpu-preflight-v2', 'status': 'pass',
              'data_root': data_root, 'data_inputs_sha256': file_sha(root / 'data_inputs.json'),
              'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'elapsed_seconds': time.time() - started, 'manifest_sha256': file_sha(option.manifest),
              'protocols': results, 'gpu_forwards': 14, 'disposable_optimizer_steps': 4,
              'point_samples_constructed': 64, 'formal_rows': 0, 'checkpoint_writes': 0,
              'limits': 'Bounded actual native input/mode/gradient check, not Nr/Sr training or benchmark performance; formal training must reload its registered pretrained input.'}
    with (root / 'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print('NATIVE LOCAL PREFLIGHT COMPLETE', json.dumps({k: result[k] for k in [
        'status', 'time_cst', 'elapsed_seconds', 'gpu_forwards', 'disposable_optimizer_steps', 'checkpoint_writes']}), flush=True)


if __name__ == '__main__':
    main()
