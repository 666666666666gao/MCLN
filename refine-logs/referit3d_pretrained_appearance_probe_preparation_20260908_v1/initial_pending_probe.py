"""Qualified real Nr/Sr cache/model probe; disposable steps, no saved weights."""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    manifest_path = p.parse_args().manifest.resolve()
    root = manifest_path.parent
    plan = json.loads(manifest_path.read_text())
    source = Path(plan['model_source'])
    assert sha(source / 'referit_source_manifest.json') == plan['source_manifest_sha256']
    for name, digest in json.loads((source / 'referit_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == digest, name
    for name, digest in plan['files'].items():
        assert sha(root / name) == digest, name
    cache = Path(plan['cache_root'])
    assert sha(cache / 'receipt.json') == plan['cache_receipt_sha256']
    cached = json.loads((cache / 'receipt.json').read_text())
    assert cached['status'] == 'complete' and cached['scene_count'] == 1200
    for name, digest in cached['qualification']['files'].items():
        assert sha(name) == digest, name
    qualification = json.loads(Path(cached['qualification']['decision']).read_text())
    assert qualification['status'] == 'formal_evaluated_and_audited'
    assert qualification['formal_rows'] == 9508
    assert qualification['promotion']['advance_to_nr3d_sr3d_rec']
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    assert sha(Path(plan['data_root']) / 'train_v3scans.pkl') == plan['train_pickle_sha256']
    annotation = json.loads((root / 'annotation_receipt.json').read_text())
    for path, info in annotation['annotations_and_split_files'].items():
        current = Path(path) if '/DATA_ROOT/' in path else source / 'data/meta_data' / Path(path).name
        assert sha(current) == info['sha256'], str(current)
    for name, digest in plan['superpoints'].items():
        assert sha(Path(plan['data_root']) / 'superpoints/train' / name) == digest
    os.chdir(str(source))
    sys.path[:0] = [str(root), str(source)]
    import numpy as np
    import torch
    from main_utils import parse_option, prepare_source_moe_gate_checkpoint_config
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.referit_object_appearance import ReferItObjectAppearanceDataset
    from models.pretrained_object_appearance import PretrainedObjectAppearance
    from fixed_selection import build_probe_dataset
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    contract = json.loads((root / 'nr_contract.json').read_text())
    selection = json.loads((root / 'preflight_rows.json').read_text())
    assert sha(root / 'preflight_rows.json') == annotation['preflight_rows_sha256']
    checkpoint = torch.load(plan['checkpoint'], map_location='cpu')
    assert all(key.startswith('module.') for key in checkpoint['model'])
    initial = {key[7:]: value for key, value in checkpoint['model'].items()}
    assert len(initial) == 1144
    del checkpoint

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def equal_outputs(left, right):
        for key in ['last_center', 'last_pred_size', 'last_sem_cls_scores', 'last_proj_queries',
                    'selected_source_scores', 'sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']:
            if torch.is_tensor(left[key]):
                assert torch.equal(left[key], right[key]), key
            else:
                assert len(left[key]) == len(right[key])
                assert all(torch.equal(a, b) for a, b in zip(left[key], right[key])), key

    results = {}
    started = time.time()
    for dset in ['nr3d', 'sr3d']:
        argv = list(contract['eval_argv'])
        for key, value in [('--dataset', dset), ('--test_dataset', dset),
                           ('--data_root', plan['data_root']), ('--checkpoint_path', plan['checkpoint']),
                           ('--expected_eval_sample_count', str(annotation['protocols'][dset]['val']['total_rows']))]:
            argv[argv.index(key) + 1] = value
        sys.argv = ['referit-appearance-probe'] + argv
        args = prepare_source_moe_gate_checkpoint_config(parse_option())
        assert args.butd_cls and not args.butd and not args.butd_gt
        assert args.joint_det and args.detect_intermediate and args.use_color
        assert not args.use_height and not args.use_multiview
        assert args.use_source_choice_selector and args.eval_use_selector_choice_scores
        assert not args.eval_use_rec_reranker_scores and not args.eval_use_rec_geometry_reranker_scores
        seed(0)
        native = build_probe_dataset(Joint3DDataset, args, annotation, selection[dset], dset)
        original_annotations = copy.deepcopy(native.annos)
        dataset = ReferItObjectAppearanceDataset(native, cache, plan['cache_receipt_sha256'])
        model = TrainTester.get_model(args).cuda().eval().requires_grad_(False)
        assert model.object_appearance is None and model.decoder[-1].local_visual is None
        model.load_state_dict(initial, strict=True)
        fusion = PretrainedObjectAppearance().cuda()
        criterion, set_criterion = TrainTester.get_criterion(args)
        records = []
        forwards = 0
        torch.cuda.reset_peak_memory_stats()

        def loader():
            return torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0,
                                               generator=torch.Generator().manual_seed(0))

        native.augment = False
        seed(1000)
        for index, raw in enumerate(loader()):
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            assert inputs['det_visual_features'].shape == (len(raw['scan_ids']), 132, 1280)
            assert inputs['det_visual_available'].any()
            model.object_appearance = None
            with torch.no_grad():
                reference = model(inputs)
                model.object_appearance = fusion
                candidate = model(inputs)
                equal_outputs(reference, candidate)
            forwards += 2
            records.append(dict(phase='zero_init', batch=index, rows=len(raw['scan_ids']),
                available_slots=int(inputs['det_visual_available'].sum()), native_output_parity=True))
            del reference, candidate, inputs, batch, raw
        assert all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in initial.items())
        # Same limited trainable scope as the Scan bridge; native input augmentation remains on.
        core = []
        for name, value in model.named_parameters():
            if name.startswith(('cross_encoder.', 'decoder.', 'prediction_heads.')):
                value.requires_grad_(True)
                core.append((name, value))
        assert core and model.object_appearance is fusion
        optimizer = torch.optim.AdamW([
            dict(params=[v for _, v in core], lr=1e-6),
            dict(params=list(fusion.parameters()), lr=1e-4)], weight_decay=.0005)
        allowed = {name for name, _ in core} | {'object_appearance.projection.weight'}
        native.annos = copy.deepcopy(original_annotations)
        native.augment = True
        seed(2000)
        for index, raw in enumerate(loader()):
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            assert not model.training and not model.backbone_net.training
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            forwards += 1
            assert not set(outputs).intersection(batch)
            outputs.update(batch)
            loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            gradient = fusion.projection.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
            grads = {name:float(v.grad.norm()) for name,v in core if v.grad is not None}
            assert grads and all(np.isfinite(value) for value in grads.values())
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .1)
            assert torch.isfinite(norm)
            optimizer.step()
            records.append(dict(phase='disposable_update', batch=index, rows=len(raw['scan_ids']),
                native_gt_loss=float(loss.detach()), appearance_gradient_norm=float(gradient.norm()),
                core_gradient_tensors=len(grads), total_gradient_norm_before_clip=float(norm),
                native_augmentation=True, frozen_bn_dropout=True,
                annotation_datasets=raw['sample_dataset'],
                point_sha256=[hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in raw['point_clouds']]))
            print('REFERIT APPEARANCE PROBE', dset, index + 1, json.dumps(records[-1]), flush=True)
            del loss, outputs, inputs, batch, raw, gradient
        assert index == 1 and forwards == 6
        current = model.state_dict()
        for name, value in initial.items():
            if name not in allowed:
                assert torch.equal(current[name].cpu(), value), name
        assert fusion.projection.weight.detach().abs().sum() > 0
        results[dset] = dict(rows=16, forwards=forwards, disposable_optimizer_steps=2,
            checkpoint_initialization='protected Nr3D averaged E57; Sr3D probe also uses this initialization',
            records=records, frozen_old_state_equal=True,
            changed_allowed_tensors=[name for name in allowed if name not in initial or not torch.equal(current[name].cpu(), initial[name])],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        del model, fusion, optimizer, criterion, set_criterion, native, dataset, current, core
        gc.collect()
        torch.cuda.empty_cache()
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    receipt = dict(status='pass', schema='referit-pretrained-appearance-probe-v1', protocols=results,
        manifest_sha256=sha(manifest_path), cache_receipt_sha256=plan['cache_receipt_sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'], elapsed_seconds=time.time()-started,
        time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        model_forwards=12, disposable_optimizer_steps=4, checkpoint_writes=0, formal_rows=0,
        limits='Real input/model engineering check only; no full Nr/Sr training or new benchmark result.')
    with (root / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    print('REFERIT APPEARANCE PROBE COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
