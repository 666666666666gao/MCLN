"""Check the joint REC readout on actual ScanRefer training inputs, with zero updates."""

import argparse
import copy
import datetime
import hashlib
import json
import logging
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


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    source = Path(manifest['model_source'])
    assert file_sha(source / 'g0_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert file_sha(item['path']) == item['sha256'], name
    cpu = json.loads(Path(manifest['cpu_receipt']).read_text())
    assert file_sha(manifest['cpu_receipt']) == manifest['cpu_receipt_sha256']
    assert cpu['status'] == 'pass' and cpu['source_sha256'] == manifest['files']['scripts/scanrefer_joint_readout.py']
    os.chdir(str(source))
    sys.path.insert(0, str(source))

    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester, build_rec_reranker_outputs, build_rec_geometry_runtime_outputs
    from src.joint_det_dataset import Joint3DDataset
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_joint_readout import JointRecReadout, joint_rec_readout_loss
    from scripts.scanrefer_rec_evaluation import rec_evaluation_view

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    command = build_authoritative_command(directory / 'unused_official_output')
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
    assert args.butd and not args.butd_cls and not args.butd_gt
    assert args.use_color and not args.use_height and not args.use_multiview
    assert args.batch_size == 12 and args.num_decoder_layers == 6
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    backbone_payload = torch.load(args.checkpoint_path, map_location='cpu')
    initial = {name[7:]: value for name, value in backbone_payload['model'].items()}
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict(initial, strict=True)
    prefixes = ('decoder.5.', 'prediction_heads.5.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    core_parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    artifacts = {name: torch.load(item['path'], map_location='cpu')
                 for name, item in manifest['artifacts'].items() if name != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval()
    frozen_readout = copy.deepcopy(readout).eval().requires_grad_(False)
    readout_initial = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}
    criterion, set_criterion = TrainTester.get_criterion(args)

    protocol = {}
    class ProbeDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            partitions = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                space = row['scan_id'].split('_')[0]
                code = (manifest['split_salt'] + '\0' + space).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                partitions['holdout' if fold == 0 else 'fit'].append(index)
                row['_joint_probe_row_id'] = index
            selected_ids, selected_scenes = [], set()
            for index in partitions['fit']:
                scan = annos[index]['scan_id']
                if scan not in selected_scenes:
                    selected_ids.append(index)
                    selected_scenes.add(scan)
                if len(selected_ids) == 16:
                    break
            assert len(selected_ids) == 16
            spaces = {name: {annos[index]['scan_id'].split('_')[0] for index in ids}
                      for name, ids in partitions.items()}
            assert not spaces['fit'].intersection(spaces['holdout'])
            protocol.update(row_ids=partitions, selected_ids=selected_ids,
                selected_scenes=sorted(selected_scenes),
                rows={name: len(ids) for name, ids in partitions.items()},
                physical_spaces={name: len(ids) for name, ids in spaces.items()},
                physical_space_overlap=0, previous_pretraining_has_seen_development_holdout=True)
            # Keep every annotation in each selected scene for the original distractor construction.
            annos[:] = [row for row in annos if row['scan_id'] in selected_scenes]
            super()._scene_graph_parse(annos)

    dataset = ProbeDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    by_id = {row['_joint_probe_row_id']: row for row in dataset.annos}
    dataset.annos = [by_id[index] for index in protocol['selected_ids']]
    dataset.augment = False
    write_json(directory / 'protocol.json', protocol)
    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0,
                                         generator=torch.Generator().manual_seed(0))
    native_tester = object.__new__(TrainTester)
    native_tester.logger = logging.getLogger('scanrefer-joint-probe')
    evaluations = [native_tester._build_grounding_evaluator(args, ['last_']) for _ in range(2)]

    def same_runtime(first, second):
        return all(torch.equal(value, second[key]) if torch.is_tensor(value) else value == second[key]
                   for key, value in first.items())

    def gradients(loss, parameters, retain_graph):
        values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=retain_graph, allow_unused=True)
        records = {}
        for name, value in zip(parameters, values):
            if value is None:
                records[name] = None
            else:
                assert torch.isfinite(value).all(), name
                records[name] = float(value.norm())
        return records

    observations = []
    started = time.time()
    for index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        torch.cuda.synchronize()
        before_forward = time.time()
        outputs = model(inputs)
        raw_sizes = outputs['last_pred_size'].detach().clone()
        size_diagnostic = {'batch': index, 'minimum_raw_size': float(raw_sizes.min()),
            'negative_size_entries': int((raw_sizes < 0).sum()),
            'negative_size_queries_per_row': (raw_sizes < 0).any(dim=-1).sum(dim=-1).tolist()}
        write_json(directory / ('raw_size_batch_{}.json'.format(index)), size_diagnostic)
        torch.cuda.synchronize()
        native_forward_seconds = time.time() - before_forward
        reference_parent = build_rec_reranker_outputs(
            outputs, inputs, frozen_readout.scorers['parent'], frozen_readout.metadata['parent'])
        reference = build_rec_geometry_runtime_outputs(
            outputs, inputs, reference_parent, frozen_readout.scorers['geometry'], frozen_readout.metadata['geometry'],
            hierarchical_model=frozen_readout.scorers['v99'], hierarchical_artifact=frozen_readout.metadata['v99'])
        joint = readout(outputs, inputs)
        detached = readout(outputs, inputs, detach_visual=True)
        assert same_runtime(joint['runtime'], reference)
        assert same_runtime(detached['runtime'], reference)
        root_boxes = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
        root_valid = batch['box_label_mask'][:, :1].bool()
        joint_loss, joint_stats = joint_rec_readout_loss(joint, root_boxes, root_valid)
        detached_loss, detached_stats = joint_rec_readout_loss(detached, root_boxes, root_valid)
        assert joint_loss.item() == detached_loss.item() and joint_stats == detached_stats
        detached_core_grads = gradients(detached_loss, core_parameters, True)
        assert all(value is None for value in detached_core_grads.values())
        readout_grads = gradients(joint_loss, dict(readout.named_parameters()), True)
        joint_core_grads = gradients(joint_loss, core_parameters, True)
        assert any(value is not None and value > 0 for value in joint_core_grads.values())
        assert all(value is not None for value in readout_grads.values())
        outputs.update(batch)
        native_loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(native_loss)
        assert torch.equal(outputs['last_pred_size'].detach(), raw_sizes)
        native_core_grads = gradients(native_loss, core_parameters, False)
        for evaluator, attachment in zip(evaluations, [reference, joint['runtime']]):
            evaluated = rec_evaluation_view(outputs)
            evaluated.update(attachment)
            evaluator.evaluate(evaluated, 'last_')
        points = inputs['point_clouds'].detach().cpu().contiguous().numpy().tobytes()
        observations.append({'batch': index, 'rows': len(batch['scan_ids']),
            'point_sha256': hashlib.sha256(points).hexdigest(),
            'forward_parity': True, 'native_loss': float(native_loss), 'readout_loss': float(joint_loss),
            'readout_stats': joint_stats, 'native_forward_seconds': native_forward_seconds,
            'size_diagnostic': size_diagnostic, 'raw_sizes_unchanged_by_loss': True,
            'detached_core_gradients': detached_core_grads,
            'readout_parameter_gradients': readout_grads,
            'joint_readout_core_gradients': joint_core_grads, 'native_core_gradients': native_core_grads})
        print('SCANREFER JOINT NATIVE PROBE', json.dumps({key: observations[-1][key]
              for key in ['batch', 'rows', 'forward_parity', 'native_loss', 'readout_loss', 'native_forward_seconds']}), flush=True)
        del outputs, joint, detached, native_loss, joint_loss, detached_loss, evaluated, inputs, batch

    assert evaluations[0].dets == evaluations[1].dets and evaluations[0].gts == evaluations[1].gts
    assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value.detach().cpu(), readout_initial[name]) for name, value in readout.state_dict().items())
    receipt = {'schema': 'mcln-scanrefer-joint-readout-native-probe-v1', 'status': 'pass',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'manifest_sha256': file_sha(option.manifest), 'real_train_rows': 16, 'formal_rows': 0,
        'optimizer_steps': 0, 'checkpoint_writes': 0, 'backbone_forwards': 2,
        'frozen_v99_and_joint_forward_equal': True, 'native_evaluator_equal': True,
        'evaluation_extent_policy': 'existing rec_candidate_adapter floor at 1e-6;raw loss inputs unchanged',
        'detached_gradient_boundary_correct': True, 'readout_to_core_gradient_connected': True,
        'all_protected_model_states_unchanged': True, 'observations': observations,
        'candidate_trainable_tensors': list(core_parameters),
        'candidate_trainable_parameter_count': sum(value.numel() for value in core_parameters.values()),
        'readout_trainable_tensors': list(dict(readout.named_parameters())),
        'readout_parameter_count': sum(value.numel() for value in readout.parameters()),
        'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2, 'elapsed_seconds': time.time() - started,
        'torch': torch.__version__, 'python': sys.version}
    write_json(directory / 'receipt.json', receipt)
    print('SCANREFER JOINT NATIVE COMPLETE', json.dumps({key: value for key, value in receipt.items()
          if key not in ['observations', 'candidate_trainable_tensors', 'readout_trainable_tensors']}), flush=True)


if __name__ == '__main__':
    main()
