"""Measure what a frozen V99 teacher adds on ScanRefer fit rows, with zero updates."""

import argparse
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


def transfer_rows(boxes, scores, default_indices, parent_indices, runtime, matches, roots):
    """Correspond by current geometry; Query IDs are diagnostic provenance only."""
    import torch
    from models.rec_reranker import compute_query_ious

    root_valid = torch.ones(roots.shape[:2], dtype=torch.bool, device=roots.device)
    native_ious = compute_query_ious(boxes, roots, root_valid)
    variant_boxes = runtime['rec_geometry_boxes']
    variant_scores = runtime['rec_geometry_scores']
    variant_valid = runtime['rec_geometry_valid_mask']
    assert variant_valid.any(dim=1).all()
    # The protected ScanRefer contract uses stable, first-index Top-1 on this axis.
    selected_flat = variant_scores.masked_fill(~variant_valid, -float('inf')).argmax(dim=1)
    batch_indices = torch.arange(len(boxes), device=boxes.device)
    teacher_boxes = variant_boxes[batch_indices, selected_flat]
    teacher_ious = compute_query_ious(teacher_boxes[:, None], roots, root_valid)[:, 0]
    overlaps = compute_query_ious(boxes, teacher_boxes[:, None], root_valid)
    corresponding = overlaps.argmax(dim=1)
    native_selected = scores.argsort(dim=1, descending=True)[:, 0]
    records = []
    for index in range(len(boxes)):
        source_query = int(parent_indices[index, selected_flat[index] // 7])
        matched_query = matches[index][0][matches[index][1] == 0]
        assert matched_query.numel() == 1
        matched_query = int(matched_query.item())
        values = {
            'native': native_ious[index, native_selected[index]],
            'raw_default': native_ious[index, default_indices[index]],
            'hungarian_root': native_ious[index, matched_query],
            'native_best': native_ious[index].max(),
            'teacher': teacher_ious[index],
            'teacher_source_query': native_ious[index, source_query],
            'corresponding_query': native_ious[index, corresponding[index]],
        }
        records.append({
            'ious': {name: float(value) for name, value in values.items()},
            'native_query_index': int(native_selected[index]),
            'hungarian_root_query_index': matched_query,
            'teacher_source_query_index': source_query,
            'teacher_variant_index': int(selected_flat[index] % 7),
            'corresponding_query_index': int(corresponding[index]),
            'teacher_correspondence_iou': float(overlaps[index, corresponding[index]]),
            'teacher_box': teacher_boxes[index].tolist(),
            'corresponding_box': boxes[index, corresponding[index]].tolist(),
            'root_box': roots[index, 0].tolist(),
        })
    return records


def summarize(records):
    names = records[0]['ious']
    hits = {name: {str(t): sum(row['ious'][name] > t for row in records)
                   for t in (.25, .5)} for name in names}
    effects = {}
    for reference in ('native', 'hungarian_root', 'native_best', 'teacher_source_query'):
        effects[reference] = {}
        for threshold in (.25, .5):
            repair = sum(row['ious']['teacher'] > threshold >= row['ious'][reference]
                         for row in records)
            damage = sum(row['ious'][reference] > threshold >= row['ious']['teacher']
                         for row in records)
            effects[reference][str(threshold)] = {'repair': repair, 'damage': damage,
                                                  'net': repair - damage}
    return {
        'rows': len(records), 'hits': hits, 'teacher_effects': effects,
        'teacher_iou_greater_than': {
            name: sum(row['ious']['teacher'] > row['ious'][name] for row in records)
            for name in ('native', 'hungarian_root', 'native_best', 'teacher_source_query')},
        'non_regressed_variant_rows': sum(row['teacher_variant_index'] != 0 for row in records),
        'geometry_correspondence_differs_from_source_query': sum(
            row['corresponding_query_index'] != row['teacher_source_query_index'] for row in records),
        'teacher_passing_but_corresponding_query_failing': {
            str(t): sum(row['ious']['teacher'] > t >= row['ious']['corresponding_query']
                        for row in records) for t in (.25, .5)},
    }


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
    assert manifest['rows'] == 512 and manifest['optimizer_steps'] == 0
    os.chdir(str(source))
    sys.path.insert(0, str(source))

    import copy
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_joint_readout import JointRecReadout

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    command = build_authoritative_command(directory / 'unused_official_output')
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
    assert args.butd and not args.butd_cls and not args.butd_gt
    assert args.eval_use_selector_choice_scores
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    payload = torch.load(args.checkpoint_path, map_location='cpu')
    initial = {name[7:]: value for name, value in payload['model'].items()}
    model = TrainTester.get_model(args).cuda().eval().requires_grad_(False)
    model.load_state_dict(initial, strict=True)
    artifacts = {name: torch.load(item['path'], map_location='cpu')
                 for name, item in manifest['artifacts'].items() if name != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    readout_initial = {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}
    criterion, set_criterion = TrainTester.get_criterion(args)
    protocol = {}

    class AuditDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            fit = []
            for index, row in enumerate(annos):
                row['_teacher_audit_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                if int(hashlib.sha256(code).hexdigest()[:8], 16) % 5 != 0:
                    fit.append(index)
            selected = [fit[index * len(fit) // manifest['rows']] for index in range(manifest['rows'])]
            selected_scenes = {annos[index]['scan_id'] for index in selected}
            protocol.update(selected_row_ids=selected, fit_rows=len(fit),
                            selected_scans=sorted(selected_scenes),
                            selected_spaces=sorted({scan.split('_')[0] for scan in selected_scenes}),
                            selection='512 equally spaced fit-row positions, without quality selection',
                            previous_pretraining_has_seen_these_rows=True)
            # Original distractor construction needs all expressions of each selected scene.
            annos[:] = [row for row in annos if row['scan_id'] in selected_scenes]
            super()._scene_graph_parse(annos)

    dataset = AuditDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    by_id = {row['_teacher_audit_id']: row for row in dataset.annos}
    dataset.annos = [by_id[index] for index in protocol['selected_row_ids']]
    dataset.augment = False
    write_json(directory / 'protocol.json', protocol)
    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0,
                                         generator=torch.Generator().manual_seed(0))
    tester = object.__new__(TrainTester)
    tester.logger = logging.getLogger('scanrefer-teacher-audit')
    teacher_evaluator = tester._build_grounding_evaluator(args, ['last_'])
    assert not teacher_evaluator.filter_non_gt_boxes
    native_evaluator = copy.deepcopy(teacher_evaluator)
    native_evaluator.eval_use_rec_geometry_reranker_scores = False
    native_evaluator.eval_use_rec_reranker_scores = False
    records, batch_seconds = [], []
    started = time.time()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            before = time.time()
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            inputs['train'] = False
            outputs = model(inputs)
            teacher = readout(outputs, inputs)
            boxes = torch.cat([outputs['last_center'], outputs['last_pred_size']], dim=-1)
            roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
            assert batch['box_label_mask'][:, 0].bool().all() and (boxes[..., 3:] > 0).all()
            captured = []

            def capture_matches(module, arguments, result):
                if torch.equal(arguments[0]['pred_boxes'], boxes):
                    assert all(torch.equal(target['boxes'][0], roots[i, 0])
                               for i, target in enumerate(arguments[1]))
                    captured.append(result)

            handle = set_criterion.matcher.register_forward_hook(capture_matches)
            outputs.update(batch)
            native_loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            handle.remove()
            assert len(captured) == 1 and torch.isfinite(native_loss)
            rows = transfer_rows(boxes, outputs['selected_source_scores'],
                teacher['parent']['candidate_batch']['default_top1_query_index'],
                teacher['parent']['candidate_batch']['query_indices'], teacher['runtime'], captured[0], roots)
            for offset, row in enumerate(rows):
                position = len(records) + offset
                row.update(row_id=protocol['selected_row_ids'][position], scan_id=batch['scan_ids'][offset])
            records.extend(rows)
            native_evaluator.evaluate_bbox_by_pos_align(outputs, 'last_')
            evaluated = dict(outputs)
            evaluated.update(teacher['runtime'])
            teacher_evaluator.evaluate_bbox_by_pos_align(evaluated, 'last_')
            torch.cuda.synchronize()
            batch_seconds.append(time.time() - before)
            if batch_index == 0 or len(records) == 512:
                print('SCANREFER TEACHER AUDIT', json.dumps({'rows': len(records),
                    'total': 512, 'batch_seconds': batch_seconds[-1]}), flush=True)
            del outputs, teacher, evaluated, inputs, batch, boxes, roots, native_loss

    summary = summarize(records)
    assert len(records) == 512
    for name, evaluator in [('native', native_evaluator), ('teacher', teacher_evaluator)]:
        for threshold in (.25, .5):
            key = ('last_', threshold, 1, 'bbs')
            assert evaluator.gts[key] == 512
            assert evaluator.dets[key] == summary['hits'][name][str(threshold)]
    assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value.detach().cpu(), readout_initial[name]) for name, value in readout.state_dict().items())
    write_json(directory / 'rows.json', records)
    receipt = {'schema': 'mcln-scanrefer-v99-teacher-transfer-audit-v1', 'status': 'pass',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'manifest_sha256': file_sha(option.manifest), 'summary': summary,
        'rows_sha256': file_sha(directory / 'rows.json'), 'optimizer_steps': 0, 'formal_rows': 0,
        'checkpoint_writes': 0, 'native_evaluator_matches_row_counts': True,
        'all_model_states_unchanged': True, 'batch_seconds': batch_seconds,
        'elapsed_seconds': time.time() - started, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'interpretation': 'teacher-target feasibility on previously seen fit rows; not student quality or generalization'}
    write_json(directory / 'receipt.json', receipt)
    print('SCANREFER TEACHER AUDIT COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
