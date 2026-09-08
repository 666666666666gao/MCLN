"""CPU integration check against actual pinned PV-Ground Evaluator methods.

The inputs here are explicitly synthetic. This checks decision arithmetic and
Query correspondence, not a dataset metric or an executed model prediction.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--auditor', type=Path)
    args = parser.parse_args()
    started = time.time()
    root = args.runtime / 'PV-Ground'
    bundle = json.loads((args.runtime / 'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']
    for name in ['src/grounding_evaluator.py', 'models/losses.py']:
        assert sha(root / name) == bundle['files'][name]['sha256'], name
    os.chdir(str(root))
    sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    upstream = load_module('pvg_reference_evaluator', root / 'src/grounding_evaluator.py')
    readout = load_module('pvg_prepared_rec_readout', args.readout)
    auditor = load_module('pvg_independent_rec_audit', args.auditor) if args.auditor else None
    assert Path(sys.modules['models.losses'].__file__).resolve() == root / 'models/losses.py'
    cls = upstream.GroundingEvaluator
    tree = ast.parse((root / 'src/grounding_evaluator.py').read_text())
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GroundingEvaluator')
    trace_lines = {}
    for name, mode in [('evaluate_bbox_by_pos_align', 'bbs'), ('evaluate_bbox_by_sem_align', 'bbf')]:
        function = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)
        line = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'pbox' for target in node.targets))
        trace_lines[getattr(cls, name).__code__] = (line, mode)

    generator = torch.Generator().manual_seed(2027)
    batch_size, queries, tokens, objects = 4, 256, 19, 132
    base = {
        'last_center': torch.randn(batch_size, queries, 3, generator=generator) * 4,
        'last_pred_size': torch.rand(batch_size, queries, 3, generator=generator) + .5,
        'last_sem_cls_scores': torch.randn(batch_size, queries, 256, generator=generator),
        'last_proj_queries': torch.nn.functional.normalize(torch.randn(batch_size, queries, 64, generator=generator), dim=-1),
        'proj_tokens': torch.nn.functional.normalize(torch.randn(batch_size, tokens, 64, generator=generator), dim=-1),
        'all_detected_boxes': torch.zeros(batch_size, objects, 6),
        'all_detected_bbox_label_mask': torch.zeros(batch_size, objects, dtype=torch.bool),
        'center_label': torch.zeros(batch_size, 1, 3),
        'size_gts': torch.ones(batch_size, 1, 3),
        'box_label_mask': torch.ones(batch_size, 1),
        'is_view_dep': torch.tensor([True, False, True, False]),
        'is_hard': torch.tensor([True, True, False, False]),
        'is_unique': torch.tensor([False, True, False, True]),
    }
    for name in ['positive_map', 'modify_positive_map', 'pron_positive_map', 'rel_positive_map', 'other_entity_map', 'auxi_entity_positive_map']:
        base[name] = torch.zeros(batch_size, 1, 256)
    base['positive_map'][:, 0, :3] = 1. / 3
    for name, token, weight in [('modify_positive_map', 4, .6), ('pron_positive_map', 5, .4),
                                ('rel_positive_map', 6, .8), ('other_entity_map', 7, .7),
                                ('auxi_entity_positive_map', 8, .9)]:
        base[name][:, 0, token] = weight
    slots = [0, 2, 11, 80]
    for bid in range(batch_size):
        for number, slot in enumerate(slots):
            query = number * 5 + 1
            base['all_detected_boxes'][bid, slot] = torch.cat([base['last_center'][bid, query], base['last_pred_size'][bid, query]])
        base['all_detected_bbox_label_mask'][bid, slots] = True
        base['center_label'][bid, 0] = base['all_detected_boxes'][bid, 0, :3]
        base['size_gts'][bid, 0] = base['all_detected_boxes'][bid, 0, 3:]
    base['last_pred_size'][:, 0, 0] = -.2
    base['last_pred_size'][:, 3, 2] = 0.
    results = []
    packet_results = []
    for score_case in ['ordinary_components', 'negative_valid_scores']:
        end_points = copy.deepcopy(base)
        if score_case == 'negative_valid_scores':
            end_points['other_entity_map'].fill_(1.)
        original = {key: value.clone() for key, value in end_points.items()}
        for filtered in [False, True]:
            actual = readout.native_root_rec_readout(end_points, filtered)
            assert all(torch.equal(end_points[key], value) for key, value in original.items())
            assert torch.isfinite(actual['scores']).all()
            if filtered:
                assert actual['overlap_valid'].any(-1).all() and (~actual['overlap_valid']).any(-1).all()
            evaluator = cls(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10], prefixes=['last_'],
                            filter_non_gt_boxes=filtered, model='PVGround')
            author_input = dict(end_points)
            author_input['last_pred_size'] = end_points['last_pred_size'].clamp(min=1e-6)
            captures = {}

            def trace(frame, event, arg):
                if event == 'line' and frame.f_code in trace_lines:
                    line, mode = trace_lines[frame.f_code]
                    if frame.f_lineno == line:
                        row = int(frame.f_locals['bid'])
                        captures[(row, mode)] = {'scores': frame.f_locals['scores'].detach().clone(),
                                                'top10': frame.f_locals['top'].detach().clone()}
                return trace

            assert sys.gettrace() is None
            sys.settrace(trace)
            evaluator.evaluate_bbox_by_pos_align(author_input, 'last_')
            evaluator.evaluate_bbox_by_sem_align(author_input, 'last_')
            sys.settrace(None)
            assert len(captures) == batch_size * 2
            for mode_index, mode in enumerate(['bbs', 'bbf']):
                for bid in range(batch_size):
                    assert torch.equal(actual['scores'][bid, mode_index], captures[(bid, mode)]['scores'][0]), (score_case, filtered, bid, mode)
                    assert torch.equal(actual['ranks'][bid, mode_index, :10], captures[(bid, mode)]['top10'][0])
                overlap = torch.stack([upstream._iou3d_par(upstream.box_cxcyczwhd_to_xyzxyz(torch.cat([end_points['center_label'][bid], end_points['size_gts'][bid]], -1)),
                                                          upstream.box_cxcyczwhd_to_xyzxyz(actual['boxes'][bid]))[0][0]
                                       for bid in range(batch_size)])
                for threshold in [.25, .5]:
                    for count in [1, 5, 10]:
                        found = overlap.gather(1, actual['ranks'][:, mode_index, :count]).gt(threshold).any(-1)
                        assert int(found.sum()) == evaluator.dets[('last_', threshold, count, mode)]
                        assert evaluator.gts[('last_', threshold, count, mode)] == batch_size
            selected_valid = actual['overlap_valid'].gather(1, actual['ranks'][:, :, 0])
            if filtered and auditor is not None:
                packets = []
                for bid in range(batch_size):
                    packet = auditor.audit_packet(
                        actual['raw_boxes'][bid].numpy(), actual['raw_scores'][bid].numpy(),
                        actual['scores'][bid].numpy(), actual['overlap_valid'][bid].numpy(),
                        end_points['all_detected_boxes'][bid].numpy(),
                        end_points['all_detected_bbox_label_mask'][bid].numpy(),
                        actual['ranks'][bid, :, 0].numpy(),
                        torch.cat([end_points['center_label'][bid, 0], end_points['size_gts'][bid, 0]]).numpy())
                    packets.append(packet)
                for mode in ['bbs', 'bbf']:
                    for threshold, field in [(.25, 'hit25'), (.5, 'hit50')]:
                        count = sum(packet['modes'][mode][field] for packet in packets)
                        assert count == evaluator.dets[('last_', threshold, 1, mode)]
                packet_results.append({'case': score_case, 'packets': packets})
            if score_case == 'negative_valid_scores' and filtered:
                assert (actual['raw_scores'][actual['overlap_valid'][:, None].expand_as(actual['raw_scores'])] < 0).all()
                assert not selected_valid.any()
            # Remove target geometry and evaluation-only flags altogether: the
            # decision function must operate on predictions and text maps alone.
            no_target_geometry = {key: value for key, value in end_points.items()
                                  if key not in ['center_label', 'size_gts', 'box_label_mask', 'is_view_dep', 'is_hard', 'is_unique']}
            without_gt = readout.native_root_rec_readout(no_target_geometry, filtered)
            assert all(torch.equal(actual[key], without_gt[key]) for key in ['raw_boxes', 'boxes', 'scores', 'ranks', 'overlap_valid'])
            results.append({'case': score_case, 'filter_non_gt_boxes': filtered, 'rows': batch_size,
                            'score_elements_compared': batch_size * 2 * queries,
                            'top10_indices_compared': batch_size * 2 * 10,
                            'rec_count_comparisons': 12, 'selected_overlap_valid': selected_valid.tolist(),
                            'input_tensors_unchanged': True, 'no_target_geometry_required': True})
    assert not torch.cuda.is_initialized()
    receipt = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'elapsed_seconds': time.time() - started, 'pv_commit': bundle['commit'],
               'evaluator_sha256': sha(root / 'src/grounding_evaluator.py'), 'losses_sha256': sha(root / 'models/losses.py'),
               'readout_sha256': sha(args.readout), 'audit_sha256': sha(__file__),
               'input_kind': 'synthetic deterministic predictions and component maps; not real model outputs',
               'torch_version': torch.__version__, 'torch_cuda_initialized': False,
               'model_forwards': 0, 'optimizer_steps': 0, 'formal_rows': 0, 'checks': results}
    if auditor is not None:
        receipt['independent_auditor_sha256'] = sha(args.auditor)
        receipt['independent_packet_checks'] = packet_results
        # Counts here are boundary fixtures, never reported as measured metrics.
        for dataset, rows, lower, strict in [('nr3d', 7899, 4726, 4059), ('sr3d', 17726, 12130, 10157)]:
            assert auditor.formal_rec_check(dataset, rows, lower, strict)['rec_baseline_pass']
            assert not auditor.formal_rec_check(dataset, rows, lower - 1, strict)['rec_baseline_pass']
            assert not auditor.formal_rec_check(dataset, rows, lower, strict - 1)['rec_baseline_pass']
        receipt['formal_floor_boundary_fixtures'] = 6
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'checks'}), flush=True)


if __name__ == '__main__':
    main()
