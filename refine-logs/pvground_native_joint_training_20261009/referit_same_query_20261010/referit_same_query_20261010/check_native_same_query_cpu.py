"""Exercise native bbox/mask reporting on explicit CPU fixtures, never benchmark scores."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
root = Path(__file__).resolve().parent
spec = json.loads((root / 'CHECK_SPEC.json').read_bytes())
import src.grounding_evaluator as prepared_module
assert Path(prepared_module.__file__).resolve() == root / 'PV-Ground/src/grounding_evaluator.py'
original_spec = importlib.util.spec_from_file_location('original_native_evaluator',
    str(Path(spec['warm_source']) / 'src/grounding_evaluator.py'))
original_module = importlib.util.module_from_spec(original_spec)
original_spec.loader.exec_module(original_module)


def fixture(negative=False, all_outside=False, multi_gt=False):
    data = {}
    for name in ('positive_map', 'modify_positive_map', 'pron_positive_map',
                 'other_entity_map', 'auxi_entity_positive_map', 'rel_positive_map'):
        data[name] = torch.zeros(1, 1, 256)
    data['positive_map'][0, 0, 0] = 1
    logits = torch.zeros(1, 256, 256)
    logits[0, 0, 0], logits[0, 1, 0], logits[0, 2, 0] = 7, 6, 5
    if negative:
        data['other_entity_map'][0, 0, 1] = 1
        logits[0, 1, 0], logits[0, 1, 1] = -2, 7
        logits[0, 2, 0], logits[0, 2, 1] = -1, 6
    centers = torch.full((1, 256, 3), 30.)
    centers[0, 1] = torch.tensor([0., 0., 0.])
    centers[0, 2] = torch.tensor([0.5, 0., 0.])
    if all_outside:
        centers.fill_(30.)
    data.update(last_sem_cls_scores=logits, last_center=centers,
        last_pred_size=torch.full((1, 256, 3), 2.),
        center_label=torch.zeros(1, 1, 3), size_gts=torch.full((1, 1, 3), 2.),
        box_label_mask=torch.ones(1, 1),
        all_detected_boxes=torch.tensor([[[0., 0., 0., 2., 2., 2.]]]),
        all_detected_bbox_label_mask=torch.ones(1, 1, dtype=torch.bool),
        gt_masks=torch.tensor([[[1, 1, 0, 0]]]),
        superpoints=torch.tensor([[0, 1, 2, 3]]),
        last_pred_masks=torch.zeros(1, 256, 4),
        adaptive_weights=torch.zeros(1),
        sp_last_pred_masks=torch.full((1, 256, 4), -8.),
        proj_tokens=torch.zeros(1, 256, 2),
        last_proj_queries=torch.zeros(1, 256, 2),
        is_view_dep=torch.tensor([False]), is_unique=torch.tensor([True]),
        is_hard=torch.tensor([False]))
    data['sp_last_pred_masks'][0, 0, :2] = 8
    data['sp_last_pred_masks'][0, 2, :2] = 8
    data['sp_last_pred_masks'][0, 1, 2:] = 8
    data['proj_tokens'][0, 0, 0] = 1
    data['proj_tokens'][0, 1, 1] = 1
    data['last_proj_queries'][0, 2, 0] = 1
    data['last_proj_queries'][0, 1, 1] = 1
    if multi_gt:
        for name in ('positive_map', 'modify_positive_map', 'pron_positive_map',
                     'other_entity_map', 'auxi_entity_positive_map', 'rel_positive_map'):
            data[name] = torch.cat((data[name], torch.zeros(1, 1, 256)), dim=1)
        data['positive_map'][0, 1, 2] = 1
        data['center_label'] = torch.tensor([[[0., 0., 0.], [10., 0., 0.]]])
        data['size_gts'] = torch.full((1, 2, 3), 2.)
        data['box_label_mask'] = torch.ones(1, 2)
        data['gt_masks'] = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1]]])
        data['all_detected_boxes'] = torch.tensor([[[0., 0., 0., 2., 2., 2.],
            [10., 0., 0., 2., 2., 2.]]])
        data['all_detected_bbox_label_mask'] = torch.ones(1, 2, dtype=torch.bool)
    return data


cases = []
started = time.perf_counter()
for name, filtered, negative, outside, multi_gt in (
        ('positive_scene_filter', True, False, False, False),
        ('unfiltered', False, False, False, False),
        ('negative_valid_scores', True, True, False, False),
        ('all_candidates_outside_scene', True, False, True, False),
        ('multi_GT_root_only', True, False, False, True)):
    data = fixture(negative, outside, multi_gt)
    original = original_module.GroundingEvaluator(only_root=True, prefixes=['last_'],
        filter_non_gt_boxes=filtered, model='PVGround')
    prepared = prepared_module.GroundingEvaluator(only_root=True, prefixes=['last_'],
        filter_non_gt_boxes=filtered, model='PVGround')
    original.evaluate(copy.deepcopy(data), 'last_')
    prepared_data = copy.deepcopy(data)
    prepared.evaluate(prepared_data, 'last_')
    bbox_keys = [key for key in original.dets if isinstance(key, tuple)]
    assert all(original.dets[key] == prepared.dets[key] and
        original.gts[key] == prepared.gts[key] for key in bbox_keys)
    assert prepared.gts[('last_', .5, 1, 'bbs')] == 1
    selected = int(prepared_data['last_bbs_selected_query'][0])
    if name == 'positive_scene_filter':
        assert selected == 1
        assert original.dets['mask_pos'] == 1 and prepared.dets['mask_pos'] == 0
        assert original.dets['mask_sem'] == 1 and prepared.dets['mask_sem'] == 0
    if name == 'unfiltered':
        assert selected == 0
    predicted_mask = data['sp_last_pred_masks'][0, selected] > 0
    gt_mask = data['gt_masks'][0, 0].bool()
    expected_iou = float((predicted_mask & gt_mask).sum()) / float((predicted_mask | gt_mask).sum())
    assert abs(float(prepared.dets['mask_pos']) - expected_iou) < 1e-12
    assert abs(float(prepared.dets['mask_sem']) - expected_iou) < 1e-12
    assert prepared.dets['overall_mask'] == int(expected_iou > .25)
    assert prepared.dets['overall50_mask'] == int(expected_iou > .5)
    assert abs(prepared.gts['mask_pos'] - 1) < 1e-12
    assert abs(prepared.gts['mask_sem'] - 1) < 1e-12
    cases.append(dict(name=name, bbox_counters_equal_to_original=True,
        selected_query=selected, both_mask_counters_equal_selected_query_IoU=True,
        synthetic_selected_mask_IoU=expected_iou))
assert not torch.cuda.is_initialized()
result = dict(status='SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU_PASS', cases=cases,
    fixture_shape=dict(batch=1, queries=256, tokens=256, points=4),
    known_Box_Mask_disagreement_fixture_detected=True,
    evaluator_instances=len(cases)*2, native_evaluate_calls=len(cases)*2,
    elapsed_seconds=time.perf_counter()-started, CUDA_initialized=False,
    PV_model_constructors=0, PV_forwards=0, dataset_constructors=0,
    criterion_calls=0, optimizer_steps=0, saved_weights=0,
    current_training_queries=0, formal_accuracy=None,
    full_benchmark_evaluation_completed=False, GPU_training_admission=False,
    full_goal_complete=False)
(root / 'SAME_QUERY_CPU_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
