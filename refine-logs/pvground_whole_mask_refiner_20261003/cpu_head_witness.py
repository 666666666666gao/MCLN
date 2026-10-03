"""Real point geometry with constructed inputs; standalone CPU head only."""
import argparse
import ast
import datetime
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from pvground_whole_mask_box_refiner import WholeMaskSupportBoxRefiner, install_whole_mask_refinement


def native_box_loss(path):
    tree = ast.parse(path.read_text())
    scope = dict(torch=torch, F=F)
    wanted = {'box_cxcyczwhd_to_xyzxyz', '_volume_par', '_intersect_par', '_iou3d_par', 'generalized_box_iou3d'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in functions} == wanted
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
        and any(isinstance(member, ast.FunctionDef) and member.name == 'loss_boxes' for member in node.body))
    methods = {node.name: node for node in owner.body if isinstance(node, ast.FunctionDef)}
    functions += [methods['loss_boxes'], methods['_get_src_permutation_idx']]
    exec(compile(ast.Module(body=functions), str(path), 'exec'), scope)
    native = SimpleNamespace()
    native._get_src_permutation_idx = lambda indices: scope['_get_src_permutation_idx'](native, indices)
    return lambda boxes, targets, indices: scope['loss_boxes'](native, {'pred_boxes': boxes}, targets, indices, 1, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--native-loss', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    torch.manual_seed(2027)
    receipt = json.loads((args.fixtures / 'receipt.json').read_bytes())
    fixture = receipt['rows'][0]
    path = args.fixtures / fixture['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['file_sha256']
    inputs = torch.load(str(path), map_location='cpu')
    points = inputs['point_clouds']
    superpoints = inputs['superpoint'][0]
    assert points.shape == (1, 50000, 6) and superpoints.shape == (50000,)
    slots = int(superpoints.max()) + 1
    text = torch.linspace(-2, 2, slots)[None, :].requires_grad_()
    masks = (torch.linspace(-1, 1, 256)[:, None] + torch.cos(torch.arange(slots)[None, :] / 13)).requires_grad_()
    alpha = torch.tensor(.35, requires_grad=True)
    query = torch.randn(1, 256, 288).requires_grad_()
    center = points[:, torch.linspace(0, 49999, 256).long(), :3].clone().requires_grad_()
    span = points[0, :, :3].max(0).values - points[0, :, :3].min(0).values
    size = (.1 * span).repeat(1, 256, 1).requires_grad_()

    def endpoints():
        return dict(superpoints=[superpoints], last_pred_masks=[text.expand(256, -1)[None]],
                    sp_last_pred_masks=[masks], adaptive_weights=[alpha])

    slot = SimpleNamespace(candidate_box_refiner=None)
    install_whole_mask_refinement(slot, use_whole_range=True)
    head = slot.candidate_box_refiner
    control = WholeMaskSupportBoxRefiner(use_whole_range=False)
    control.load_state_dict(head.state_dict(), strict=True)
    whole_initial = head(query, points, center, size, endpoints())
    local_initial = control(query, points, center, size, endpoints())
    assert torch.equal(whole_initial[0], center) and torch.equal(whole_initial[1], size)
    assert all(torch.equal(a, b) for a, b in zip(whole_initial, local_initial))
    assert head.aggregate[0].in_features == 1302 and len(head.state_dict()) == 10
    assert all(torch.equal(value, control.state_dict()[name]) for name, value in head.state_dict().items())
    assert hashlib.sha256(args.native_loss.read_bytes()).hexdigest() == '920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de'
    loss_boxes = native_box_loss(args.native_loss)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [dict(boxes=torch.cat([center[0, :1].detach() + .02, size[0, :1].detach() * 1.07], dim=-1))]
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-5, weight_decay=5e-4)
    records = []
    for step in (1, 2):
        optimizer.zero_grad()
        output = endpoints()
        refined_center, refined_size = head(query, points, center, size, output)
        losses = loss_boxes(torch.cat([refined_center, refined_size], dim=-1), targets, indices)
        objective = 10 * losses['loss_bbox'] + 2 * losses['loss_giou']
        assert bool(torch.isfinite(objective))
        global_gradient = torch.autograd.grad(objective, output['whole_mask_range_evidence'], retain_graph=True)[0]
        route = torch.autograd.grad(output['whole_mask_range_evidence'], (text, masks, alpha),
            grad_outputs=global_gradient, retain_graph=True)
        parent = torch.autograd.grad(objective, (center, size, query), retain_graph=True)
        objective.backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in head.named_parameters()}
        assert gradients['output.weight'] > 0 and all(bool(torch.isfinite(value).all()) for value in route + parent)
        if step == 1:
            assert float(global_gradient.norm()) == 0 and all(float(value.norm()) == 0 for value in route)
        else:
            assert float(global_gradient.norm()) > 0 and all(float(value.norm()) > 0 for value in route)
            assert float(parent[2].norm()) > 0
            assert gradients['member.0.weight'] > 0 and gradients['condition.weight'] > 0 and gradients['aggregate.0.weight'] > 0
        torch.nn.utils.clip_grad_norm_(head.parameters(), .1)
        optimizer.step()
        records.append(dict(math_step=step, native_bbox=float(losses['loss_bbox']),
            native_giou=float(losses['loss_giou']), standalone_head_gradients=gradients,
            global_range_gradient_norm=float(global_gradient.norm()),
            global_only_route_gradient_norms=[float(value.norm()) for value in route],
            coarse_and_geometry_query_gradient_norms=[float(value.norm()) for value in parent]))
    stream = io.BytesIO()
    torch.save(dict(model=head.state_dict(), optimizer=optimizer.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu')
    restored = WholeMaskSupportBoxRefiner(use_whole_range=True)
    restored.load_state_dict(saved['model'], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-5, weight_decay=5e-4)
    restored_optimizer.load_state_dict(saved['optimizer'])
    assert len(restored_optimizer.state) == 10 and all(int(value['step']) == 2 for value in restored_optimizer.state.values())
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer['param_groups'] == saved['optimizer']['param_groups']
    assert set(actual_optimizer['state']) == set(saved['optimizer']['state'])
    for key, value in saved['optimizer']['state'].items():
        actual = actual_optimizer['state'][key]
        assert int(actual['step']) == int(value['step'])
        assert torch.equal(actual['exp_avg'], value['exp_avg'])
        assert torch.equal(actual['exp_avg_sq'], value['exp_avg_sq'])
    with torch.no_grad():
        before = head(query, points, center, size, endpoints())
        after = restored(query, points, center, size, endpoints())
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert not torch.cuda.is_initialized()
    result = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
        fixture_row=fixture['training_row_id'], scan_id=fixture['scan_id'], fixture_sha256=fixture['file_sha256'],
        point_count=50000, candidate_count=256, range_channels=109, coarse_context_channels=6,
        head_input_channels=1302, standalone_parameters=sum(p.numel() for p in head.parameters()),
        native_loss_sha256=hashlib.sha256(args.native_loss.read_bytes()).hexdigest(),
        initial_zero_residual_exact=True, shared_initial_state_exact=True, CPU_only=True,
        constructed_queries_masks_coarse_boxes_and_targets=True, dataset_GT_used=False,
        standalone_head_forwards=6, standalone_math_optimizer_updates=2, real_fit_optimizer_updates=0,
        PVGround_model_forwards=0, CUDA_initialized=False, installed_in_actual_model=False,
        in_memory_model_and_optimizer_restore_exact=True, optimizer_moments_and_groups_exact=True, weight_files_created=0, records=records,
        accuracy_results_available=False, limitations=['One actual geometry fixture; constructed mathematical inputs and targets.',
            'AST executes native loss_boxes with fixed constructed correspondence, not actual Hungarian/whole-criterion behavior.',
            'CPU standalone gradients/restore do not prove CUDA, real-model routing, capacity or accuracy.'])
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
