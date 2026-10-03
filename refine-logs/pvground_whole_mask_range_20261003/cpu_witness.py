"""Real fixture geometry, constructed mathematical Masks, CPU-only equivalence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from whole_mask_range import member_statistics, mask_range_evidence


def objective(evidence):
    profile = evidence['axis_profile']
    bin_weight = torch.linspace(-1, 1, profile.shape[-1], dtype=profile.dtype)
    return ((profile * bin_weight.square()).sum() +
            evidence['mean_normalized'].square().sum() +
            evidence['second_normalized'].sum() +
            evidence['member_lower_mean_normalized'].sum() +
            evidence['member_upper_mean_normalized'].square().sum() +
            evidence['support_fraction'].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    receipt = json.loads((args.fixtures / 'receipt.json').read_bytes())
    assert receipt['status'] == 'pass' and receipt['training_fixture_rows'] == 4
    assert receipt['model_forwards'] == 0 and receipt['optimizer_steps'] == 0
    records = []
    for fixture in receipt['rows']:
        path = args.fixtures / fixture['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['file_sha256']
        inputs = torch.load(str(path), map_location='cpu')
        assert inputs['point_clouds'].shape == (1, 50000, 6)
        assert inputs['superpoint'].shape == (1, 50000)
        xyz = inputs['point_clouds'][0, :, :3].numpy().astype(np.float64)
        native_ids = inputs['superpoint'][0].numpy()
        geometry = member_statistics(xyz, native_ids, bins=32)
        assert int(geometry['count'].sum()) == len(xyz)
        assert np.array_equal(geometry['histogram'].sum(-1), np.repeat(geometry['count'][:, None], 3, axis=1))
        assert np.isfinite(geometry['lower']).all() and np.isfinite(geometry['upper']).all()
        slots = int(native_ids.max()) + 1
        shared_text = torch.linspace(-2, 2, slots, dtype=torch.float64).unsqueeze(0).requires_grad_()
        query = (torch.linspace(-1, 1, 256, dtype=torch.float64)[:, None] +
                 torch.cos(torch.arange(slots, dtype=torch.float64)[None, :] / 13)).requires_grad_()
        alpha = torch.tensor(.35, dtype=torch.float64, requires_grad=True)
        compressed = mask_range_evidence(shared_text.expand_as(query), query, alpha, geometry)
        assert compressed['axis_profile'].shape == (256, 3, 32)
        assert torch.allclose(compressed['axis_profile'].sum(-1), torch.ones(256, 3, dtype=torch.float64), atol=1e-12, rtol=0)
        selected = torch.tensor([0, 31, 127, 255])
        compressed_selected = {name: value[selected] for name, value in compressed.items()}
        fused = alpha * shared_text + (1 - alpha) * query[selected]
        point_probability = fused[:, inputs['superpoint'][0]].sigmoid()
        weights = point_probability / point_probability.sum(-1, keepdim=True)
        normalized = (xyz - geometry['origin']) / geometry['span']
        xyz_tensor = torch.from_numpy(normalized)
        bin_ids = np.minimum((normalized * 32).astype(np.int64), 31)
        profile = torch.zeros(4, 3, 32, dtype=torch.float64)
        for axis in range(3):
            profile[:, axis].scatter_add_(1, torch.from_numpy(bin_ids[:, axis]).expand(4, -1), weights)
        observed_ids = geometry['native_ids']
        inverse = np.searchsorted(observed_ids, native_ids)
        mean = weights @ xyz_tensor
        second = weights @ xyz_tensor.square()
        dense = dict(axis_profile=profile, mean_normalized=mean, second_normalized=second,
            variance_normalized=second - mean.square(),
            mean_xyz=torch.from_numpy(geometry['origin']) + mean * torch.from_numpy(geometry['span']),
            member_lower_mean_normalized=weights @ torch.from_numpy(geometry['lower'][inverse]),
            member_upper_mean_normalized=weights @ torch.from_numpy(geometry['upper'][inverse]),
            support_fraction=point_probability.mean(-1))
        forward_errors = {name: float((value - dense[name]).abs().max()) for name, value in compressed_selected.items()}
        assert all(error <= 1e-9 for error in forward_errors.values()), forward_errors
        parameters = (shared_text, query, alpha)
        cg = torch.autograd.grad(objective(compressed_selected), parameters, retain_graph=True)
        dg = torch.autograd.grad(objective(dense), parameters)
        gradient_errors = [float((a-b).abs().max()) for a, b in zip(cg, dg)]
        assert all(error <= 1e-9 for error in gradient_errors), gradient_errors
        assert all(bool(torch.isfinite(value).all()) and float(value.abs().sum()) > 0 for value in cg)
        center_bins = np.minimum((geometry['mean'] * 32).astype(np.int64), 31)
        centered = np.zeros_like(geometry['histogram'])
        for axis in range(3):
            centered[np.arange(len(observed_ids)), axis, center_bins[:, axis]] = geometry['count']
        records.append(dict(training_row_id=fixture['training_row_id'], scan_id=fixture['scan_id'],
            point_count=len(xyz), observed_sp_slots=len(observed_ids), native_sp_slots=slots,
            fixture_sha256=fixture['file_sha256'], profile_bins=32, candidate_count=256,
            compressed_geometry_bytes=sum(value.nbytes for value in geometry.values()),
            forward_errors=forward_errors, gradient_errors=gradient_errors,
            center_only_histogram_l1_difference=int(np.abs(centered - geometry['histogram']).sum()),
            superpoints_spanning_multiple_bins=int((np.count_nonzero(geometry['histogram'], axis=-1) > 1).any(-1).sum())))
    assert not torch.cuda.is_initialized()
    result = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
        fixture_rows=len(records), records=records, CPU_only=True, cuda_initialized=False,
        masks='constructed mathematical inputs; not model predictions',
        GT_used=False, model_forwards=0, optimizer_updates=0, weight_files_created=0,
        integrated_into_model=False, accuracy_results_available=False,
        limitations=['32-bin directional profiles are quantized and do not recover arbitrary shape.',
                     'Weighted member min/max summaries are not final box boundaries.',
                     'Only four real training-scene fixtures and CPU double math are checked; CUDA/whole-model behavior is untested.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
