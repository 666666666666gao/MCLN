"""Exercise the exact CUDA/data operators needed by the isolated PV-Ground port."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.time()
    import numpy as np
    import torch
    import spconv
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as stack
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils as roi
    from models.pv_ground import PVGround
    from prepare_data import DataProcessor
    import pointnet2._ext as point_extension

    assert torch.__version__ == '1.10.2+cu111' and spconv.__version__ == '2.3.6'
    assert torch.cuda.is_available()
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    xyz0 = torch.tensor([[0., 0., 0.], [.1, .01, 0.], [.33, -.02, .01], [.9, .03, .02]])
    source = torch.stack([xyz0, xyz0 + torch.tensor([2., .2, .1])]).cuda()
    fps = stack.farthest_point_sample(source.contiguous(), 3).long().cpu()
    reference_fps = []
    for batch in source.cpu():
        selected, distances, current = [], torch.full((4,), float('inf')), 0
        for _ in range(3):
            selected.append(current)
            distances = torch.minimum(distances, ((batch - batch[current]) ** 2).sum(-1))
            current = int(distances.argmax())
        reference_fps.append(selected)
    assert fps.tolist() == reference_fps

    xyz = source.reshape(-1, 3).contiguous()
    centers = torch.stack([source[0, 0], source[0, 0] + 5., source[1, 0], source[1, 0] + 5.]).contiguous()
    xyz_count = torch.tensor([4, 4], dtype=torch.int32, device='cuda')
    center_count = torch.tensor([2, 2], dtype=torch.int32, device='cuda')
    features = torch.randn(8, 3, device='cuda', requires_grad=True)
    index, empty = stack.ball_query(.2, 4, xyz, xyz_count, centers, center_count)
    assert empty.cpu().tolist() == [False, True, False, True]
    grouped, grouped_index = stack.QueryAndGroup(.2, 4, use_xyz=True)(xyz, xyz_count, centers, center_count, features)
    assert torch.equal(index, grouped_index)
    reference = torch.zeros(4, 6, 4)
    expected_gradient = torch.zeros(8, 3)
    for row in (0, 2):
        start = (row // 2) * 4
        local = index[row].long().cpu()
        assert set(local.tolist()) == {0, 1}
        global_index = local + start
        reference[row, :3] = (xyz.cpu()[global_index] - centers.cpu()[row]).T
        reference[row, 3:] = features.detach().cpu()[global_index].T
        for item in global_index.tolist():
            expected_gradient[item] += 1.
    assert torch.allclose(grouped.cpu(), reference, atol=1e-6, rtol=1e-6)
    grouped[:, 3:].sum().backward()
    assert torch.equal(features.grad.cpu(), expected_gradient)

    boxes = torch.tensor([[[.15, 0., 0., .5, .5, .5, 0.]], [[2.15, .2, .1, .5, .5, .5, 0.]]], device='cuda')
    membership = roi.points_in_boxes_gpu(source.contiguous(), boxes.contiguous()).cpu()
    reference_membership = []
    for points, one_box in zip(source.cpu(), boxes.cpu()):
        inside = ((points - one_box[0, :3]).abs() <= one_box[0, 3:6] / 2).all(-1)
        reference_membership.append(torch.where(inside, torch.zeros_like(inside, dtype=torch.int32), torch.full_like(inside, -1, dtype=torch.int32)))
    assert torch.equal(membership, torch.stack(reference_membership))

    cfg_from_yaml_file('wandb_config.yaml', cfg)
    processor = DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), False, 6)
    batches = []
    for points in source.cpu().numpy():
        points = np.concatenate([points, np.full_like(points, .25)], axis=1).astype(np.float32)
        value = processor.forward({'points': points.copy(), 'use_lead_xyz': True})
        assert np.array_equal(value['points'], points)
        for voxel, coord, count in zip(value['voxels'], value['voxel_coords'], value['voxel_num_points']):
            # Point2VoxelCPU3d quantizes float32 coordinates; float64 promotion moves boundary points to a different cell.
            minimum = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[:3], dtype=np.float32)
            expected = np.floor((voxel[0, :3] - minimum) / np.float32(.02)).astype(np.int32)[::-1]
            assert np.array_equal(coord, expected) and count == 1
        batches.append(value)
    collated = processor.collate_batch(batches)
    assert collated['points'].shape == (8, 7) and collated['voxel_coords'].shape == (8, 4)
    assert collated['batch_size'] == 2
    assert np.array_equal(collated['points'][:, 0], np.repeat([0, 1], 4))
    torch.cuda.synchronize()
    checkpoint = Path('/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth')
    assert checkpoint.stat().st_size == 830036622
    receipt = {
        'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'torch': torch.__version__, 'spconv': spconv.__version__, 'gpu': torch.cuda.get_device_name(0),
        'fps_indices': fps.tolist(), 'empty_groups': empty.cpu().tolist(),
        'grouping_max_abs_error': float((grouped.cpu() - reference).abs().max()),
        'grouping_gradient_matches_reference': True, 'roi_membership_matches_reference': True,
        'voxel_coordinate_order': 'zyx', 'voxel_reference_arithmetic': 'float32', 'raw_point_order_preserved': True,
        'point_extension': point_extension.__file__, 'stack_extension': stack.pointnet2.__file__,
        'full_model_imported': PVGround.__name__, 'model_forwards': 0, 'optimizer_steps': 0, 'dataset_rows': 0,
        'elapsed_seconds': time.time() - started,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('PVG_RUNTIME_WITNESS ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
