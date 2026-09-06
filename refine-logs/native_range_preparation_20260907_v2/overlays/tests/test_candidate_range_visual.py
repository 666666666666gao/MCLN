import importlib
from pathlib import Path
import sys
import types

import torch

# Exercise the reader without importing the unrelated full MCLN/CUDA package.
package = types.ModuleType('range_test_models')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'models')]
sys.modules[package.__name__] = package
old_module = importlib.import_module('range_test_models.candidate_local_visual')
new_module = importlib.import_module('range_test_models.candidate_range_visual')
CandidateLocalVisual = old_module.CandidateLocalVisual
CandidateRangeVisual = new_module.CandidateRangeVisual
candidate_region_points = new_module.candidate_region_points


def test_extent_covers_octants_with_same_slot_budget():
    torch.manual_seed(3)
    dense = .1 + torch.rand(128, 3) * .02
    surface = torch.cat([torch.tensor([.8 if code & bit else -.8 for bit in (4, 2, 1)])[None]
                         + torch.rand(8, 3) * .01 for code in range(8)])
    xyz = torch.cat([dense, surface])[None]
    box = torch.tensor([[[0., 0., 0., 2., 2., 2.]]])
    for arm, expected_regions in [('center', 1), ('extent', 8)]:
        indices, valid, regions = candidate_region_points(xyz, box, arm)
        assert indices.shape == valid.shape == regions.shape == (1, 1, 64)
        assert valid.all() and indices.unique().numel() == 64
        assert regions.unique().numel() == expected_regions


def test_empty_regions_do_not_duplicate_points_as_observations():
    xyz = torch.full((1, 64, 3), .2)
    box = torch.tensor([[[0., 0., 0., 2., 2., 2.], [10., 10., 10., 2., 2., 2.]]])
    indices, valid, regions = candidate_region_points(xyz, box, 'extent')
    assert valid[0, 0].sum() == 8 and not valid[0, 1].any()
    assert indices[0, 0, valid[0, 0]].unique().numel() == 8
    assert (regions[0, 0, valid[0, 0]] == 7).all()


def test_same_parameters_zero_start_and_invalid_evidence_invariance():
    torch.manual_seed(4)
    reader = CandidateRangeVisual('extent')
    old = CandidateLocalVisual()
    assert {k: v.shape for k, v in reader.state_dict().items()} == {k: v.shape for k, v in old.state_dict().items()}
    assert sum(p.numel() for p in reader.parameters()) == 145008
    query, box = torch.randn(1, 2, 288), torch.tensor([[[0., 0., 0., 2., 2., 2.]]]).expand(1, 2, 6)
    xyz, rgb, features = torch.randn(1, 2, 64, 3), torch.randn(1, 2, 64, 3), torch.randn(1, 2, 64, 128)
    regions = torch.arange(8).repeat_interleave(8)[None, None].expand(1, 2, 64)
    valid = torch.ones(1, 2, 64, dtype=torch.bool)
    valid[:, 0, 8:] = False
    valid[:, 1] = False
    assert (reader.read_regions(query, box, xyz, rgb, features, valid, regions) == 0).all()
    with torch.no_grad():
        reader.output_projection.weight.normal_()
        reader.output_projection.bias.fill_(1.)
    before = reader.read_regions(query, box, xyz, rgb, features, valid, regions)
    xyz[~valid], rgb[~valid], features[~valid] = 100., 100., 100.
    after = reader.read_regions(query, box, xyz, rgb, features, valid, regions)
    assert torch.equal(before, after)
    assert (after[:, 1] == 0).all() and torch.isfinite(after).all()
    after.square().sum().backward()
    assert reader.point_encoder[0].weight.grad.norm() > 0
