import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location('candidate_local_visual',
    Path(__file__).resolve().parents[1] / 'models/candidate_local_visual.py')
local_visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_visual)


class CandidateLocalVisualTest(unittest.TestCase):
    def test_candidate_shape_changes_which_point_is_near(self):
        xyz = torch.tensor([[[1., 0., 0.], [0., .4, 0.]]])
        boxes = torch.tensor([[[0., 0., 0., 20., 2., 2.],
                               [0., 0., 0., 2., 2., 2.]]])
        indices = local_visual.nearest_candidate_points(xyz, boxes, count=1, query_chunk_size=1)
        self.assertEqual(indices.tolist(), [[[0], [1]]])
        permuted = local_visual.nearest_candidate_points(xyz.flip(1), boxes, count=1)
        self.assertEqual(permuted.tolist(), [[[1], [0]]])

    def test_nonpositive_extents_do_not_change_raw_boxes(self):
        xyz = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
        boxes = torch.tensor([[[0., 0., 0., -1., 0., 2.]]])
        before = boxes.clone()
        self.assertEqual(local_visual.nearest_candidate_points(xyz, boxes, count=1).item(), 0)
        self.assertTrue(torch.equal(before, boxes))

    def make_reader(self):
        torch.manual_seed(2)
        reader = local_visual.CandidateLocalVisual(d_model=12, point_dim=8, hidden_dim=12, heads=3)
        query = torch.randn(2, 3, 12)
        boxes = torch.cat([torch.randn(2, 3, 3), torch.ones(2, 3, 3)], dim=-1)
        xyz = torch.randn(2, 3, 5, 3)
        rgb = torch.randn_like(xyz)
        features = torch.randn(2, 3, 5, 8)
        return reader, (query, boxes, xyz, rgb, features)

    def test_zero_initialization_then_learning_reaches_point_and_query_paths(self):
        reader, inputs = self.make_reader()
        before = reader.read_points(*inputs)
        self.assertTrue(torch.equal(before, torch.zeros_like(before)))
        target = torch.randn_like(before)
        optimizer = torch.optim.AdamW(reader.parameters(), lr=.01)
        (before - target).square().mean().backward()
        self.assertGreater(reader.output_projection.weight.grad.norm().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        (reader.read_points(*inputs) - target).square().mean().backward()
        for name, parameter in reader.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for layer in (reader.point_encoder[0], reader.query_projection,
                      reader.key_projection, reader.value_projection):
            self.assertGreater(layer.weight.grad.norm().item(), 0)

    def test_point_permutation_invariant_but_visual_evidence_matters(self):
        reader, inputs = self.make_reader()
        torch.nn.init.normal_(reader.output_projection.weight, std=.02)
        query, boxes, xyz, rgb, features = inputs
        expected = reader.read_points(*inputs)
        permuted = reader.read_points(query, boxes, xyz.flip(2), rgb.flip(2), features.flip(2))
        self.assertTrue(torch.allclose(expected, permuted, atol=1e-7, rtol=1e-6))
        changed = reader.read_points(query, boxes, xyz, rgb, features + 2.)
        self.assertGreater((expected - changed).abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
