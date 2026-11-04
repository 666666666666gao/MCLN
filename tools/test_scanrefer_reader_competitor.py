"""Remote-only checks for zero-initialized Reader competitor context."""

import unittest

import torch
from models.candidate_local_visual import CandidateLocalVisual


class ReaderCompetitorTest(unittest.TestCase):
    def test_initial_identity_and_scale_gradient(self):
        torch.manual_seed(0)
        reader = CandidateLocalVisual(d_model=8, point_dim=4, hidden_dim=8, heads=2)
        torch.nn.init.normal_(reader.output_projection.weight)
        query = torch.randn(2, 6, 8)
        boxes = torch.cat([torch.randn(2, 6, 3), torch.ones(2, 6, 3)], dim=-1)
        xyz = torch.randn(2, 6, 3, 3)
        rgb = torch.randn(2, 6, 3, 3)
        features = torch.randn(2, 6, 3, 4)
        output = reader.read_points(query, boxes, xyz, rgb, features)
        reader.competitor_scale.data.fill_(0.25)
        changed = reader.read_points(query, boxes, xyz, rgb, features)
        self.assertFalse(torch.equal(output, changed))
        reader.competitor_scale.data.zero_()
        restored = reader.read_points(query, boxes, xyz, rgb, features)
        self.assertTrue(torch.equal(output, restored))
        restored.square().sum().backward()
        self.assertTrue(torch.isfinite(reader.competitor_scale.grad))
        self.assertNotEqual(reader.competitor_scale.grad.item(), 0)


if __name__ == "__main__":
    unittest.main()
