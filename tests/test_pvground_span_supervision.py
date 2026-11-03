"""Run on the remote training server only; no dataset or model construction."""

import unittest

import torch

from models.pvground_span_supervision import span_gate_supervision


class SpanGateSupervisionTests(unittest.TestCase):
    def inputs(self):
        raw = torch.tensor([[[-0.5, 1.5, 0.25], [0.0, 0.0, 0.0]]],
                           requires_grad=True)
        center = torch.zeros(1, 2, 3, requires_grad=True)
        size = torch.ones(1, 2, 3, requires_grad=True)
        reference_center = torch.ones(1, 2, 3, requires_grad=True)
        reference_size = torch.full((1, 2, 3), 2.0, requires_grad=True)
        valid = torch.ones(1, 2, dtype=torch.bool)
        gt = torch.tensor([[0.5, 0.5, 0.5, 1.5, 1.5, 1.5]],
                          requires_grad=True)
        targets = [{"boxes": gt}]
        matches = [(torch.tensor([0]), torch.tensor([0]))]
        return [raw, center, size, reference_center, reference_size,
                valid, targets, matches]

    def test_axis_projection_and_outside_clamp_gradient(self):
        args = self.inputs()
        result = span_gate_supervision(*args)
        self.assertEqual(result["valid_axes"].item(), 3)
        self.assertAlmostEqual(result["target_mean"].item(), 0.5)
        result["loss"].backward()
        self.assertLess(args[0].grad[0, 0, 0].item(), 0)
        self.assertGreater(args[0].grad[0, 0, 1].item(), 0)
        self.assertEqual(args[0].grad[0, 1].abs().sum().item(), 0)
        for endpoint in args[1:5] + [args[6][0]["boxes"]]:
            self.assertIsNone(endpoint.grad)

    def test_matched_object_is_not_always_root(self):
        args = self.inputs()
        args[6][0]["boxes"] = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        ])
        args[7] = [(torch.tensor([1]), torch.tensor([1]))]
        result = span_gate_supervision(*args)
        self.assertAlmostEqual(result["target_mean"].item(), 1.0)
        result["loss"].backward()
        self.assertEqual(args[0].grad[0, 0].abs().sum().item(), 0)
        self.assertLess(args[0].grad[0, 1].max().item(), 0)

    def test_invalid_reference_excluded(self):
        args = self.inputs()
        args[5][:] = False
        result = span_gate_supervision(*args)
        self.assertEqual(result["valid_axes"].item(), 0)
        self.assertEqual(result["loss"].item(), 0)
        result["loss"].backward()
        self.assertEqual(args[0].grad.abs().sum().item(), 0)

    def test_identical_endpoints_excluded(self):
        args = self.inputs()
        args[3] = args[1].detach().clone()
        args[4] = args[2].detach().clone()
        result = span_gate_supervision(*args)
        self.assertEqual(result["valid_axes"].item(), 0)
        self.assertEqual(result["loss"].item(), 0)

    def test_empty_matches_have_zero_differentiable_loss(self):
        args = self.inputs()
        args[7] = [(torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long))]
        result = span_gate_supervision(*args)
        result["loss"].backward()
        self.assertEqual(result["loss"].item(), 0)
        self.assertEqual(args[0].grad.abs().sum().item(), 0)

    def test_axis_specific_target(self):
        args = self.inputs()
        args[6][0]["boxes"] = torch.tensor([
            [0.0, 0.5, 1.0, 1.0, 1.5, 2.0],
        ])
        args[0] = torch.tensor([[[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]],
                               requires_grad=True)
        self.assertEqual(span_gate_supervision(*args)["loss"].item(), 0)


if __name__ == "__main__":
    unittest.main()
