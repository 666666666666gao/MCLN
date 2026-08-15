import unittest

import torch

from models.mcln import DecoderQueryTextAdapter


class DecoderQueryTextAdapterTest(unittest.TestCase):
    def _inputs(self, batch=2, queries=7, tokens=5, dim=24):
        generator = torch.Generator().manual_seed(132)
        query = torch.randn(batch, queries, dim, generator=generator)
        text = torch.randn(batch, tokens, dim, generator=generator)
        padding = torch.zeros(batch, tokens, dtype=torch.bool)
        padding[0, -1] = True
        centers = torch.randn(batch, queries, 3, generator=generator)
        sizes = torch.rand(batch, queries, 3, generator=generator) + 0.1
        return query, text, padding, centers, sizes

    def test_zero_initialized_identity_and_bound(self):
        model = DecoderQueryTextAdapter(
            d_model=24, hidden_dim=24, num_heads=4,
            dropout=0.0, max_delta=0.25,
        ).eval()
        inputs = self._inputs()
        output, residual = model(*inputs)
        self.assertTrue(torch.equal(output, inputs[0]))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

        with torch.no_grad():
            model.output.weight.fill_(100.0)
            model.output.bias.fill_(100.0)
        _output, residual = model(*inputs)
        self.assertLessEqual(float(residual.abs().max()), 0.25)

    def test_query_permutation_equivariance(self):
        model = DecoderQueryTextAdapter(
            d_model=24, hidden_dim=24, num_heads=4,
            dropout=0.0, max_delta=0.25,
        ).eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.output.weight, std=0.01)
        query, text, padding, centers, sizes = self._inputs()
        permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
        output, residual = model(query, text, padding, centers, sizes)
        permuted_output, permuted_residual = model(
            query[:, permutation], text, padding,
            centers[:, permutation], sizes[:, permutation],
        )
        self.assertTrue(torch.allclose(
            permuted_output, output[:, permutation], atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            permuted_residual, residual[:, permutation], atol=1e-6, rtol=1e-5
        ))

    def test_training_reaches_text_geometry_and_set_paths(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DecoderQueryTextAdapter(
            d_model=24, hidden_dim=24, num_heads=4,
            dropout=0.0, max_delta=0.25,
        ).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        query, text, padding, centers, sizes = [
            value.to(device) for value in self._inputs()
        ]
        target = 0.05 * torch.tanh(
            text[:, :1].expand(-1, query.shape[1], -1)
            + centers.mean(dim=-1, keepdim=True)
        )
        target = target[..., :query.shape[-1]]
        initial_loss = None
        for _ in range(24):
            optimizer.zero_grad(set_to_none=True)
            _output, residual = model(query, text, padding, centers, sizes)
            loss = torch.nn.functional.mse_loss(residual, target)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial_loss * 0.75)
        for parameter_name in (
            'text_attention.in_proj_weight',
            'geometry_encoder.0.weight',
            'set_attention.in_proj_weight',
        ):
            parameter = dict(model.named_parameters())[parameter_name]
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
