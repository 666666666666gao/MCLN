"""Remote-only checks of real optimizer/mode methods without dataset imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


def trainer_method(name):
    source = ast.parse(Path("main_utils.py").read_text())
    trainer = next(node for node in source.body
                   if isinstance(node, ast.ClassDef) and node.name == "BaseTrainTester")
    method = next(node for node in trainer.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "main_utils.py", "exec"), namespace)
    return namespace[name]


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_net = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(.5))
        self.reader = nn.Linear(3, 3)
        self.pvground_support_corrector = nn.Linear(3, 3)
        self.pvground_span_mixer = nn.Linear(3, 3)

    def forward(self, inputs):
        features = self.pvground_support_corrector(self.reader(self.backbone_net(inputs)))
        return self.pvground_span_mixer(features)


class SpanOnlyFreezingTest(unittest.TestCase):
    def test_actual_mode_and_optimizer_preserve_all_upstream_state(self):
        model = SmallModel()
        args = SimpleNamespace(
            pvground_train_only=False, pvground_span_train_only=True,
            candidate_local_visual_lr=2e-5, weight_decay=.0005,
            lr_backbone=4e-6, text_encoder_lr=0.,
        )
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = trainer_method("get_optimizer")(args, model)
        trainer_method("_set_source_moe_train_mode")(model, args)
        self.assertTrue(model.training)
        self.assertTrue(model.pvground_span_mixer.training)
        for module in (model.backbone_net, model.reader, model.pvground_support_corrector):
            self.assertFalse(module.training)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, {"pvground_span_mixer.weight", "pvground_span_mixer.bias"})
        optimizer.zero_grad()
        loss = model(torch.ones(8, 3)).square().mean()
        loss.backward()
        optimizer.step()
        after = model.state_dict()
        for name in initial:
            if not name.startswith("pvground_span_mixer."):
                self.assertTrue(torch.equal(initial[name], after[name]), name)
        self.assertTrue(any(not torch.equal(initial[name], after[name]) for name in trainable))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
