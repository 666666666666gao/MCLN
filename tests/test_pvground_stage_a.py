import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def methods():
    path = Path(__file__).resolve().parents[1] / "main_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "BaseTrainTester")
    tree.body = [node for node in trainer.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("get_optimizer", "_set_source_moe_train_mode")]
    for node in tree.body:
        node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        self.decoder = torch.nn.ModuleList([torch.nn.Module() for _ in range(6)])
        self.decoder[5].local_visual = torch.nn.Linear(3, 3)
        self.pvground_support_corrector = torch.nn.Linear(3, 3)
        self.pvground_span_mixer = torch.nn.Linear(3, 3)

    def forward(self, value):
        value = self.native(value)
        return (self.decoder[5].local_visual(value)
                + self.pvground_support_corrector(value)
                + self.pvground_span_mixer(value))


def test_only_modules_update_and_native_bn_stays_fixed():
    model = Fixture()
    options = SimpleNamespace(pvground_train_only=True, candidate_local_visual_lr=1e-4,
                              lr_backbone=0.0, text_encoder_lr=0.0, weight_decay=0.0005)
    functions = methods()
    optimizer = functions["get_optimizer"](options, model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    functions["_set_source_moe_train_mode"](model, options)
    assert model.training and not model.native.training
    assert not model.native[1].training
    assert len(optimizer.param_groups) == 3
    model(torch.randn(8, 3)).square().mean().backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        if name.startswith("native."):
            assert torch.equal(value, before[name])
        else:
            assert not torch.equal(value, before[name])
