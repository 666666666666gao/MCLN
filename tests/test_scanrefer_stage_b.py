import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from train_scanrefer_stage_b import (
    MODULE_PREFIXES, ValidationHistory, build_optimizer, semantic_row, training_arguments,
)


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = torch.nn.Linear(3, 3)
        self.backbone_net = torch.nn.Linear(3, 3)
        self.text_encoder = torch.nn.Linear(3, 3)
        self.decoder = torch.nn.ModuleList([torch.nn.Module() for _ in range(6)])
        self.decoder[5].local_visual = torch.nn.Linear(3, 3)
        self.pvground_support_corrector = torch.nn.Linear(3, 3)
        self.pvground_span_mixer = torch.nn.Linear(3, 3)

    def forward(self, value):
        return (
            self.native(value) + self.backbone_net(value)
            + self.decoder[5].local_visual(value)
            + self.pvground_support_corrector(value) + self.pvground_span_mixer(value)
        )


def test_joint_optimizer_unfreezes_native_and_updates_all_three_modules():
    torch.manual_seed(0)
    model = Fixture()
    model.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    args = SimpleNamespace(lr=2e-6, lr_backbone=4e-6,
                           candidate_local_visual_lr=2e-5, weight_decay=0.0005)
    optimizer = build_optimizer(args, model)
    assert not optimizer.state
    assert [(group["name"], group["lr"]) for group in optimizer.param_groups] == [
        ("native", 2e-6), ("backbone", 4e-6), ("pvground", 2e-5),
    ]
    module_names = optimizer.param_groups[2]["parameter_names"]
    assert all(any(name.startswith(prefix) for name in module_names)
               for prefix in MODULE_PREFIXES)
    assert all(name.startswith(MODULE_PREFIXES) for name in module_names)
    loss = model(torch.randn(8, 3)).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        if name.startswith("text_encoder."):
            assert torch.equal(value, before[name])
        else:
            assert not torch.equal(value, before[name])


def test_semantic_row_uses_bbf_and_rejects_partial_validation():
    evaluator = SimpleNamespace(
        gts={("last_", threshold, 1, branch): 9508
             for threshold in (0.25, 0.5) for branch in ("bbf", "bbs")},
        dets={("last_", threshold, 1, branch): hits
              for threshold, hits in ((0.25, 5705), (0.5, 4945))
              for branch in ("bbf", "bbs")},
    )
    evaluator.dets[("last_", 0.5, 1, "bbs")] = 100
    assert semantic_row(evaluator, 1)["hits050"] == 4945
    evaluator.gts[("last_", 0.5, 1, "bbf")] = 9507
    with pytest.raises(AssertionError):
        semantic_row(evaluator, 1)


def test_equal_or_improving_full_epoch_resets_decrease_streak():
    history = ValidationHistory()
    for epoch, hits050 in enumerate((4400, 4399, 4399, 4398, 4397), start=1):
        stop = history.observe({
            "epoch": epoch, "sample_count": 9508,
            "hits025": 5500, "hits050": hits050,
        })
        assert stop == (epoch == 5)


def test_stage_b_arguments_reset_optimizer_without_backbone_reload(tmp_path):
    argv = training_arguments(tmp_path, {"initialization_checkpoint": "selected.pth"})
    assert "--model_only_initialization" in argv and "--pvground_train_only" not in argv
    assert "--pp_checkpoint" not in argv
    assert argv[argv.index("--checkpoint_path") + 1] == "selected.pth"
    assert argv[argv.index("--max_epoch") + 1] == "10"


def test_native_validation_hook_stops_before_next_train_and_skips_extra_eval(tmp_path):
    path = Path(__file__).parents[1] / "main_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "BaseTrainTester")
    main = next(node for node in base.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    loop = next(node for node in main.body if isinstance(node, ast.For)
                and ast.unparse(node.target) == "epoch")
    validation = next(node for node in loop.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "epoch % args.val_freq == 0")
    final_eval = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) ==
                      "not stopped_early and last_eval_epoch != args.max_epoch")
    loop.body = [validation]
    module = ast.Module(body=[loop, final_eval], type_ignores=[])
    history = ValidationHistory()
    calls = []

    class Runner:
        logger = SimpleNamespace(info=lambda *args: None)

        def evaluate_one_epoch(self, epoch, *args):
            calls.append(epoch)
            return None

        def should_stop_after_epoch(self, epoch, metrics, args):
            return history.observe({
                "epoch": epoch, "sample_count": 9508,
                "hits025": 5500, "hits050": 4401 - epoch,
            })

    namespace = {
        "args": SimpleNamespace(start_epoch=1, max_epoch=10, val_freq=1, log_dir=tmp_path),
        "self": Runner(), "metric_retention": True, "stopped_early": False,
        "last_eval_epoch": None, "test_loader": None, "model": None,
        "criterion": None, "set_criterion": None, "retained_metrics": (),
        "dist": SimpleNamespace(get_rank=lambda: 0),
        "save_eval_metrics_receipt": lambda *args: None,
        "update_checkpoint_retention": lambda *args, **kwargs: {"manifest": {"best": {}}},
    }
    exec(compile(module, str(path), "exec"), namespace)
    assert calls == [1, 2, 3] and namespace["stopped_early"]
