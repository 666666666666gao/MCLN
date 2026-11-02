from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from eval_scanrefer_restart_e0 import initialize_evaluation_model


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([float(index)])) for index in range(1134)
        ])
        self.decoder = torch.nn.ModuleList([torch.nn.Module() for _ in range(6)])
        self.decoder[5].local_visual = torch.nn.Linear(1, 1)
        self.pvground_support_corrector = torch.nn.Linear(1, 1)
        self.pvground_span_mixer = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1)) for _ in range(30)
        ])


def payload(model, official):
    return {
        "epoch": 54 if official else 2,
        "model": {
            "module." + name: value.clone() + 3
            for name, value in model.state_dict().items()
            if not official or name.startswith("native.")
        },
    }


@pytest.mark.parametrize("official", [True, False])
def test_official_identity_or_full_trained_weights_load_exactly(tmp_path, official):
    model = Fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint = payload(model, official)
    path = tmp_path / "weight.pth"
    torch.save(checkpoint, path)
    settings = SimpleNamespace(checkpoint_path=str(path))
    receipt = initialize_evaluation_model(
        settings, SimpleNamespace(module=model), checkpoint["epoch"], 1134 if official else 1168,
    )
    assert receipt["native_tensors"] == 1134 and not receipt["optimizer_restored"]
    assert receipt["new_tensors"] == (34 if official else 0)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name] + (
            3 if not official or name.startswith("native.") else 0
        ))


@pytest.mark.parametrize("fault", ["epoch", "missing_trained_module", "shape", "dtype"])
def test_invalid_trained_checkpoint_rejected_before_any_state_change(tmp_path, fault):
    model = Fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint = payload(model, False)
    key = "module.native.0"
    if fault == "epoch":
        checkpoint["epoch"] = 1
    elif fault == "missing_trained_module":
        del checkpoint["model"]["module.pvground_span_mixer.0"]
    elif fault == "shape":
        checkpoint["model"][key] = torch.zeros(2)
    else:
        checkpoint["model"][key] = checkpoint["model"][key].double()
    path = tmp_path / "invalid.pth"
    torch.save(checkpoint, path)
    with pytest.raises(AssertionError):
        initialize_evaluation_model(
            SimpleNamespace(checkpoint_path=str(path)), SimpleNamespace(module=model), 2, 1168,
        )
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())
