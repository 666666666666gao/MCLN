import pytest
import torch

from models.candidate_local_visual import CandidateLocalVisual
from models.candidate_local_visual_training import (
    validate_local_visual_checkpoint,
)
from models.pvground_innovations import (
    CandidateMaskSupportCorrector,
    ExtremalSpanMixer,
    mask_extent_reference,
)


def test_reader_zero_output_and_output_projection_gradient():
    torch.manual_seed(3)
    reader = CandidateLocalVisual()
    query = torch.randn(1, 256, 288)
    boxes = torch.zeros(1, 256, 6)
    boxes[..., 3:] = 2
    xyz = torch.randn(1, 256, 4, 3)
    rgb = torch.randn(1, 256, 4, 3)
    features = torch.randn(1, 256, 4, 128)

    residual = reader.read_points(query, boxes, xyz, rgb, features)

    assert torch.equal(residual, torch.zeros_like(residual))
    residual.sum().backward()
    assert reader.output_projection.weight.grad.abs().sum() > 0
    assert reader.output_projection.bias.grad.abs().sum() > 0


def test_support_and_span_start_as_native_identity_and_receive_gradients():
    torch.manual_seed(7)
    point_xyz = torch.stack([
        torch.linspace(0, 3, 24),
        torch.linspace(1, 4, 24),
        torch.linspace(-2, 2, 24),
    ], dim=1)
    raw_points = torch.cat([point_xyz, torch.zeros(24, 3)], dim=1).unsqueeze(0)
    superpoints = torch.arange(24).remainder(4).unsqueeze(0)
    query = torch.randn(1, 256, 288)
    supports = [torch.randn(288, 4)]
    native_center = torch.zeros(1, 256, 3)
    native_size = torch.full((1, 256, 3), 2.0)
    text_masks = torch.tensor([-4.0, -4.0, 4.0, 4.0]).expand(1, 256, 4)
    own_masks = torch.tensor([-4.0, -4.0, 4.0, 4.0]).expand(1, 256, 4)
    alpha = [torch.tensor(0.5)]
    predictions = {
        "last_pred_masks": [text_masks],
        "sp_last_pred_masks": [own_masks[0]],
        "adaptive_weights": alpha,
        "superpoints": superpoints,
    }
    corrector = CandidateMaskSupportCorrector()
    corrected, geometries = corrector(
        query, supports, raw_points, native_center, native_size, predictions
    )

    assert torch.equal(corrected[0], own_masks[0])
    corrected[0].sum().backward(retain_graph=True)
    assert corrector.output.weight.grad.abs().sum() > 0
    assert corrector.output.bias.grad.abs().sum() > 0

    mask_center, mask_size, mask_valid = mask_extent_reference(
        native_center,
        native_size,
        text_masks,
        torch.stack(corrected),
        torch.stack(alpha),
        geometries,
    )
    assert mask_valid.all()
    span_inputs = {
        "native_coarse_center": native_center,
        "native_coarse_size": native_size,
        "last_center": native_center,
        "last_pred_size": native_size,
        "last_pred_masks": predictions["last_pred_masks"],
        "sp_last_pred_masks": corrected,
        "adaptive_weights": alpha,
        "mask_reference_valid": mask_valid,
        "mask_reference_center": mask_center,
        "mask_reference_size": mask_size,
    }
    span = ExtremalSpanMixer()
    center, size, evidence = span(query, supports, span_inputs, geometries)

    assert torch.equal(center, native_center)
    assert torch.equal(size, native_size)
    assert evidence[0]["axis_gate"].shape == (256, 3)
    (center - mask_center).sum().backward()
    assert span.output.bias.grad.abs().sum() > 0
    assert span.output.weight.grad.abs().sum() > 0
    assert mask_size.shape == native_size.shape


def test_model_only_checkpoint_allows_only_three_new_module_prefixes():
    current = {
        "module.backbone_net.weight": torch.ones(2, 2),
        "module.decoder.5.local_visual.weight": torch.ones(2, 2),
        "module.pvground_support_corrector.weight": torch.ones(2, 2),
        "module.pvground_span_mixer.weight": torch.ones(2, 2),
    }
    saved = {"module.backbone_net.weight": torch.ones(2, 2)}

    validate_local_visual_checkpoint(
        current,
        saved,
        model_only_initialization=True,
        additional_new_prefixes=(
            "pvground_support_corrector.",
            "pvground_span_mixer.",
        ),
    )

    with pytest.raises(ValueError, match="key mismatch"):
        validate_local_visual_checkpoint(
            current,
            {**saved, "module.unexpected.weight": torch.ones(2, 2)},
            model_only_initialization=True,
            additional_new_prefixes=(
                "pvground_support_corrector.",
                "pvground_span_mixer.",
            ),
        )
