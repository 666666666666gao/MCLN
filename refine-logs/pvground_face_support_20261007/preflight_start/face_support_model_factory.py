"""Rebuild one declared sampler with the existing protected parent and head."""
import torch

from selected_mask_reference_factory import build_selected_mask_reference_model
from face_region_box_refiner import install_face_region_reading


def build_face_support_model(cfg, official_payload, g_payload, reference_payload,
                             arm_payload, data_root):
    assert arm_payload['sampling_mode'] in ('face_center', 'face_region')
    assert arm_payload['step'] in (0, 2, 3723)
    model, config, receipt = build_selected_mask_reference_model(
        cfg, official_payload, g_payload, reference_payload, data_root)
    install_face_region_reading(model, arm_payload['sampling_mode'])
    names = {'candidate_box_refiner.' + name for name in model.candidate_box_refiner.state_dict()}
    assert set(arm_payload['state_delta']) == names and len(names) == 10
    state = model.state_dict()
    for name, value in arm_payload['state_delta'].items():
        assert value.shape == state[name].shape and value.dtype == state[name].dtype
    state.update(arm_payload['state_delta'])
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in arm_payload['state_delta'].items())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.candidate_box_refiner.parameters():
        parameter.requires_grad_(True)
    model.eval()
    receipt.update(sampling_mode=model.candidate_box_refiner.sampling_mode,
                   arm_optimizer_step=arm_payload['step'], deployed_geometry_heads=1)
    return model, config, receipt
