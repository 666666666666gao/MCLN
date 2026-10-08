"""Rebuild the protected model and optionally install one deployable Mask head."""
from mask_support_corrector import install_support_correction
from selected_mask_reference_factory import build_selected_mask_reference_model


ARMS = ('content', 'box_conditioned')


def build_support_model(cfg, official, original_g, selected, data_root, payload=None):
    model, config, receipt = build_selected_mask_reference_model(
        cfg, official, original_g, selected, data_root)
    assert receipt['full_state_tensors'] == 1304
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.candidate_support_corrector is None
    if payload is not None:
        assert payload['arm'] in ARMS and payload['reference_mode'] == 'fused_mask'
        assert payload['geometry_encoding'] == ('signed_log' if payload['arm'] == 'box_conditioned' else 'zero')
        assert payload['mask_loss_coefficients'] == [5, 1, 10, 2]
        prefix = 'candidate_support_corrector.'
        assert len(payload['state_delta']) == 10
        assert all(name.startswith(prefix) for name in payload['state_delta'])
        state = {name[len(prefix):]: value for name, value in payload['state_delta'].items()}
        install_support_correction(model, payload['arm'] == 'box_conditioned', state)
        receipt.update(full_state_tensors=len(model.state_dict()),
            added_parameters=27841, added_state_tensors=10, deployed_support_heads=1,
            explicit_box_geometry=payload['arm'] == 'box_conditioned')
        assert receipt['full_state_tensors'] == 1314
    model.eval()
    return model, config, receipt
