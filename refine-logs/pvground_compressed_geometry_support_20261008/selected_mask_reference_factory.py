"""Strict reconstruction of the declared Mask-reference architecture and delta."""
import copy

import torch

from models.pv_ground import PVGround
from pvground_task_observation_query import install_task_observation_query_read
from pvground_boundary_box_refiner import install_boundary_refinement
from install_boundary_evidence_readback import install_boundary_evidence_readback
from mask_reference import install_mask_reference


def build_selected_mask_reference_model(cfg, official_payload, g_payload, selected_payload, data_root):
    """Load actual step0 or step3723 state without requiring the old geometry file."""
    assert selected_payload['step'] in (0, 3723)
    assert selected_payload['reference_mode'] in ('native', 'fused_mask')
    assert selected_payload['common_output_reset'] == ['output.weight', 'output.bias']
    assert selected_payload['retained_hidden_prior_updates'] == 11169
    config = official_payload['config']
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_soft_token_loss and config.use_contrastive_align
    initial = {name[7:]: value for name, value in official_payload['model'].items()}
    assert all(name.startswith('module.') for name in official_payload['model']) and len(initial) == 1234
    model = PVGround(copy.deepcopy(cfg), num_class=256, num_queries=256, num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding, contrastive_align_loss=True,
        butd=True, pointnet_ckpt=None, data_path=data_root, self_attend=config.self_attend)
    position_ids = torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1))
    assert set(model.state_dict()) - set(initial) == {'text_encoder.embeddings.position_ids'}
    assert not set(initial) - set(model.state_dict())
    assert torch.equal(model.text_encoder.embeddings.position_ids, position_ids)
    model.text_encoder.embeddings.register_buffer('position_ids',
        model.text_encoder.embeddings.position_ids, persistent=False)
    model.load_state_dict(initial, strict=True)
    install_task_observation_query_read(model)
    reader_keys = set(model.state_dict()) - set(initial)
    assert len(reader_keys) == 37 and all(name.startswith('decoder.5.source_query_read.') for name in reader_keys)
    initial.update({name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in reader_keys})
    expected_g = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_g.update(name for name, _ in model.named_buffers() if name in initial)
    assert set(g_payload['state_delta']) == expected_g and len(expected_g) == 1072
    for name, value in g_payload['state_delta'].items():
        assert value.shape == initial[name].shape and value.dtype == initial[name].dtype
    initial.update(g_payload['state_delta'])
    model.load_state_dict(initial, strict=True)

    # Keep the original construction order, including the frozen zero-output R.
    # Loading geometry tensors changes no RNG state; the old geometry file is
    # unnecessary once this candidate supplies every geometry tensor itself.
    install_boundary_refinement(model, 'distribution')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    install_boundary_evidence_readback(model, True)
    install_mask_reference(model, selected_payload['reference_mode'])
    geometry_names = {name for name, _ in model.named_parameters() if name.startswith('candidate_box_refiner.')}
    assert len(geometry_names) == 10 and set(selected_payload['state_delta']) == geometry_names
    assert sum(parameter.numel() for parameter in model.candidate_box_refiner.parameters()) == 456102
    state = model.state_dict()
    for name, value in selected_payload['state_delta'].items():
        assert value.shape == state[name].shape and value.dtype == state[name].dtype
    state.update(selected_payload['state_delta'])
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in selected_payload['state_delta'].items())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.candidate_box_refiner.parameters():
        parameter.requires_grad_(True)
    model.eval()
    return model, config, dict(geometry_state_tensors=10, geometry_parameters=456102,
        reference_mode=selected_payload['reference_mode'], actual_checkpoint_step=selected_payload['step'],
        old_geometry_checkpoint_required=False, native_final_semantic_head_deferred=True,
        full_state_tensors=len(model.state_dict()), parent_and_R_frozen=True)
