"""Explicit frozen4506-provider factory draft; no CPU/GPU construction yet."""
import copy
import torch

from models.pv_ground import PVGround
from pvground_task_observation_query import install_task_observation_query_read
from pvground_boundary_box_refiner import install_boundary_refinement
from install_boundary_evidence_readback import install_boundary_evidence_readback


def build_readback_model(cfg, official_payload, g_payload, geometry_payload, data_root,
                         use_geometry_evidence):
    """Call only after the future runner verifies all three file identities."""
    assert isinstance(use_geometry_evidence, bool)
    config = official_payload['config']
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_soft_token_loss and config.use_contrastive_align
    initial = {name[7:]: value for name, value in official_payload['model'].items()}
    assert all(name.startswith('module.') for name in official_payload['model'])
    assert len(initial) == 1234
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
    assert len(reader_keys) == 37
    assert all(name.startswith('decoder.5.source_query_read.') for name in reader_keys)
    initial.update({name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                    if name in reader_keys})
    expected_g_delta = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_g_delta.update(name for name, _ in model.named_buffers() if name in initial)
    assert set(g_payload['state_delta']) == expected_g_delta and len(expected_g_delta) == 1072
    for name, value in g_payload['state_delta'].items():
        assert value.shape == initial[name].shape and value.dtype == initial[name].dtype
    initial.update(g_payload['state_delta'])
    model.load_state_dict(initial, strict=True)

    install_boundary_refinement(model, 'distribution')
    geometry_keys = set(model.state_dict()) - set(initial)
    assert len(geometry_keys) == 10
    assert all(name.startswith('candidate_box_refiner.') for name in geometry_keys)
    assert sum(parameter.numel() for parameter in model.candidate_box_refiner.parameters()) == 456102
    assert geometry_payload['step'] == 3723 and geometry_payload['head_only']
    assert geometry_payload['boundary_mode'] == 'distribution'
    assert geometry_payload['support_arm'] == 'whole_range' and geometry_payload['use_whole_range']
    assert geometry_payload['boundary_loss_weight'] == 1.0 / 7
    assert set(geometry_payload['state_delta']) == geometry_keys
    state = model.state_dict()
    for name, value in geometry_payload['state_delta'].items():
        assert value.shape == state[name].shape and value.dtype == state[name].dtype
    initial.update(geometry_payload['state_delta'])
    model.load_state_dict(initial, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    install_boundary_evidence_readback(model, use_geometry_evidence)
    readback_keys = set(model.state_dict()) - set(initial)
    assert len(readback_keys) == 23
    assert all(name.startswith('boundary_evidence_readback.') for name in readback_keys)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == readback_keys
    readback_parameters = sum(parameter.numel() for parameter in model.boundary_evidence_readback.parameters())
    assert readback_parameters == 96672
    model.eval()
    model.boundary_evidence_readback.train()
    return model, config, initial, dict(g_delta_tensors=1072, geometry_state_tensors=10,
        geometry_parameters=456102, readback_state_tensors=23, readback_parameters=readback_parameters,
        use_geometry_evidence=use_geometry_evidence, frozen_geometry_provider=True,
        native_final_semantic_head_deferred=True, fresh_readback_optimizer_required=True)
