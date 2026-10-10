"""Configure the native PV model, keeping its original trainable core policy."""
import hashlib
import json
from pathlib import Path

import torch

from extremal_span_mixer import ExtremalSpanMixer
from mask_support_corrector import CandidateMaskSupportCorrector
from pvground_task_observation_query import install_task_observation_query_read


def checked_payload(path, digest):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    return torch.load(path, map_location='cpu')


def configure_native_model(model, spec_path, initialize_weights):
    spec = json.loads(Path(spec_path).read_bytes())
    assert spec['dataset'] == 'scanrefer' and spec['seed'] == 2027
    assert spec['span_source_mode'] in ('whole_support', 'extremal_support')
    assert model.candidate_box_refiner is None and model.boundary_evidence_readback is None
    assert model.candidate_support_corrector is None and model.candidate_span_mixer is None
    original_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    position_ids = torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1))
    assert torch.equal(model.text_encoder.embeddings.position_ids, position_ids)
    model.text_encoder.embeddings.register_buffer('position_ids', position_ids, persistent=False)
    assert len(model.state_dict()) == 1234
    if initialize_weights:
        official = checked_payload(spec['official_checkpoint'], spec['official_checkpoint_sha256'])
        initial = {name[7:]: value for name, value in official['model'].items()}
        assert all(name.startswith('module.') for name in official['model'])
        model.load_state_dict(initial, strict=True)
    install_task_observation_query_read(model)
    assert len(model.state_dict()) == 1271
    if initialize_weights:
        parent_g = checked_payload(spec['g_checkpoint'], spec['g_checkpoint_sha256'])
        state = model.state_dict()
        assert len(parent_g['state_delta']) == 1072
        for name, value in parent_g['state_delta'].items():
            assert name in state and state[name].shape == value.shape and state[name].dtype == value.dtype
        state.update(parent_g['state_delta'])
        model.load_state_dict(state, strict=True)
    model.candidate_support_corrector = CandidateMaskSupportCorrector(False)
    assert sum(parameter.numel() for parameter in model.candidate_support_corrector.parameters()) == 27841
    if initialize_weights:
        support = checked_payload(spec['support_checkpoint'], spec['support_checkpoint_sha256'])
        assert support['arm'] == 'selected_query' and support['geometry_encoding'] == 'zero'
        prefix = 'candidate_support_corrector.'
        assert len(support['state_delta']) == 10
        assert all(name.startswith(prefix) for name in support['state_delta'])
        model.candidate_support_corrector.load_state_dict(
            {name[len(prefix):]: value for name, value in support['state_delta'].items()}, strict=True)
    model.candidate_span_mixer = ExtremalSpanMixer(spec['span_source_mode'])
    if initialize_weights and spec['span_checkpoint'] is not None:
        span = checked_payload(spec['span_checkpoint'], spec['span_checkpoint_sha256'])
        assert span['source_mode'] == spec['span_source_mode']
        assert span['parent_identity']['parent_support_terminal_sha256'] == spec['support_checkpoint_sha256']
        model.candidate_span_mixer.load_state_dict(span['mixer_state'], strict=True)
    assert len(model.state_dict()) == 1295
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if name in original_trainable)
    assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
    model.use_predicted_mask_reference = True
    model.use_g_supervision = spec['use_g_supervision']
    model.use_selected_mask_supervision = spec['use_selected_mask_supervision']
    model.native_training_architecture = dict(
        model='PVGround', support_parameters=27841, span_parameters=29793,
        span_source_mode=spec['span_source_mode'], full_state_tensors=1295,
        inactive_geometry_and_R_parameters_retained=False,
        original_core_trainability_preserved=True, text_encoder_frozen_by_native_policy=True,
        mask_reference_is_discrete=True, separate_deployed_ranking=False,
        use_g_supervision=model.use_g_supervision,
        use_selected_mask_supervision=model.use_selected_mask_supervision,
        dataset='scanrefer', seed=2027)
    return model
