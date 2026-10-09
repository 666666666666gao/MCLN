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
    assert spec['dataset'] in ('nr3d', 'sr3d') and spec['seed'] == 2027
    assert spec['new_module_initialization'] == 'fresh'
    assert spec['g_checkpoint'] is None and spec['support_checkpoint'] is None and spec['span_checkpoint'] is None
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
    model.candidate_support_corrector = CandidateMaskSupportCorrector(False)
    assert sum(parameter.numel() for parameter in model.candidate_support_corrector.parameters()) == 27841
    model.candidate_span_mixer = ExtremalSpanMixer(spec['span_source_mode'])
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
        dataset=spec['dataset'], seed=2027,
        pretrained_core='corresponding_author_checkpoint', new_module_initialization='fresh',
        scanrefer_core_or_module_state_loaded=False)
    return model
