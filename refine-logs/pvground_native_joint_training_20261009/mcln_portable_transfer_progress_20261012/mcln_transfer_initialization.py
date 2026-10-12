"""Strict E71 core plus exact, portable current-best PV support/range states."""
import hashlib
from pathlib import Path

import torch

from mask_support_corrector import CandidateMaskSupportCorrector
from extremal_span_mixer import ExtremalSpanMixer


def checked_payload(path, digest):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    assert h.hexdigest() == digest
    return torch.load(path, map_location='cpu')


def load_mcln_core(model, payload):
    assert all(name.startswith('module.') for name in payload['model'])
    source = {name[7:]: value for name, value in payload['model'].items()}
    selector = [name for name in source if name.startswith('source_choice_selector.')]
    assert len(selector) == 9
    for name in selector: del source[name]
    assert len(source) == 1135 and set(source) == set(model.state_dict())
    assert model.source_choice_selector is None and model.source_moe is None
    model.load_state_dict(source, strict=True)
    assert all(torch.equal(value.detach().cpu(), source[name]) for name, value in model.state_dict().items())


def install_portable_states(model, spec):
    assert model.portable_geometry_arm == 'native'
    assert model.candidate_support_corrector is None and model.candidate_span_mixer is None
    model.candidate_support_corrector = CandidateMaskSupportCorrector(False)
    support = checked_payload(spec['support_checkpoint'], spec['support_checkpoint_sha256'])
    assert support['arm'] == 'selected_query' and support['geometry_encoding'] == 'zero'
    prefix = 'candidate_support_corrector.'
    assert len(support['state_delta']) == 10 and all(name.startswith(prefix) for name in support['state_delta'])
    model.candidate_support_corrector.load_state_dict({name[len(prefix):]: value for name, value in support['state_delta'].items()}, strict=True)
    model.candidate_span_mixer = ExtremalSpanMixer('extremal_support')
    span = checked_payload(spec['span_checkpoint'], spec['span_checkpoint_sha256'])
    assert span['source_mode'] == 'extremal_support'
    assert span['parent_identity']['parent_support_terminal_sha256'] == spec['support_checkpoint_sha256']
    model.candidate_span_mixer.load_state_dict(span['mixer_state'], strict=True)
    assert sum(p.numel() for p in model.candidate_support_corrector.parameters()) == 27841
    assert sum(p.numel() for p in model.candidate_span_mixer.parameters()) == 29793
    assert len(model.state_dict()) == 1159

