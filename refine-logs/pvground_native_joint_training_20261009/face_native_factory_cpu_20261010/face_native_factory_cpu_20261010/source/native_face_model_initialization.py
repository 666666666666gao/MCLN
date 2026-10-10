"""Source-only native factory extension; the retained prior loads strictly first."""
import json
from pathlib import Path

from native_model_initialization import configure_native_model
from face_residual_span_mixer import FaceResidualSpanMixer


def configure_native_face_model(model, spec_path, initialize_weights):
    spec = json.loads(Path(spec_path).read_bytes())
    assert spec['face_residual_mode'] in ('source_conditioned', 'without_additional_source')
    # Preserve the existing1234 ->1271 ->1295 strict official/G/support/span
    # initialization. Do not load14 old prior keys loosely into a20-key wrapper.
    model = configure_native_model(model, spec_path, initialize_weights)
    assert len(model.state_dict()) == 1295
    prior = model.candidate_span_mixer
    model.candidate_span_mixer = FaceResidualSpanMixer(prior,
        use_source_evidence=spec['face_residual_mode']=='source_conditioned')
    assert model.candidate_span_mixer.axis_prior is prior
    assert len(model.state_dict()) == 1301
    assert sum(parameter.numel() for parameter in model.candidate_span_mixer.face_residual.parameters()) == 23425
    assert sum(parameter.numel() for parameter in model.candidate_span_mixer.parameters()) == 53218
    model.native_training_architecture.update(
        span_parameters=53218,full_state_tensors=1301,
        retained_axis_prior_parameters=29793,face_residual_parameters=23425,
        face_residual_mode=spec['face_residual_mode'],
        prior_state_layout='candidate_span_mixer.axis_prior',
        direct_source_control_scope='Additional face residual evidence only; retained axis prior common',
        face_output_zero_initialized=True,face_size_parameterization='absolute size with native1e-6 minimum')
    return model
