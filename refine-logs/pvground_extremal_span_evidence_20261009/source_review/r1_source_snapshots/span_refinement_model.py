"""One frozen PV parent and one trainable span decoder, shared by train and inference."""
import torch
from torch import nn

from extremal_span_mixer import ExtremalSpanMixer


def apply_span_mixer(mixer, predictions, geometries):
    center, size, evidence = mixer(predictions['support_geometry_query'],
                                   predictions['support_super_features'], predictions, geometries)
    final = dict(predictions)
    final['span_mask_center'] = predictions['last_center'].detach()
    final['span_mask_size'] = predictions['last_pred_size'].detach()
    final['last_center'], final['last_pred_size'] = center, size
    final['span_axis_gate'] = torch.stack([item['axis_gate'] for item in evidence])
    final['span_raw_axis_gate'] = torch.stack([item['raw_axis_gate'] for item in evidence])
    final['span_source_fraction'] = torch.stack([item['source_fraction'] for item in evidence])
    assert final['last_sem_cls_scores'] is predictions['last_sem_cls_scores']
    assert final['sp_last_pred_masks'] is predictions['sp_last_pred_masks']
    assert final['last_pred_masks'] is predictions['last_pred_masks']
    return final


class SpanRefinementModel(nn.Module):
    def __init__(self, parent, source_mode):
        super().__init__()
        assert parent.candidate_support_corrector is not None
        assert parent.candidate_box_refiner.reference_mode == 'fused_mask'
        assert torch.count_nonzero(parent.candidate_box_refiner.output.weight) == 0
        assert torch.count_nonzero(parent.candidate_box_refiner.output.bias) == 0
        assert torch.count_nonzero(parent.boundary_evidence_readback.output.weight) == 0
        assert torch.count_nonzero(parent.boundary_evidence_readback.output.bias) == 0
        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.mixer = ExtremalSpanMixer(source_mode)
        assert sum(parameter.numel() for parameter in self.mixer.parameters()) == 29793
        assert len(self.mixer.state_dict()) == 14
        self.eval()

    def forward(self, **inputs):
        with torch.no_grad():
            predictions = self.parent(**inputs)
        return apply_span_mixer(self.mixer, predictions, predictions['support_member_geometry'])
