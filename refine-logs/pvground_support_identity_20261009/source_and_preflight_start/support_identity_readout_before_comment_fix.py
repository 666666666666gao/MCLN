"""Read candidate Mask-supported visual content into the one native semantic head."""
import torch
from torch import nn
from torch.nn import functional as F


class SupportIdentityReadout(nn.Module):
    def __init__(self, candidate_specific):
        super().__init__()
        self.candidate_specific = candidate_specific
        self.query_projection = nn.Linear(288,64)
        self.support_projection = nn.Linear(288,64)
        self.compatibility = nn.Sequential(nn.Linear(256,128),nn.GELU())
        self.output = nn.Linear(128,288)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, semantic_query, text, text_padding_mask, raw_points, end_points):
        batch,candidates,width = semantic_query.shape
        assert candidates == 256 and width == 288
        assert len(end_points['support_super_features']) == batch
        assert len(end_points['support_member_geometry']) == batch
        pooled = []
        support_mass = []
        for bid in range(batch):
            own = end_points['sp_last_pred_masks'][bid].detach()
            shared = end_points['last_pred_masks'][bid][0].detach()
            alpha = end_points['adaptive_weights'][bid].detach()
            features = end_points['support_super_features'][bid].detach()
            geometry = end_points['support_member_geometry'][bid]
            slots = torch.as_tensor(geometry['native_ids'],device=own.device,dtype=torch.long)
            count = own.new_tensor(geometry['count'])
            assert own.shape == shared.shape and own.shape[0] == 256 and alpha.ndim == 0
            assert features.shape == (288,own.shape[1]) and count.shape == slots.shape
            assert int(count.sum().item()) == 50000
            # The two arms differ only in whether the deployed candidate-specific
            # fused foreground is visible or the common Text foreground is used.
            logits = alpha*shared+(1-alpha)*own if self.candidate_specific else shared
            foreground = (logits[:,slots].sigmoid() > .5).to(features.dtype)
            weight = foreground*count[None]
            mass = weight.sum(-1,keepdim=True)
            # Empty foreground is observed in40 selected current validation rows.
            # A zero numerator stays zero; no alternative score or region is used.
            pooled.append((weight @ features[:,slots].T)/mass.clamp_min(1))
            support_mass.append(mass[:,0])
        visual = torch.stack(pooled)
        q = F.gelu(self.query_projection(semantic_query))
        v = F.gelu(self.support_projection(visual))
        compatibility = torch.cat((q,v,q*v,q-v),dim=-1)
        residual = self.output(self.compatibility(compatibility))
        end_points['identity_support_mass'] = torch.stack(support_mass).detach()
        end_points['identity_supported_visual_content'] = visual.detach()
        return semantic_query+residual


def install_support_identity_readout(model,candidate_specific,state=None):
    assert model.candidate_support_corrector is not None
    assert not model.candidate_support_corrector.use_box_geometry
    assert model.candidate_box_refiner.reference_mode == 'fused_mask'
    old = model.boundary_evidence_readback
    assert old is not None
    assert torch.count_nonzero(old.output.weight).item() == 0
    assert torch.count_nonzero(old.output.bias).item() == 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = SupportIdentityReadout(candidate_specific)
    if state is not None:
        head.load_state_dict(state,strict=True)
    assert sum(parameter.numel() for parameter in head.parameters()) == 107040
    assert len(head.state_dict()) == 8
    # Reuse the existing deferred final semantic-head hook; remove the ineffective
    # zero-output geometry R rather than append another deployed branch.
    model.boundary_evidence_readback = head
    model.eval()
    return head
