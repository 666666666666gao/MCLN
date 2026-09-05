"""Object appearance from existing box crops, for an isolated native pilot.

Inputs are the same sampled colored points and supplied object boxes/mask.
No instance memberships, target labels or text parsing are used for cropping.
"""

import torch
from torch import nn


def box_crop_mask(xyz, box):
    """Use the explicit float32 AABB bounds supplied to this branch."""
    half_size = box[3:] * .5
    return ((xyz >= box[:3] - half_size) & (xyz <= box[:3] + half_size)).all(dim=-1)


class ObjectPointAppearanceResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.output = nn.Linear(128, 288, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, point_clouds, boxes, valid):
        """(B,N,6), (B,M,6), (B,M) -> (B,M,288); padded slots stay zero."""
        residual = point_clouds.new_zeros(boxes.shape[:2] + (288,))
        for batch_index in range(point_clouds.shape[0]):
            points = point_clouds[batch_index]
            for object_index in valid[batch_index].nonzero().flatten():
                box = boxes[batch_index, object_index]
                half_size = box[3:] * .5
                assert (half_size > 0).all()
                inside = box_crop_mask(points[:, :3], box)
                crop = points[inside]
                # Validate the full planned input set before native training.
                assert crop.shape[0] > 0
                local = torch.cat(((crop[:, :3] - box[:3]) / half_size, crop[:, 3:6]), dim=-1)
                encoded = self.point_encoder(local)
                pooled = torch.cat((encoded.mean(dim=0), encoded.max(dim=0).values))
                residual[batch_index, object_index] = self.output(pooled)
        return residual


class LastDecoderObjectAppearanceIntervention:
    """Add appearance only to the last Decoder's existing object memory."""

    def __init__(self, model, addon):
        self.model = model
        self.addon = addon
        self.inputs = None
        self.calls = 0
        self.original_decoder = model.decoder[-1].forward
        self.handles = [model.register_forward_pre_hook(self.begin_batch),
                        model.register_forward_hook(self.end_batch)]
        model.decoder[-1].forward = self.decode

    def begin_batch(self, module, arguments):
        inputs = arguments[0]
        self.inputs = (inputs['point_clouds'], inputs['det_boxes'], inputs['det_bbox_label_mask'])
        self.calls = 0

    def decode(self, *arguments, **keywords):
        original_memory = keywords['detected_feats']
        residual = self.addon(*self.inputs)
        assert original_memory.shape == residual.shape
        keywords['detected_feats'] = original_memory + residual
        self.calls += 1
        return self.original_decoder(*arguments, **keywords)

    def end_batch(self, module, arguments, outputs):
        assert self.calls == 1
        self.inputs = None

    def remove(self):
        self.model.decoder[-1].forward = self.original_decoder
        for handle in self.handles:
            handle.remove()
        self.inputs = None
