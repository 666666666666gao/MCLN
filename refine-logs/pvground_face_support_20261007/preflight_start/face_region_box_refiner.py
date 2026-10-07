"""Read actual members near a full face rectangle, preserving the existing head."""
import torch

from mask_reference import MaskReferenceBoxRefiner


class FaceRegionBoxRefiner(MaskReferenceBoxRefiner):
    """Change only which 16 input members are read at each of the six faces.

    The center uses the existing nearest-point implementation. Face members are
    closest to the bounded face rectangle, including its full tangential area.
    Encoded distances remain Euclidean distances to the original face center,
    so the existing 14-channel member encoding, pooling and six-face decoder
    have unchanged meanings and parameter count. There is no radius rejection,
    foreground-only filtering, added score, GT input or alternative box output.
    """

    def __init__(self, sampling_mode):
        assert sampling_mode in ('face_center', 'face_region')
        super().__init__('fused_mask')
        self.sampling_mode = sampling_mode

    @torch.no_grad()
    def nearest_members(self, xyz, locations):
        if self.sampling_mode == 'face_center':
            return super().nearest_members(xyz, locations)
        batch = len(xyz)
        assert locations.shape == (batch, 256 * 7, 3)
        layout = locations.reshape(batch, 256, 7, 3)
        center = layout[:, :, 0]
        half_size = .5 * ((layout[:, :, 1]-layout[:, :, 2])
                         +(layout[:, :, 3]-layout[:, :, 4])
                         +(layout[:, :, 5]-layout[:, :, 6])).abs()
        center_indices, center_distances = super().nearest_members(xyz, center)
        all_indices = [center_indices]
        all_distances = [center_distances]
        for face, axis in ((1,0),(2,0),(3,1),(4,1),(5,2),(6,2)):
            indices = []
            distances = []
            for begin in range(0,256,self.query_chunk):
                end = begin+self.query_chunk
                positions = layout[:,begin:end,face]
                # Distance to the rectangle: normal displacement and only the
                # part of each tangential displacement outside its true extent.
                square = (xyz[:,None,:,axis]-positions[:,:,None,axis]).square()
                for tangent in (value for value in range(3) if value != axis):
                    gap = (xyz[:,None,:,tangent]-center[:,begin:end,None,tangent]).abs()
                    gap = (gap-half_size[:,begin:end,None,tangent]).clamp_min(0)
                    square.add_(gap.square())
                members = square.topk(self.neighbors,dim=-1,largest=False,sorted=True).indices
                bid = torch.arange(batch,device=xyz.device)[:,None,None]
                member_xyz = xyz[bid,members]
                encoded_distance = (member_xyz-positions[:,:,None]).square().sum(-1).sqrt()
                indices.append(members)
                distances.append(encoded_distance)
            all_indices.append(torch.cat(indices,dim=1))
            all_distances.append(torch.cat(distances,dim=1))
        return (torch.stack(all_indices,dim=2).reshape(batch,256*7,self.neighbors),
                torch.stack(all_distances,dim=2).reshape(batch,256*7,self.neighbors))


def install_face_region_reading(model, sampling_mode):
    previous = model.candidate_box_refiner
    assert isinstance(previous,MaskReferenceBoxRefiner) and previous.reference_mode=='fused_mask'
    replacement = FaceRegionBoxRefiner(sampling_mode)
    replacement.load_state_dict(previous.state_dict(),strict=True)
    assert set(replacement.state_dict())==set(previous.state_dict())
    assert all(torch.equal(value,replacement.state_dict()[key]) for key,value in previous.state_dict().items())
    assert len(replacement.state_dict())==10
    assert sum(parameter.numel() for parameter in replacement.parameters())==456102
    model.candidate_box_refiner = replacement
