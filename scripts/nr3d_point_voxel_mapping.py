"""Point-preserving voxel coordinates and continuous local XYZ/RGB inputs."""

import torch


def point_voxel_mapping(point_cloud, voxel_size=.02):
    """One scene (N,6): unique XYZ cells, N inverse ids, local (N,6), origin."""
    xyz = point_cloud[:, :3]
    origin = xyz.min(dim=0).values
    scaled = (xyz - origin) / voxel_size
    cells, inverse = torch.unique(scaled.floor().long(), dim=0, return_inverse=True)
    offsets = scaled - cells.index_select(0, inverse).to(scaled.dtype) - .5
    local = torch.cat((offsets, point_cloud[:, 3:6]), dim=-1)
    return cells, inverse, local, origin
