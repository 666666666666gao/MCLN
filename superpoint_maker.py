# superpoints fetch
import argparse
import segmentator
import open3d as o3d
import torch
import os
from six.moves import cPickle
import numpy as np

# BRIEF read from pkl
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def _scan_mesh_path(data_path_scannet, scan):
    nested = os.path.join(
        data_path_scannet, scan, scan + "_vh_clean_2.ply"
    )
    if os.path.exists(nested):
        return nested
    flat = os.path.join(data_path_scannet, scan + "_vh_clean_2.ply")
    if os.path.exists(flat):
        return flat
    raise FileNotFoundError(nested)


def fallback_superpoints_from_scan_pc(scan, voxel_size=0.18):
    """Build deterministic pseudo-superpoints from the sampled Scan point cloud."""
    coords = np.asarray(scan.pc[:, :3], dtype=np.float32)
    mins = coords.min(axis=0, keepdims=True)
    voxel = np.floor((coords - mins) / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def generate_superpoint(data_path, data_path_scannet, split,
                        limit=None, skip_existing=True,
                        skip_missing_mesh=False,
                        fallback_from_scan_pc=False,
                        fallback_voxel_size=0.18):


    scans = unpickle_data(f'{data_path}/{split}_v3scans.pkl')
    scans = list(scans)[0]
    output_dir = os.path.join(data_path, "superpoints", split)
    os.makedirs(output_dir, exist_ok=True)

    for idx, scan in enumerate(scans):
        if limit is not None and idx >= limit:
            break
        output_path = os.path.join(output_dir, scan + "_superpoint.pth")
        if skip_existing and os.path.exists(output_path):
            continue
        try:
            spformer_file = _scan_mesh_path(data_path_scannet, scan)
        except FileNotFoundError:
            if fallback_from_scan_pc:
                superpoint = fallback_superpoints_from_scan_pc(
                    scans[scan],
                    voxel_size=fallback_voxel_size,
                )
                torch.save(superpoint, output_path)
                print("Saving fallback " + scan)
                continue
            if skip_missing_mesh:
                print("Missing mesh " + scan)
                continue
            raise
        mesh = o3d.io.read_triangle_mesh(spformer_file)
        vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
        faces = torch.tensor(np.array(mesh.triangles), dtype=torch.int64)
        superpoint = segmentator.segment_mesh(vertices, faces)
        select_idx = torch.tensor(scans[scan].choices)
        superpoint = torch.index_select(superpoint, 0, select_idx).numpy()
        torch.save(superpoint, output_path)
        print("Saving " + scan)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/root/autodl-tmp/DATA_ROOT/')
    parser.add_argument('--data_path_scannet',
                        default='/root/autodl-tmp/DATA_ROOT/scannet/scans')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_missing_mesh', action='store_true')
    parser.add_argument('--fallback_from_scan_pc', action='store_true')
    parser.add_argument('--fallback_voxel_size', type=float, default=0.18)
    args = parser.parse_args()
    generate_superpoint(
        data_path=args.data_path,
        data_path_scannet=args.data_path_scannet,
        split=args.split,
        limit=args.limit,
        skip_existing=not args.overwrite,
        skip_missing_mesh=args.skip_missing_mesh,
        fallback_from_scan_pc=args.fallback_from_scan_pc,
        fallback_voxel_size=args.fallback_voxel_size,
    )



