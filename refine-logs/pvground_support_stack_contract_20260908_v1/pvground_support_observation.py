"""CPU geometry diagnostics for exported 3D VSA support; never model inputs.

One packet represents one source and radius. Indices must be converted by the
exporter to GLOBAL rows of support_xyz; operator empty masks must be preserved.
BEV needs a separate 2D audit and is intentionally outside this format.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def stack_indices_to_global(local_indices, empty_ball_mask,
                            support_batch_counts, query_batch_counts):
    """Convert BallQuery's returned indices BEFORE its empty mask is discarded.

    Counts specify the same contiguous batch segments consumed by CUDA. The
    caller must separately export true coordinate batch IDs to audit alignment.
    """
    indices = np.asarray(local_indices)
    empty = np.asarray(empty_ball_mask)
    supports = np.asarray(support_batch_counts)
    queries = np.asarray(query_batch_counts)
    assert supports.ndim == queries.ndim == 1 and supports.shape == queries.shape
    assert indices.ndim == 2 and len(indices) == queries.sum()
    assert empty.shape == (len(indices),) and empty.dtype == np.bool_
    for values in [indices, supports, queries]:
        assert np.issubdtype(values.dtype, np.integer)
    assert (supports >= 0).all() and (queries >= 0).all()
    batch = np.repeat(np.arange(len(queries)), queries)
    valid = ~empty
    assert ((indices[valid] >= 0) & (indices[valid] < supports[batch[valid], None])).all()
    offsets = np.cumsum(supports) - supports
    global_indices = np.where(valid[:, None], indices + offsets[batch, None], -1)
    return global_indices, valid


def observe_support(support_xyz, support_batch, query_xyz, query_batch,
                    selected_global_indices, selected_valid, radius):
    support_xyz = np.asarray(support_xyz, dtype=np.float64)
    query_xyz = np.asarray(query_xyz, dtype=np.float64)
    support_batch = np.asarray(support_batch)
    query_batch = np.asarray(query_batch)
    indices = np.asarray(selected_global_indices)
    valid = np.asarray(selected_valid)
    assert support_xyz.ndim == query_xyz.ndim == 2
    assert support_xyz.shape[1] == query_xyz.shape[1] == 3
    assert support_batch.shape == (len(support_xyz),)
    assert query_batch.shape == valid.shape == (len(query_xyz),)
    assert indices.ndim == 2 and len(indices) == len(query_xyz)
    assert valid.dtype == np.bool_
    for values in [support_batch, query_batch, indices]:
        assert np.issubdtype(values.dtype, np.integer)
    assert np.isfinite(support_xyz).all() and np.isfinite(query_xyz).all()
    assert np.isfinite(radius) and radius > 0
    rows = []
    for i, query in enumerate(query_xyz):
        same_batch = np.flatnonzero(support_batch == query_batch[i])
        distances = np.linalg.norm(support_xyz[same_batch] - query, axis=1)
        neighbors = same_batch[distances < radius]
        # Empty operators may return [0,0,...]. Do not dereference these slots.
        selected = indices[i] if valid[i] else np.empty(0, dtype=np.int64)
        assert ((selected >= 0) & (selected < len(support_xyz))).all()
        selected_distances = np.linalg.norm(support_xyz[selected] - query, axis=1)
        selected_same_batch = support_batch[selected] == query_batch[i]
        rows.append({
            'query_index': i,
            'batch': int(query_batch[i]),
            'radius_m': float(radius),
            'available_support_rows': int(len(neighbors)),
            'available_distinct_xyz': int(len(np.unique(support_xyz[neighbors], axis=0))),
            'nearest_same_batch_distance_m': float(distances.min()) if len(distances) else None,
            'operator_valid': bool(valid[i]),
            'allocated_slots': int(indices.shape[1]),
            'selected_slots': int(len(selected)),
            'selected_unique_rows': int(len(np.unique(selected))),
            'selected_distinct_xyz': int(len(np.unique(support_xyz[selected], axis=0))),
            'selected_wrong_batch_slots': int((~selected_same_batch).sum()),
            'selected_outside_radius_slots': int((selected_distances >= radius).sum()),
            'selected_max_distance_m': float(selected_distances.max()) if len(selected) else None,
            'geometry_nonempty': bool(len(neighbors)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--source', choices=['raw_points', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4'], required=True)
    parser.add_argument('--query-location', choices=['vsa_keypoint', 'predicted_box_center'], required=True)
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(str(args.packet), allow_pickle=False) as packet:
        rows = observe_support(*(packet[key] for key in [
            'support_xyz', 'support_batch', 'query_xyz', 'query_batch',
            'selected_global_indices', 'selected_valid']), radius=args.radius)
    report = {'source': args.source, 'query_location': args.query_location,
              'packet_sha256': hashlib.sha256(args.packet.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'CPU float64 center-distance audit; not CUDA boundary equivalence, receptive-field coverage, semantic quality, or unique sensor observations',
              'rows': rows}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
