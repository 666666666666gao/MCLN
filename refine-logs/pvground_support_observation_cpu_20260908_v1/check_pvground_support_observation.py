"""Known-geometry fixtures, not model measurements or benchmark results."""
import json
import numpy as np

from pvground_support_observation import observe_support


def main():
    # Interleaved batches, repeated coordinates, strict radius boundary at 1m.
    xyz = np.array([[0, 0, 0], [0, 0, 0], [.25, 0, 0],
                    [1, 0, 0], [.25, 0, 0], [10, 0, 0]])
    batch = np.array([0, 1, 0, 0, 0, 1])
    queries = np.array([[0, 0, 0], [5, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    rows = observe_support(xyz, batch, queries, np.array([0, 0, 1, 2, 0]),
                           np.array([[0, 2, 2, 4], [0, 0, 0, 0], [1, 1, 1, 1],
                                     [0, 0, 0, 0], [1, 3, 3, 3]]),
                           np.array([True, False, True, False, True]), 1.)
    assert rows[0]['available_support_rows'] == 3
    assert rows[0]['available_distinct_xyz'] == 2
    assert rows[0]['selected_slots'] == 4
    assert rows[0]['selected_unique_rows'] == 3
    assert rows[0]['selected_distinct_xyz'] == 2
    assert rows[1]['selected_slots'] == 0 and rows[1]['nearest_same_batch_distance_m'] == 4.
    assert rows[1]['selected_max_distance_m'] is None
    assert rows[2]['available_support_rows'] == rows[2]['selected_unique_rows'] == 1
    assert rows[3]['nearest_same_batch_distance_m'] is None and not rows[3]['geometry_nonempty']
    assert rows[4]['selected_wrong_batch_slots'] == 1
    assert rows[4]['selected_outside_radius_slots'] == 3
    print(json.dumps({'status': 'pass', 'scope': 'five synthetic queries; not real scene coverage',
                      'model_forwards': 0, 'optimizer_steps': 0, 'formal_rows': 0,
                      'rows': rows}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
