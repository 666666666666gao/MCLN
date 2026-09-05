"""CPU counterexample using the actual target/proposal box augmentation paths.

Synthetic boxes only. This measures a mechanism, not its prevalence in Nr3D.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset, DC
    from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
    from models.rec_evaluator_filter import build_detector_overlap_valid

    class SyntheticScan:
        pc = np.zeros((8, 3))
        three_d_objects = [{"points": np.arange(8)}]

        def get_object_bbox(self, index):
            assert index == 0
            return np.array([[4.95, -.05, -.05], [5.05, .05, .05]])

        def get_object_instance_label(self, index):
            assert index == 0
            return "synthetic_object"

    dataset = Joint3DDataset.__new__(Joint3DDataset)
    dataset.split = "train"
    dataset.detect_intermediate = False
    dataset.label_map = {"synthetic_object": next(iter(DC.nyu40id2class))}
    scan = SyntheticScan()
    rows = []
    for augment in (False, True):
        dataset.augment = augment
        np.random.seed(0)
        target_boxes, proposal_boxes = [], []
        for _ in range(64):
            targets, _, _, _ = dataset._get_target_boxes({"target_id": 0}, scan)
            _, proposals, active = dataset._get_scene_objects(scan)
            assert active.tolist() == [True] + [False] * 131
            target_boxes.append(targets[0])
            proposal_boxes.append(proposals[0])
        targets = torch.tensor(np.stack(target_boxes), dtype=torch.float32)
        proposals = torch.tensor(np.stack(proposal_boxes), dtype=torch.float32)
        valid = torch.ones(64, 1, dtype=torch.bool)
        retained = build_detector_overlap_valid(
            targets[:, None], valid, proposals[:, None], valid)
        ious, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(targets),
                            box_cxcyczwhd_to_xyzxyz(proposals))
        diagonal = ious.diag()
        assert torch.equal(retained[:, 0], diagonal > .25)
        rows.append({"augmentation": augment, "synthetic_trials": 64,
                     "root_gt_identical_candidates_retained": int(retained.sum()),
                     "root_gt_identical_candidates_removed": int((~retained).sum()),
                     "target_proposal_center_max_abs_delta": float((targets[:, :3] - proposals[:, :3]).abs().max()),
                     "target_proposal_iou_min": float(diagonal.min()),
                     "target_proposal_iou_max": float(diagonal.max())})
    assert rows[0]["root_gt_identical_candidates_removed"] == 0
    assert rows[1]["root_gt_identical_candidates_removed"] > 0
    result = {"schema": "mcln-target-proposal-jitter-counterexample-v1",
              "synthetic_only": True, "benchmark_rows": 0, "optimizer_steps": 0,
              "base_center": [5., 0., 0.], "base_size": [.1, .1, .1],
              "numpy_seed": 0, "device": "cpu",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dataset_sha256": hashlib.sha256(Path(inspect.getfile(Joint3DDataset)).read_bytes()).hexdigest(),
              "filter_sha256": hashlib.sha256(Path(inspect.getfile(build_detector_overlap_valid)).read_bytes()).hexdigest(),
              "cases": rows}
    with opt.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
