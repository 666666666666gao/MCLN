#!/usr/bin/env python
"""Audit MCLN source-choice adapter shapes and one-batch source headroom."""

import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTNET2 = os.path.join(ROOT, "pointnet2")
for path in (ROOT, POINTNET2):
    if path not in sys.path:
        sys.path.insert(0, path)

from main_utils import parse_option
from train_dist_mod import TrainTester
from src.joint_det_dataset import Joint3DDataset
from models.source_choice_adapter import build_mcln_source_choice_batch
from models.source_choice_selector import compute_source_top1_ious


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/autodl-tmp/DATA_ROOT/")
    parser.add_argument("--pp_checkpoint",
                        default="/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args, remaining = parser.parse_known_args()
    return args, remaining


def _shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, list):
        return [tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
                for v in value[:2]]
    return type(value).__name__


def main():
    audit_args, remaining = _parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "4455")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        backend = "nccl" if audit_args.cuda and torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=5400),
        )

    opt = parse_option()
    opt.data_root = audit_args.data_root
    opt.pp_checkpoint = audit_args.pp_checkpoint
    opt.batch_size = audit_args.batch_size
    opt.num_workers = audit_args.num_workers
    opt.dataset = ["scanrefer"]
    opt.test_dataset = "scanrefer"
    opt.model = "MCLN"
    opt.num_decoder_layers = 6
    opt.num_target = 256
    opt.use_color = True
    opt.use_soft_token_loss = True
    opt.use_contrastive_align = True
    opt.detect_intermediate = True
    opt.joint_det = True
    opt.butd = True
    opt.self_attend = True
    opt.augment_det = True
    opt.eval = True
    opt.eval_train = False
    opt.use_source_choice_selector = False

    _ = remaining
    test_dataset = Joint3DDataset(
        dataset_dict={"scanrefer": 1},
        test_dataset="scanrefer",
        split="val",
        use_color=True,
        use_height=False,
        overfit=False,
        data_path=opt.data_root,
        detect_intermediate=True,
        use_multiview=False,
        butd=True,
        butd_gt=False,
        butd_cls=False,
        wo_obj_name=opt.wo_obj_name,
        skip_missing_superpoints=True,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=audit_args.batch_size,
        shuffle=False,
        num_workers=audit_args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    model = TrainTester.get_model(opt)
    if audit_args.cuda and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    batch_data = next(iter(loader))
    batch_data = TrainTester._to_gpu(batch_data) if audit_args.cuda else batch_data
    inputs = TrainTester._get_inputs(batch_data)
    with torch.no_grad():
        end_points = model(inputs)
    for key in batch_data:
        end_points[key] = batch_data[key]

    audit_keys = [
        "last_center",
        "last_pred_size",
        "source_choice_candidate_feats",
        "text_feats",
        "text_attention_mask",
        "last_pred_masks",
        "sp_last_pred_masks",
        "adaptive_weights",
    ]
    print("Adapter field shapes:")
    for key in audit_keys:
        print("  {}: {}".format(key, _shape(end_points.get(key))))

    adapter_batch = build_mcln_source_choice_batch(
        end_points,
        end_points,
        source_names=["default", "mask_text"],
    )
    for source_name, scores in adapter_batch["source_scores"].items():
        print(
            "  source {} scores: shape={} min={:.4f} max={:.4f}".format(
                source_name,
                tuple(scores.shape),
                scores.min().item(),
                scores.max().item(),
            )
        )

    gt_boxes = torch.cat(
        [end_points["center_label"][..., :3], end_points["size_gts"]],
        dim=-1,
    )
    source_ious = compute_source_top1_ious(
        candidate_boxes=adapter_batch["candidate_boxes"],
        source_scores=adapter_batch["source_scores"],
        source_names=["default", "mask_text"],
        gt_boxes=gt_boxes,
        gt_mask=end_points["box_label_mask"],
    )
    for idx, source_name in enumerate(["default", "mask_text"]):
        print(
            "  fixed {} Acc@0.25={:.4f} Acc@0.50={:.4f}".format(
                source_name,
                (source_ious[:, idx] >= 0.25).float().mean().item(),
                (source_ious[:, idx] >= 0.50).float().mean().item(),
            )
        )
    best_iou = source_ious.max(dim=1).values
    print(
        "  oracle Acc@0.25={:.4f} Acc@0.50={:.4f}".format(
            (best_iou >= 0.25).float().mean().item(),
            (best_iou >= 0.50).float().mean().item(),
        )
    )


if __name__ == "__main__":
    main()
