#!/usr/bin/env python
"""Eval-mode mask-head fine-tune for MCLN.

Trains ONLY x_mask + x_query (no BatchNorm/Dropout) while the entire model
stays in eval() mode permanently.  BN running stats are never updated, so the
backbone/box/position-prediction path produces bit-identical outputs to the
protected baseline — guaranteeing REC Position Acc does not change.

Launch:
  export PYTHONPATH=/home/gb/new\ butd/butd_detr-main/MCLN-main:/home/gb/new\ butd/butd_detr-main/MCLN-main/pointnet2
  export TOKENIZERS_PARALLELISM=false
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /root/miniconda3/envs/bdetr/bin/python -m torch.distributed.launch \
      --nproc_per_node 1 --master_port 29682 \
      scripts/run_mask_head_evalmode_ft.py \
      --data_root /root/autodl-tmp/DATA_ROOT/ \
      --checkpoint_path /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth \
      --log_dir /root/autodl-tmp/DATA_ROOT/output/mask_head_finetune/evalmode_001 \
      --lr 2e-4 --max_epoch 80
"""
import datetime
import hashlib
import os
import sys

# Make project importable
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "pointnet2"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from tqdm import tqdm

from main_utils import parse_option, save_checkpoint
from train_dist_mod import TrainTester


BACKBONE_SHA256 = "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
# Full mask-decoder head. All of these are MASK-ONLY modules — none feed the
# box/position path (decoder, prediction_heads, proposal_head, backbone_net,
# cross_encoder, contrastive_align_projection_*, text_encoder/projector), so
# training them in permanent eval() mode keeps REC Position bit-identical.
#   x_mask      : point mask feature generator (-> mask_feats -> super_features)
#   x_query     : seed-query segmentation head (-> sp_last_pred_masks)
#   rel_encoder : superpoint relative-coord encoder (mask feats only)
#   swa_layers/swa_ffn_layers : text-mask SWA refinement (-> last_pred_masks)
#   out_norm/out_score        : prediction_head scores + adaptive fusion weight
MASK_HEAD_PREFIXES = (
    "x_mask.", "x_query.", "rel_encoder.",
    "swa_layers.", "swa_ffn_layers.", "out_norm.", "out_score.",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_backbone(path):
    got = _sha256(path)
    if got != BACKBONE_SHA256:
        raise ValueError(
            "Backbone SHA-256 mismatch: got {} expected {}".format(
                got, BACKBONE_SHA256))
    mode = oct(os.stat(path).st_mode & 0o777)
    print("[evalmode-ft] backbone verified OK  mode={}".format(mode))


def configure_mask_head_only(model):
    """Freeze all params; unfreeze x_mask.* and x_query.* only."""
    model.requires_grad_(False)
    trained = []
    for name, param in model.named_parameters():
        # under DDP the name has 'module.' prefix
        bare = name.removeprefix("module.") if hasattr(name, "removeprefix") else (
            name[len("module."):] if name.startswith("module.") else name
        )
        if any(bare.startswith(p) for p in MASK_HEAD_PREFIXES):
            param.requires_grad_(True)
            trained.append(name)
    if not trained:
        raise RuntimeError(
            "No x_mask/x_query parameters found — "
            "check model wrapping or MASK_HEAD_PREFIXES"
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[evalmode-ft] trainable params: {:,} ({})".format(
        n_params, ", ".join(trained)))
    return trained


# ---------------------------------------------------------------------------
# Subclass — only override train_one_epoch and get_optimizer
# ---------------------------------------------------------------------------

class EvalModeTrainTester(TrainTester):
    """Keeps model permanently in eval() during training."""

    @staticmethod
    def get_optimizer(args, model):
        """Train only x_mask + x_query; everything else frozen."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("[evalmode-ft] no trainable parameters found")
        optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
        return optimizer

    def train_one_epoch(self, epoch, train_loader, model,
                        criterion, set_criterion, optimizer, scheduler, args):
        """Forward/backward in permanent eval() mode."""
        stat_dict = {}
        # NEVER call model.train() — BN running stats must not change
        model.eval()

        train_loader_wrapped = tqdm(train_loader, ascii=True)
        for batch_idx, batch_data in enumerate(train_loader_wrapped):
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            model.eval()  # belt-and-suspenders: stay eval every step
            with torch.enable_grad():
                end_points = model(inputs)

            for key in batch_data:
                if key not in end_points:
                    end_points[key] = batch_data[key]

            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args
            )

            optimizer.zero_grad()
            loss.backward()

            if args.clip_norm > 0:
                trainable = [p for p in model.parameters() if p.requires_grad]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_norm)

            optimizer.step()
            scheduler.step()
            model.eval()  # restore eval in case any hook changed it

            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:
                self.logger.info(
                    "Train: [{}][{}/{}]".format(
                        epoch, batch_idx + 1, len(train_loader))
                )
                self.logger.info("  " + "  ".join([
                    "{} {:.4f}".format(k, stat_dict[k] / (batch_idx + 1))
                    for k in sorted(stat_dict)
                    if "loss" in k
                    and "proposal_" not in k
                    and "last_" not in k
                    and "head_" not in k
                ]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
        timeout=datetime.timedelta(seconds=5400)
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Verify backbone is intact
    _verify_backbone(opt.checkpoint_path)

    # Build trainer — uses same loaders/model/evaluator as the original
    trainer = EvalModeTrainTester(opt)

    # Obtain loaders + model via standard path
    train_loader, test_loader = trainer.get_loaders(opt)
    model = trainer.get_model(opt)

    if torch.cuda.is_available():
        model = model.cuda()

    from torch.nn.parallel import DistributedDataParallel
    model = DistributedDataParallel(
        model, device_ids=[opt.local_rank],
        broadcast_buffers=False, find_unused_parameters=True
    )

    # Load backbone weights (strict=False; optimizer state NOT loaded)
    from main_utils import load_checkpoint
    opt.reduce_lr = True  # skip optimizer state load
    load_checkpoint(opt, model, None, None)

    # After load, configure trainability (load_checkpoint uses strict=False
    # and does not touch requires_grad, but set explicitly for clarity)
    configure_mask_head_only(model)
    model.eval()

    # Criterion, optimizer, scheduler
    criterion, set_criterion = trainer.get_criterion(opt)
    optimizer = EvalModeTrainTester.get_optimizer(opt, model)

    from utils.lr_scheduler import get_scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), opt)

    trainer.logger.info("=== Eval-mode mask-head fine-tune ===")
    trainer.logger.info("log_dir: {}".format(opt.log_dir))
    trainer.logger.info("lr={} max_epoch={} batch={}".format(
        opt.lr, opt.max_epoch, opt.batch_size))
    trainer.logger.info("warmup_epoch={}".format(opt.warmup_epoch))
    trainer.logger.info("trainable: {:,} / {:,} params".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters())
    ))

    import time
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        train_loader.sampler.set_epoch(epoch)
        tic = time.time()

        trainer.train_one_epoch(
            epoch, train_loader, model,
            criterion, set_criterion, optimizer, scheduler, opt
        )

        trainer.logger.info("Epoch {} done in {:.1f}s".format(
            epoch, time.time() - tic))

        if dist.get_rank() == 0 and epoch % opt.save_freq == 0:
            save_checkpoint(opt, epoch, model, optimizer, scheduler)

        if epoch % opt.val_freq == 0:
            trainer.logger.info("Evaluating epoch {} ...".format(epoch))
            trainer.evaluate_one_epoch(
                epoch, test_loader, model, criterion, set_criterion, opt
            )

    if dist.get_rank() == 0:
        save_checkpoint(opt, "last", model, optimizer, scheduler, save_cur=True)
    trainer.logger.info("Training complete.")
