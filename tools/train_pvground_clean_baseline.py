"""Train the no-PVI/no-TGS baseline using the author's data, heads and evaluator."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from main_utils import BaseTrainTester, parse_option
from train_dist_mod import TrainTester
from models import PVGround
from src.grounding_evaluator import GroundingEvaluator
from utils import get_scheduler


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class CleanTrainer(TrainTester):
    def __init__(self, args):
        BaseTrainTester.__init__(self, args)
        self.args = args

    def _get_inputs(self, batch):
        return {
            "point_clouds": batch["point_clouds"].float(),
            "text": batch["utterances"],
            "superpoint": batch["superpoint"],
            "det_boxes": batch["all_detected_boxes"],
            "det_bbox_label_mask": batch["all_detected_bbox_label_mask"],
            "det_class_ids": batch["all_detected_class_ids"],
        }

    def get_model(self, args):
        return PVGround(
            None, num_class=256, input_feature_dim=3,
            num_queries=args.num_target, num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=True, butd=True, data_path=args.data_root,
            self_attend=args.self_attend,
        )

    @torch.no_grad()
    def evaluate_clean(self, epoch, loader, model, criterion, set_criterion):
        model.eval()
        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[0.25, 0.5], topks=[1, 5, 10],
            prefixes=["last_"], filter_non_gt_boxes=self.args.butd_cls,
            logger=self.logger, model="PVGround",
        )
        for index, batch in enumerate(loader):
            _, end_points = self._main_eval_branch(
                index, batch, loader, model, {}, criterion, set_criterion, self.args)
            assert torch.isfinite(end_points["loss"]).item()
            evaluator.evaluate(end_points, "last_")
            if index % 100 == 0:
                print(f"CLEAN_VAL epoch={epoch} batch={index + 1}/{len(loader)}", flush=True)
        evaluator.synchronize_between_processes()
        expected = {"scanrefer": 9508, "nr3d": 7899, "sr3d": 17726}[self.args.test_dataset]
        counts = [int(evaluator.gts[("last_", threshold, 1, "bbf")])
                  for threshold in (0.25, 0.5)]
        assert counts == [expected, expected], counts
        hits = [int(evaluator.dets[("last_", threshold, 1, "bbf")])
                for threshold in (0.25, 0.5)]
        row = {
            "epoch": epoch, "trained": epoch > 0, "count": expected,
            "hits025": hits[0], "hits050": hits[1],
            "acc025": hits[0] / expected * 100, "acc050": hits[1] / expected * 100,
            "primary_score": "last/bbf", "same_complete_model": True,
            "architecture": "PointNet++_TopK_noPVI_noTGS",
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(Path(self.args.log_dir) / "native_metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("CLEAN_METRICS " + json.dumps(row), flush=True)
        return row

    def run(self):
        args = self.args
        output = Path(args.log_dir)
        model = self.get_model(args)
        payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        source = payload["model"]
        assert all(key.startswith("module.") for key in source)
        source = {key[7:]: value for key, value in source.items()}
        state = model.state_dict()
        # Prior author-checkpoint audits established this obsolete saved buffer.
        excluded = [key for key in source
                    if key.startswith(("backbone_net.", "gumbel."))
                    or key == "text_encoder.embeddings.position_ids"]
        loaded = {key: value for key, value in source.items() if key not in excluded}
        assert set(loaded) <= set(state), sorted(set(loaded) - set(state))
        assert all(state[key].shape == value.shape for key, value in loaded.items())
        missing = set(state) - set(loaded)
        assert all(key.startswith(("backbone_net.", "points_obj_cls.")) for key in missing), missing
        state.update(loaded)
        model.load_state_dict(state, strict=True)
        digest = hashlib.sha256()
        with open(args.checkpoint_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        expected_hash = {
            "scanrefer": "6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3",
            "nr3d": "d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d",
            "sr3d": "a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc",
        }[args.test_dataset]
        assert digest.hexdigest() == expected_hash
        audit = {
            "checkpoint": args.checkpoint_path, "source_epoch": payload["epoch"],
            "source_sha256": digest.hexdigest(),
            "loaded_tensors": sorted(loaded), "excluded_tensors": excluded,
            "new_tensors": sorted(missing),
            "new_initialization": "PointNet++ and Top-K use seeded default initialization",
            "fresh_optimizer_scheduler": True,
        }
        write_json(output / "load_audit.json", audit)
        del payload, source, loaded, state
        train_loader, val_loader = self.get_loaders(args)
        model = model.cuda()
        assert not hasattr(model, "gumbel")
        assert model.backbone_net.__class__.__name__ == "Pointnet2Backbone"
        optimizer = self.get_optimizer(args, model)
        scheduler = get_scheduler(optimizer, len(train_loader), args)
        criterion, set_criterion = self.get_criterion(args)
        model = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,
                                        find_unused_parameters=True)

        # One real batch tests backward and all reinstated trainable paths.
        # Restore parameters, buffers, optimizer and RNG before epoch-0 evaluation.
        initial = {key: value.detach().cpu().clone()
                   for key, value in model.module.state_dict().items()}
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state(),
               np.random.get_state(), random.getstate())
        model.train()
        batch = self._to_gpu(next(iter(train_loader)))
        end_points = model(self._get_inputs(batch))
        for key in batch:
            assert key not in end_points, key
            end_points[key] = batch[key]
        loss, end_points = self._compute_loss(end_points, criterion, set_criterion, args)
        assert torch.isfinite(loss).item()
        optimizer.zero_grad()
        loss.backward()
        groups = ("backbone_net.", "points_obj_cls.", "decoder.", "prediction_heads.", "x_mask.")
        gradients = {}
        probes = {}
        for prefix in groups:
            candidates = [(key, param) for key, param in model.module.named_parameters()
                          if key.startswith(prefix) and param.grad is not None
                          and param.grad.abs().sum().item() > 0]
            assert candidates, prefix
            key, param = candidates[0]
            gradients[prefix] = float(param.grad.norm())
            probes[key] = param.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        parameters = dict(model.module.named_parameters())
        changed = {key: not torch.equal(value, parameters[key]) for key, value in probes.items()}
        assert all(changed.values()), changed
        witness = {"loss": float(loss.detach()), "gradient_norms": gradients,
                   "changed_parameters": changed, "optimizer_step_complete": True,
                   "batch_size": args.batch_size, "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        write_json(output / "first_update.json", witness)
        print("CLEAN_FIRST_UPDATE " + json.dumps(witness), flush=True)
        model.module.load_state_dict(initial, strict=True)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
        np.random.set_state(rng[2])
        random.setstate(rng[3])
        del initial, batch, end_points, loss, probes, parameters
        history = []
        best_key = None
        best_trained_key = None
        decline_streak = 0

        def save(name, row):
            torch.save({
                "epoch": row["epoch"], "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "metrics": row, "architecture": "PointNet++_TopK_noPVI_noTGS",
                "args": vars(args), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }, output / name)

        for epoch in range(0, args.max_epoch + 1):
            if epoch > 0:
                train_loader.sampler.set_epoch(epoch)
                self.train_one_epoch(epoch, train_loader, model, criterion, set_criterion,
                                     optimizer, scheduler, args)
            row = self.evaluate_clean(epoch, val_loader, model, criterion, set_criterion)
            save("latest.pth", row)
            key = ((row["hits050"], row["hits025"]) if args.test_dataset == "scanrefer"
                   else (row["hits025"], row["hits050"]))
            if best_key is None or key > best_key:
                best_key = key
                save("best.pth", row)
            if epoch > 0:
                if best_trained_key is None or key > best_trained_key:
                    best_trained_key = key
                    save("best_trained.pth", row)
                history.append(row)
                if len(history) >= 4:
                    declining = all(
                        sum(item[metric] for item in history[-3:])
                        < sum(item[metric] for item in history[-4:-1])
                        for metric in ("hits025", "hits050"))
                    decline_streak = decline_streak + 1 if declining else 0
                if decline_streak >= 2:
                    write_json(output / "completed.json",
                               {"reason": "automatic_early_stop", "epoch": epoch})
                    break
        else:
            write_json(output / "completed.json",
                       {"reason": "max_epoch", "epoch": args.max_epoch})
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_option()
    assert args.model == "PVGround" and not args.frozen and not args.small_lr
    assert args.lr_decay_epochs == [10, 30] and args.warmup_epoch == 0
    assert args.batch_size == 10
    assert args.lr == 1e-5 and args.lr_backbone == 2e-5
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=5400))
    CleanTrainer(args).run()
