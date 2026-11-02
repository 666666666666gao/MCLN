"""Read-only forward statistics and single-module bypasses for offline eval."""

import json

import torch


class PVGroundForwardAudit:
    """Attach after model construction and before evaluation, outside training."""

    def __init__(self, model, stream, bypass="none"):
        if bypass not in ("none", "reader", "support", "span"):
            raise ValueError("unknown module bypass")
        self.stream = stream
        self.bypass = bypass
        self.samples = 0
        self.last_gates = []
        self.reader_stats = []
        self.support_stats = []
        self.handles = [
            model.decoder[-1].local_visual.register_forward_hook(self._reader),
            model.pvground_support_corrector.register_forward_hook(self._support),
        ]
        self.handles.append(
            model.pvground_span_mixer.register_forward_hook(self._span)
        )

    @torch.no_grad()
    def _reader(self, module, inputs, output):
        query_norm = inputs[0].detach().norm(dim=-1).cpu()
        residual_norm = output.detach().norm(dim=-1).cpu()
        self.reader_stats = [
            {
                "reader_query_l2": query,
                "reader_residual_l2_before_bypass": residual,
            }
            for query, residual in zip(query_norm, residual_norm)
        ]
        if self.bypass == "reader":
            return torch.zeros_like(output)

    @torch.no_grad()
    def _support(self, module, inputs, output):
        predictions = inputs[5]
        self.support_stats = []
        for index, geometry in enumerate(output[1]):
            before = predictions["sp_last_pred_masks"][index].detach()
            after = output[0][index].detach()
            slots = torch.as_tensor(
                geometry["native_ids"], device=before.device, dtype=torch.long
            )
            delta = after[:, slots] - before[:, slots]
            text = predictions["last_pred_masks"][index][0].detach()[:, slots]
            alpha = predictions["adaptive_weights"][index].detach()
            # Use observed member slots and the same fused-logit foreground rule.
            old_foreground = (
                alpha * text + (1 - alpha) * before[:, slots]
            ).sigmoid() > 0.5
            new_foreground = (
                alpha * text + (1 - alpha) * after[:, slots]
            ).sigmoid() > 0.5
            self.support_stats.append({
                "support_delta_abs_mean_before_bypass": delta.abs().mean(-1).cpu(),
                "support_delta_abs_max_before_bypass": delta.abs().amax(-1).cpu(),
                "support_foreground_flips_before_bypass": (
                    old_foreground != new_foreground
                ).sum(-1).cpu(),
                "support_observed_members": len(slots),
            })
        if self.bypass == "support":
            # Keep observed geometry while downstream reference/Span recompute.
            return predictions["sp_last_pred_masks"], output[1]

    @torch.no_grad()
    def _span(self, module, inputs, output):
        predictions = inputs[2]
        centers, sizes, evidence = output
        native_center = predictions["native_coarse_center"]
        native_size = predictions["native_coarse_size"]
        applied_center = native_center if self.bypass == "span" else centers
        applied_size = native_size if self.bypass == "span" else sizes
        self.last_gates = []
        for batch_index, current in enumerate(evidence):
            raw = current["raw_axis_gate"].detach()
            gate = current["axis_gate"].detach()
            valid = current["reference_valid"].detach()
            self.last_gates.append({
                "raw": raw.cpu(), "clamped": gate.cpu(),
                **self.reader_stats[batch_index],
                **self.support_stats[batch_index],
            })
            row = {
                "kind": "forward_sample",
                "sample_index": self.samples,
                "bypass": self.bypass,
                "reader_bypassed": self.bypass == "reader",
                "support_bypassed": self.bypass == "support",
                "reader_residual_l2_before_bypass_mean": float(
                    self.reader_stats[batch_index][
                        "reader_residual_l2_before_bypass"
                    ].mean()
                ),
                "reader_query_l2_mean": float(
                    self.reader_stats[batch_index]["reader_query_l2"].mean()
                ),
                "support_observed_members": self.support_stats[batch_index][
                    "support_observed_members"
                ],
                "support_delta_abs_mean_before_bypass": float(
                    self.support_stats[batch_index][
                        "support_delta_abs_mean_before_bypass"
                    ].mean()
                ),
                "support_foreground_flips_before_bypass": int(
                    self.support_stats[batch_index][
                        "support_foreground_flips_before_bypass"
                    ].sum()
                ),
                "candidate_count": raw.shape[0],
                "reference_valid_count": int(valid.sum()),
                "raw_gate_finite": bool(torch.isfinite(raw).all()),
                "raw_gate_min": float(raw.min()),
                "raw_gate_max": float(raw.max()),
                "raw_gate_below_zero_by_axis": (raw < 0).sum(0).tolist(),
                "raw_gate_above_one_by_axis": (raw > 1).sum(0).tolist(),
                "clamped_gate_zero_by_axis": (gate == 0).sum(0).tolist(),
                "clamped_gate_one_by_axis": (gate == 1).sum(0).tolist(),
                "clamped_gate_mean_by_axis": gate.mean(0).tolist(),
                "center_abs_change_mean_by_axis": (
                    applied_center[batch_index] - native_center[batch_index]
                ).abs().mean(0).tolist(),
                "size_abs_change_mean_by_axis": (
                    applied_size[batch_index] - native_size[batch_index]
                ).abs().mean(0).tolist(),
            }
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.samples += 1
        self.stream.flush()
        if self.bypass == "span":
            return native_center, native_size, evidence

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
