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
        self.handles = []
        if bypass == "reader":
            self.handles.append(
                model.decoder[-1].local_visual.register_forward_hook(
                    self._reader
                )
            )
        if bypass == "support":
            self.handles.append(
                model.pvground_support_corrector.register_forward_hook(
                    self._support
                )
            )
        self.handles.append(
            model.pvground_span_mixer.register_forward_hook(self._span)
        )

    def _reader(self, module, inputs, output):
        return torch.zeros_like(output)

    def _support(self, module, inputs, output):
        # Keep observed point geometry; bypass only learned mask correction.
        return inputs[5]["sp_last_pred_masks"], output[1]

    @torch.no_grad()
    def _span(self, module, inputs, output):
        predictions = inputs[2]
        centers, sizes, evidence = output
        native_center = predictions["native_coarse_center"]
        native_size = predictions["native_coarse_size"]
        applied_center = native_center if self.bypass == "span" else centers
        applied_size = native_size if self.bypass == "span" else sizes
        for batch_index, current in enumerate(evidence):
            raw = current["raw_axis_gate"].detach()
            gate = current["axis_gate"].detach()
            valid = current["reference_valid"].detach()
            row = {
                "sample_index": self.samples,
                "bypass": self.bypass,
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
