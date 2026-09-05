"""Probe the existing spatial-attention interface with protected weights.

Synthetic CPU tensors only. No new scorer, dataset, optimizer, or checkpoint.
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    import torch
    from models.encoder_decoder_layers import MultiHeadAttentionSpatial, calc_pairwise_locs
    from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA
    from scripts.run_nr3d_view_pair_role import file_sha

    assert file_sha(opt.checkpoint) == CHECKPOINT_SHA
    checkpoint = torch.load(str(opt.checkpoint), map_location="cpu")
    layer = checkpoint["config"].num_decoder_layers - 1
    prefix = "module.decoder.{}.self_attn.".format(layer)
    state = {key[len(prefix):]: value for key, value in checkpoint["model"].items()
             if key.startswith(prefix)}
    module = MultiHeadAttentionSpatial(288, 8, dropout=0., spatial_attn_fusion="cond")
    module.load_state_dict(state, strict=True)
    module.eval()
    del checkpoint
    torch.manual_seed(0)
    features, centers = torch.randn(2, 256, 288), torch.randn(2, 256, 3)
    text = torch.randn(2, 288)
    valid = torch.ones(2, 256, dtype=torch.bool)
    valid[:, 240:] = False
    indices = torch.tensor([list(range(0, 64, 2)), list(range(1, 65, 2))])
    rows = torch.arange(2).unsqueeze(1)
    pairwise = calc_pairwise_locs(centers)

    def subset(memory, geometry):
        return module(features[rows, indices], memory, memory, geometry[rows, indices],
                      key_padding_mask=~valid, txt_embeds=text)

    with torch.no_grad():
        full_output, full_attention = module(features, features, features, pairwise,
                                             key_padding_mask=~valid, txt_embeds=text)
        target_output, target_attention = subset(features, pairwise)
        expected_output = full_output[rows, indices]
        expected_attention = torch.stack([full_attention[:, row, indices[row]] for row in range(2)], dim=1)
        assert torch.allclose(target_output, expected_output, atol=2e-6, rtol=1e-5)
        assert torch.allclose(target_attention, expected_attention, atol=2e-6, rtol=1e-5)
        padded_features, padded_centers = features.clone(), centers.clone()
        padded_features[~valid] += 1000.
        padded_centers[~valid] += 1000.
        masked_output, _ = subset(padded_features, calc_pairwise_locs(padded_centers))
        assert torch.equal(target_output, masked_output)
        wider_memory = features.clone()
        wider_memory[:, 100] += torch.linspace(-10., 10., 288)
        changed_output, _ = subset(wider_memory, pairwise)
        assert not torch.allclose(target_output, changed_output, atol=2e-6, rtol=1e-5)

    result = {"schema": "mcln-spatial-target-memory-probe-v1",
              "checkpoint_sha256": CHECKPOINT_SHA, "layer": layer,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "layer_source_sha256": file_sha(inspect.getfile(MultiHeadAttentionSpatial)),
              "synthetic_only": True, "device": "cpu", "benchmark_rows": 0,
              "optimizer_steps": 0, "target_count": 32, "memory_count": 256,
              "valid_memory_count": 240, "hidden_dim": 288, "heads": 8,
              "subset_output_max_abs_delta": float((target_output - expected_output).abs().max()),
              "subset_attention_max_abs_delta": float((target_attention - expected_attention).abs().max()),
              "masked_memory_max_abs_delta": float((target_output - masked_output).abs().max()),
              "outside_target_memory_max_abs_delta": float((target_output - changed_output).abs().max())}
    with opt.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
