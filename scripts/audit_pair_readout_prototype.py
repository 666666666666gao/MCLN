"""CPU-only initialization/gradient probe; no dataset or optimizer."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from models.candidate_edge_direct_scorer import CandidateEdgeDirectScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == args.checkpoint_sha256
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    layer = checkpoint["config"].num_decoder_layers - 1
    prefix = "module.decoder.{}.self_attn.".format(layer)
    state = {key[len(prefix):]: value for key, value in checkpoint["model"].items()
             if key.startswith(prefix)}
    models = {}
    for mode in ["global", "pair"]:
        torch.manual_seed(9)
        model = CandidateEdgeDirectScorer(mode).eval()
        model.spatial.load_state_dict(state, strict=True)
        models[mode] = model
    for key, value in models["global"].state_dict().items():
        assert torch.equal(value, models["pair"].state_dict()[key]), key
    del checkpoint

    torch.manual_seed(0)
    valid = torch.ones(1, 256, dtype=torch.bool)
    valid[:, 240:] = False
    text_padding = torch.zeros(1, 32, dtype=torch.bool)
    text_padding[:, 24:] = True
    inputs = {
        "candidate_feats": torch.randn(1, 256, 288),
        "candidate_boxes": torch.cat([torch.randn(1, 256, 3), torch.ones(1, 256, 3)], -1),
        "text_feats": torch.randn(1, 32, 288),
        "text_padding_mask": text_padding,
        "query_indices": torch.arange(32).unsqueeze(0),
        "valid_query_mask": valid,
    }
    outputs = {mode: model(**inputs) for mode, model in models.items()}
    for output in outputs.values():
        assert torch.isfinite(output["candidate_logits"]).all()
        assert torch.count_nonzero(output["anchor_attention"][..., 240:256]) == 0
    outputs["pair"]["candidate_logits"].sum().backward()
    gradients = {}
    for name, parameter in models["pair"].named_parameters():
        if name in ["pair_query.0.weight", "pair_text_attention.in_proj_weight",
                    "spatial.lang_cond_fc.weight", "null_anchor", "score_head.weight"]:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            gradients[name] = float(parameter.grad.abs().sum())
            assert gradients[name] > 0, name
    result = {
        "schema": "mcln-pair-readout-prototype-cpu-v1",
        "checkpoint_sha256": digest.hexdigest(),
        "spatial_layer": layer,
        "strict_spatial_load": True,
        "common_initial_state_equal": True,
        "parameter_counts": {mode: sum(p.numel() for p in model.parameters())
                             for mode, model in models.items()},
        "pair_score_gradient_l1": gradients,
        "synthetic_target_count": 32,
        "synthetic_memory_count": 256,
        "null_memory_states": 1,
        "valid_memory_count": 240,
        "device": "cpu",
        "torch_version": torch.__version__,
        "benchmark_rows": 0,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
