"""CPU-only padding intervention on trained text and spatial-attention weights.

The geometry/query fixture is synthetic and fixed. This diagnoses the pooling
mechanism, not final grounding accuracy or a real-scene Query ranking.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from torch import nn
    from transformers import RobertaTokenizerFast, RobertaModel
    from models.encoder_decoder_layers import MultiHeadAttentionSpatial, calc_pairwise_locs
    from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, split_rows

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    data = Path("/root/autodl-tmp/DATA_ROOT")
    digest = hashlib.sha256()
    with open(str(args.checkpoint), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA:
        raise ValueError("protected checkpoint identity drift")
    payload = torch.load(str(args.checkpoint), map_location="cpu")
    state = {k[7:]: v for k, v in payload["model"].items()}
    del payload
    tokenizer = RobertaTokenizerFast.from_pretrained(str(data / "roberta-base"), local_files_only=True)
    encoder = RobertaModel.from_pretrained(str(data / "roberta-base"), local_files_only=True).eval()
    encoder.load_state_dict({k[len("text_encoder."):]: v for k, v in state.items()
                             if k.startswith("text_encoder.")}, strict=True)
    projector = nn.Sequential(nn.Linear(768, 288), nn.LayerNorm(288, eps=1e-12), nn.Dropout(.1)).eval()
    projector.load_state_dict({k[len("text_projector."):]: v for k, v in state.items()
                               if k.startswith("text_projector.")}, strict=True)
    prefix = "cross_encoder.layers.0.self_attention_visual."
    heads = state[prefix + "lang_cond_fc.weight"].shape[0] // 6
    spatial = MultiHeadAttentionSpatial(288, heads, dropout=0).eval()
    spatial.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    train_scenes = set(ast.literal_eval(Path("data/meta_data/nr3d_train_scans.txt").read_text()))
    with open(str(data / "refer_it_3d/nr3d.csv")) as stream:
        rows = [row for row in csv.DictReader(stream) if row["scan_id"] in train_scenes]
    ids = split_rows(rows)["fit"][:16]
    query = torch.randn(1, 12, 288)
    geometry = calc_pairwise_locs(torch.randn(1, 12, 3))
    records = []
    with torch.no_grad():
        for index in ids:
            tokenized = tokenizer(rows[index]["utterance"], return_tensors="pt")
            input_ids = tokenized["input_ids"]
            attention = tokenized["attention_mask"]
            length = input_ids.shape[1]
            single = projector(encoder(input_ids=input_ids, attention_mask=attention).last_hidden_state)
            padded_ids = torch.cat([input_ids, torch.full((1, 16), tokenizer.pad_token_id, dtype=torch.long)], dim=1)
            padded_mask = torch.cat([attention, torch.zeros(1, 16, dtype=torch.long)], dim=1)
            padded = projector(encoder(input_ids=padded_ids, attention_mask=padded_mask).last_hidden_state)
            original_pool = single.max(1)[0]
            padded_pool, winners = padded.max(1)
            masked_pool = padded.masked_fill(~padded_mask.bool().unsqueeze(-1), -float("inf")).max(1)[0]
            output, weights = spatial(query, query, query, geometry, txt_embeds=original_pool)
            padded_output, padded_weights = spatial(query, query, query, geometry, txt_embeds=padded_pool)
            masked_output, masked_weights = spatial(query, query, query, geometry, txt_embeds=masked_pool)
            records.append({
                "train_row_id": index, "tokens": length, "appended_padding": 16,
                "valid_text_max_abs_delta": (single - padded[:, :length]).abs().max().item(),
                "pool_max_abs_delta": (original_pool - padded_pool).abs().max().item(),
                "padding_winner_channels": int((winners >= length).sum()),
                "masked_pool_max_abs_delta": (original_pool - masked_pool).abs().max().item(),
                "attention_max_abs_delta": (weights - padded_weights).abs().max().item(),
                "masked_attention_max_abs_delta": (weights - masked_weights).abs().max().item(),
                "output_max_abs_delta": (output - padded_output).abs().max().item(),
                "masked_output_max_abs_delta": (output - masked_output).abs().max().item(),
            })
    result = {"schema": "mcln-padding-pool-intervention-v1", "checkpoint_sha256": CHECKPOINT_SHA,
              "scope": "trained_text_and_first_encoder_spatial_attention_on_fixed_synthetic_queries",
              "actual_scene_forward": False, "formal_validation_evaluated": False,
              "optimizer_steps": 0, "weights_written": 0, "records": records}
    with open(str(args.output), "x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"rows": len(records), "rows_with_padding_winners": sum(r["padding_winner_channels"] > 0 for r in records),
                      "maxima": {k: max(r[k] for r in records) for k in records[0] if "delta" in k}}, indent=2))


if __name__ == "__main__":
    main()
