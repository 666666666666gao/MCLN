"""One frozen four-fit-row forward followed by untrained readout probes.

Run only after the registered G0/P1 jobs release the shared GPU lock. This
script does not select an experiment loss, optimize weights or evaluate a new
head's accuracy. Root-target labels are not supplied to either readout.
"""

import argparse
import inspect
import json
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    import numpy as np
    import torch
    import models

    # The standalone probe loads original MCLN and only the new readout files
    # from this addon directory. The live source tree is never edited.
    models.__path__.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    from models.candidate_edge_adapter import build_candidate_edge_inputs
    from models.candidate_edge_direct_scorer import CandidateEdgeDirectScorer
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.run_nr3d_view_pair_role import file_sha, read_train_rows
    from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, split_rows

    assert file_sha(opt.checkpoint) == CHECKPOINT_SHA
    raw_rows = read_train_rows(Path("/root/autodl-tmp/DATA_ROOT"))
    row_ids = split_rows(raw_rows)["fit"][:4]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.argv = [sys.argv[0]]
    args = parse_option()
    checkpoint = torch.load(str(opt.checkpoint), map_location="cpu")
    vars(args).update(vars(checkpoint["config"]))
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict({key[7:]: value for key, value in checkpoint["model"].items()}, strict=True)
    model.requires_grad_(False)
    del checkpoint

    class FitFour(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[index] for index in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FitFour(dataset_dict={"nr3d": 1}, test_dataset="nr3d", split="train",
                      data_path="/root/autodl-tmp/DATA_ROOT/", use_color=True,
                      detect_intermediate=True, butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    for anno, index in zip(dataset.annos, row_ids):
        assert anno["scan_id"] == raw_rows[index]["scan_id"]
        assert anno["target_id"] == int(raw_rows[index]["target_id"])
    batch = TrainTester._to_gpu(next(iter(torch.utils.data.DataLoader(dataset, batch_size=4))))
    inputs = TrainTester._get_inputs(batch)
    inputs["train"] = False
    captured = {}

    def before_mask_projection(module, arguments):
        captured["decoder_query_last"] = arguments[0].transpose(1, 2).detach().clone()

    hook = model.x_query.register_forward_pre_hook(before_mask_projection)
    with torch.no_grad():
        outputs = model(inputs)
    hook.remove()
    adapted = build_candidate_edge_inputs(outputs, inputs, captured["decoder_query_last"])
    assert adapted["candidate_feats"].shape == (4, 256, 288)
    assert adapted["text_feats"] is outputs["text_memory"]
    snapshots = {name: value.detach().clone() for name, value in adapted.items()}

    readouts = {}
    for mode in ["global", "pair"]:
        torch.manual_seed(9)
        torch.cuda.manual_seed_all(9)
        readout = CandidateEdgeDirectScorer(mode).cuda().eval()
        readout.spatial.load_state_dict(model.decoder[-1].self_attn.state_dict(), strict=True)
        readouts[mode] = readout
    for name, value in readouts["global"].state_dict().items():
        assert torch.equal(value, readouts["pair"].state_dict()[name]), name
    results = {mode: readout(**adapted) for mode, readout in readouts.items()}
    for result in results.values():
        valid = result["candidate_valid_mask"]
        assert torch.isfinite(result["candidate_logits"][valid]).all()
        assert torch.isneginf(result["candidate_logits"][~valid]).all()
    for name, before in snapshots.items():
        assert torch.equal(before, adapted[name]), name

    pair_result = results["pair"]
    # A derivative probe, not a supervised training objective or optimizer step.
    pair_result["candidate_logits"][pair_result["candidate_valid_mask"]].sum().backward()
    gradients = {}
    for name, parameter in readouts["pair"].named_parameters():
        if name in ["pair_query.0.weight", "pair_text_attention.in_proj_weight",
                    "spatial.lang_cond_fc.weight", "null_anchor", "score_head.weight"]:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            gradients[name] = float(parameter.grad.abs().sum())
            assert gradients[name] > 0, name
    assert all(parameter.grad is None for parameter in model.parameters())
    rows = []
    for row, source_id in enumerate(row_ids):
        valid_targets = pair_result["candidate_valid_mask"][row]
        rows.append({
            "fit_row_id": source_id,
            "scan_id": raw_rows[source_id]["scan_id"],
            "target_id": int(raw_rows[source_id]["target_id"]),
            "query_indices": adapted["query_indices"][row].tolist(),
            "valid_memory_count": int(adapted["valid_query_mask"][row].sum()),
            "valid_target_count": int(valid_targets.sum()),
            "valid_token_count": int((~adapted["text_padding_mask"][row]).sum()),
            "null_attention_mean": {
                mode: float(result["null_anchor_attention"][:, row, valid_targets].mean())
                for mode, result in results.items()
            },
        })
    result = {
        "schema": "mcln-pair-readout-four-fit-probe-v1",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "script_sha256": file_sha(__file__),
        "scorer_sha256": file_sha(inspect.getfile(CandidateEdgeDirectScorer)),
        "adapter_sha256": file_sha(inspect.getfile(build_candidate_edge_inputs)),
        "fit_row_ids": row_ids, "rows": rows,
        "pair_score_gradient_l1": gradients,
        "shared_initial_state_equal": True,
        "adapted_inputs_unchanged": True,
        "backbone_gradients_absent": True,
        "backbone_forward_count": 1,
        "untrained_readout_forward_count": 2,
        "optimizer_steps": 0, "weights_written": 0,
        "new_head_accuracy_evaluated": False,
        "formal_validation_evaluated": False,
        "augmentation": False,
    }
    with opt.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
