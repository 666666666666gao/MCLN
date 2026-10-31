"""CPU-only comparison of initialized and trained PV-Ground state tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def fingerprint(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())


def audit(initial_path, trained_path, trainable_path, metrics_path):
    torch.set_num_threads(1)
    trainable = json.loads(trainable_path.read_text())
    history = {row["epoch"]: row for row in
               map(json.loads, metrics_path.read_text().splitlines())}
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    assert initial["epoch"] == 0
    architecture = initial["architecture"]
    assert all(name.startswith("module.") for name in initial["model"])
    state = {name[7:]: tensor for name, tensor in initial["model"].items()}
    assert set(trainable).issubset(state)
    assert all(state[name].numel() == count for name, count in trainable.items())
    frozen = {name: fingerprint(tensor) for name, tensor in state.items()
              if name not in trainable}
    additions = {name: state[name].clone() for name in trainable}
    del state, initial

    trained = torch.load(trained_path, map_location="cpu", weights_only=False)
    assert trained["epoch"] > 0 and trained["architecture"] == architecture
    assert all(name.startswith("module.") for name in trained["model"])
    state = {name[7:]: tensor for name, tensor in trained["model"].items()}
    assert set(state) == set(frozen) | set(trainable)
    changed_frozen = [name for name, signature in frozen.items()
                      if fingerprint(state[name]) != signature]
    prefixes = ("decoder.5.source_query_read.", "candidate_support_corrector.",
                "candidate_span_mixer.")
    rows = []
    for name, count in trainable.items():
        assert name.startswith(prefixes)
        before, after = additions[name].double(), state[name].double()
        assert after.numel() == count and torch.isfinite(after).all()
        delta = after - before
        row = dict(name=name, elements=count,
                   changed_elements=int(torch.count_nonzero(delta)),
                   initial_l2=float(before.norm()), trained_l2=float(after.norm()),
                   delta_l2=float(delta.norm()), delta_max=float(delta.abs().max()))
        if count <= 3:
            row["initial_values"] = before.reshape(-1).tolist()
            row["trained_values"] = after.reshape(-1).tolist()
        rows.append(row)
    groups = []
    for prefix in prefixes:
        members = [row for row in rows if row["name"].startswith(prefix)]
        assert members
        groups.append(dict(
            module=prefix[:-1], tensors=len(members),
            changed_tensors=sum(row["changed_elements"] > 0 for row in members),
            elements=sum(row["elements"] for row in members),
            changed_elements=sum(row["changed_elements"] for row in members),
            delta_l2=sum(row["delta_l2"] ** 2 for row in members) ** .5,
        ))
    reference = history[trained["epoch"]]
    assert reference["primary_score"] == "last/bbf" and reference["same_complete_model"]
    assert reference["rows"] == {"scanrefer": 9508, "nr3d": 7899,
                                 "sr3d": 17726}[architecture["dataset"]]
    return dict(
        diagnostic="checkpoint_state_update_audit", gpu_used=False,
        initial_checkpoint=str(initial_path), trained_checkpoint=str(trained_path),
        epoch=trained["epoch"], own_epoch_metrics=reference,
        frozen_state_tensors=len(frozen), changed_frozen_state=changed_frozen,
        frozen_state_unchanged=not changed_frozen,
        trainable_tensors=len(rows), modules=groups, parameter_deltas=rows,
        limits="Parameter deltas include optimizer effects; they do not measure "
               "gradient flow, gate saturation, causal importance or accuracy gains.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--trainable-parameters", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = audit(args.initial, args.trained, args.trainable_parameters,
                   args.reference_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "parameter_deltas"}))


if __name__ == "__main__":
    main()
