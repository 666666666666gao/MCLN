"""Eval-only geometry counterfactuals using the deployed semantic evaluator.

Run with torchrun from a deployed module-FT code directory, passing this tool
the normal train_dist_mod.py arguments plus --eval and --replay-output FILE.
Wait for an idle GPU. This tool does not attach to a live training process.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap
from types import MethodType

import torch


def trace_root_queries(evaluator):
    """Observe the deployed sort result without replacing its ranking logic."""
    method = evaluator.evaluate_bbox_by_sem_align.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "top"
                           for target in node.targets)]
    assert len(assignments) == 1, "Expected one deployed query-ranking assignment"
    target = assignments[0]

    class AddTrace(ast.NodeTransformer):
        def visit_Assign(self, node):
            if node is target:
                record = ast.parse(
                    "self._replay_root_queries.append(int(top[0, 0].item()))"
                ).body[0]
                return [node, ast.copy_location(record, node)]
            return node

    tree = ast.fix_missing_locations(AddTrace().visit(tree))
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(method), "exec"), method.__globals__, namespace)
    evaluator._replay_root_queries = []
    evaluator.evaluate_bbox_by_sem_align = MethodType(namespace[method.__name__], evaluator)


class Top1HitTrace(dict):
    """Record the deployed evaluator's per-root top-1 counter increments."""

    def __init__(self, counts):
        super().__init__(counts)
        self.samples = {0.25: [], 0.5: []}

    def __setitem__(self, key, value):
        if key in (("last_", 0.25, 1, "bbf"), ("last_", 0.5, 1, "bbf")):
            increment = value - self[key]
            assert increment in (0, 1), "Paired replay requires one root per row"
            self.samples[key[1]].append(int(increment))
        super().__setitem__(key, value)


def check_recorded_metrics(path, checkpoint_path, expected_rows, formal):
    epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = checkpoint["epoch"]
        del checkpoint
    matching = [row for row in map(json.loads, path.read_text().splitlines())
                if row["epoch"] == epoch]
    assert len(matching) == 1
    reference = matching[0]
    assert reference["rows"] == expected_rows
    assert reference["primary_score"] == "last/bbf" and reference["same_complete_model"]
    assert all(formal[key] == reference[key] for key in ("rows", "hits025", "hits050"))
    return reference


class GeometryReplay:
    def __init__(self, original):
        self.original = original
        self.evaluators = {}
        for protocol in ("official_filter", "unfiltered_fixed_queries"):
            for geometry in ("final", "native", "mask"):
                self.evaluators[protocol, geometry] = type(original)(
                    only_root=True, thresholds=[0.25, 0.5], topks=[1, 5, 10],
                    prefixes=["last_"],
                    filter_non_gt_boxes=(
                        original.filter_non_gt_boxes
                        if protocol == "official_filter" else False),
                    logger=original.logger, model=original.model,
                )
                evaluator = self.evaluators[protocol, geometry]
                evaluator.dets = Top1HitTrace(evaluator.dets)
                trace_root_queries(evaluator)
        self.bbs_queries = []
        self.selected_gates = []
        self.selected_raw_gates = []
        self.gate_count = 0
        self.gate_sum = 0.0
        self.gate_zero = 0
        self.gate_one = 0
        self.raw_below_zero = 0
        self.raw_above_one = 0

    def observe(self, end_points):
        from native_root_bbs import native_root_bbs

        selected = native_root_bbs(
            end_points["last_sem_cls_scores"].detach(), end_points
        ).argsort(dim=-1, descending=True)[:, 0]
        self.bbs_queries.extend(selected.detach().cpu().tolist())
        sources = {
            "final": ("last_center", "last_pred_size"),
            "native": ("native_coarse_center", "native_coarse_size"),
            "mask": ("span_mask_center", "span_mask_size"),
        }
        for (protocol, geometry), evaluator in self.evaluators.items():
            center_key, size_key = sources[geometry]
            counterfactual = dict(end_points)
            counterfactual["last_center"] = end_points[center_key]
            counterfactual["last_pred_size"] = end_points[size_key].clamp_min(1e-6)
            assert counterfactual["last_proj_queries"] is end_points["last_proj_queries"]
            assert counterfactual["proj_tokens"] is end_points["proj_tokens"]
            evaluator.evaluate_bbox_by_sem_align(counterfactual, "last_")
            assert len(evaluator._replay_root_queries) == len(self.bbs_queries)
        gate = end_points["span_axis_gate"].detach()
        raw = end_points["span_raw_axis_gate"].detach()
        assert torch.isfinite(gate).all() and torch.isfinite(raw).all()
        assert torch.equal(gate, raw.clamp(0, 1))
        assert gate.shape == raw.shape == end_points["last_center"].shape
        batch_size = gate.shape[0]
        queries = self.evaluators["official_filter", "final"]._replay_root_queries[-batch_size:]
        indices = torch.as_tensor(queries, device=gate.device)
        batch_indices = torch.arange(batch_size, device=gate.device)
        self.selected_gates.append(gate[batch_indices, indices].cpu())
        self.selected_raw_gates.append(raw[batch_indices, indices].cpu())
        self.gate_count += gate.numel()
        self.gate_sum += gate.double().sum().item()
        self.gate_zero += (gate == 0).sum().item()
        self.gate_one += (gate == 1).sum().item()
        self.raw_below_zero += (raw < 0).sum().item()
        self.raw_above_one += (raw > 1).sum().item()

    def report(self, expected_rows):
        assert len(self.bbs_queries) == expected_rows
        rows = []
        selection = []
        for (protocol, geometry), evaluator in self.evaluators.items():
            keys = [("last_", threshold, 1, "bbf") for threshold in (0.25, 0.5)]
            assert [evaluator.gts[key] for key in keys] == [expected_rows] * 2
            hits = [int(evaluator.dets[key]) for key in keys]
            for key, total in zip(keys, hits):
                samples = evaluator.dets.samples[key[1]]
                assert len(samples) == expected_rows and sum(samples) == total
            if protocol == "official_filter" and geometry == "final":
                assert hits == [int(self.original.dets[key]) for key in keys]
                assert [self.original.gts[key] for key in keys] == [expected_rows] * 2
            rows.append(dict(
                protocol=protocol, geometry=geometry, rows=expected_rows,
                hits025=hits[0], hits050=hits[1],
                acc025=100 * hits[0] / expected_rows,
                acc050=100 * hits[1] / expected_rows,
                primary_score="last/bbf",
                counterfactual_geometry=geometry != "final",
            ))
            queries = evaluator._replay_root_queries
            assert len(queries) == expected_rows
            references = {
                "extra_mask_bbs": self.bbs_queries,
                "unfiltered_same_geometry":
                    self.evaluators["unfiltered_fixed_queries", geometry]._replay_root_queries,
                "same_protocol_final":
                    self.evaluators[protocol, "final"]._replay_root_queries,
            }
            for reference, reference_queries in references.items():
                assert len(reference_queries) == expected_rows
                agreements = sum(a == b for a, b in zip(queries, reference_queries))
                selection.append(dict(
                    protocol=protocol, geometry=geometry, reference=reference,
                    rows=expected_rows, agreements=agreements,
                    disagreements=expected_rows - agreements,
                    agreement_percent=100 * agreements / expected_rows,
                ))
        paired = []
        for protocol in ("official_filter", "unfiltered_fixed_queries"):
            for geometry in ("native", "mask"):
                for threshold in (0.25, 0.5):
                    final = self.evaluators[protocol, "final"].dets.samples[threshold]
                    alternative = self.evaluators[protocol, geometry].dets.samples[threshold]
                    transitions = dict(both_correct=0, repaired=0,
                                       damaged=0, both_wrong=0)
                    labels = {(1, 1): "both_correct", (0, 1): "repaired",
                              (1, 0): "damaged", (0, 0): "both_wrong"}
                    for before, after in zip(final, alternative):
                        transitions[labels[before, after]] += 1
                    assert sum(transitions.values()) == expected_rows
                    net = transitions["repaired"] - transitions["damaged"]
                    assert net == sum(alternative) - sum(final)
                    paired.append(dict(
                        protocol=protocol, geometry=geometry, threshold=threshold,
                        reference_geometry="final", rows=expected_rows,
                        **transitions, net_hits=net,
                    ))
        assert self.gate_count > 0
        selected_gate = torch.cat(self.selected_gates)
        selected_raw = torch.cat(self.selected_raw_gates)
        assert selected_gate.shape == selected_raw.shape == (expected_rows, 3)
        return dict(
            diagnostic="same_forward_geometry_replay",
            same_prediction_scores=True,
            plain_author_model_identity_tested=False,
            unfiltered_results_are_diagnostic_only=True,
            official_filter_enabled=self.original.filter_non_gt_boxes,
            rows=rows,
            paired_transitions=paired,
            query_selection_agreement=selection,
            query_selection_scope=(
                "eval_forward_root_queries; does not measure training matcher roles "
                "or establish causal accuracy cost"
            ),
            pairing="same_forward_batch_order_and_deployed_per_root_counter_updates",
            gates=dict(
                population="all_query_axes",
                count=self.gate_count, mean=self.gate_sum / self.gate_count,
                zero_fraction=self.gate_zero / self.gate_count,
                one_fraction=self.gate_one / self.gate_count,
                raw_below_zero_fraction=self.raw_below_zero / self.gate_count,
                raw_above_one_fraction=self.raw_above_one / self.gate_count,
            ),
            selected_gates=dict(
                population="official_final_root_query_axes",
                rows=expected_rows, count=selected_gate.numel(),
                mean=selected_gate.double().mean().item(),
                zero_fraction=(selected_gate == 0).double().mean().item(),
                one_fraction=(selected_gate == 1).double().mean().item(),
                raw_below_zero_fraction=(selected_raw < 0).double().mean().item(),
                raw_above_one_fraction=(selected_raw > 1).double().mean().item(),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    options, training_arguments = parser.parse_known_args()
    assert "--eval" in training_arguments, "Geometry replay requires --eval"
    assert "--eval_train" not in training_arguments
    assert int(os.environ["WORLD_SIZE"]) == 1, "Use one evaluation worker"
    assert os.environ.get("REFERIT_ONE_UPDATE") != "1"
    output = options.replay_output.resolve()
    assert not output.exists(), "Choose a new result path"
    source = Path.cwd() / "train_dist_mod.py"
    assert source.is_file(), "Run from the deployed code directory"
    sys.path.insert(0, str(source.parent))
    from src.grounding_evaluator import GroundingEvaluator

    original_evaluate = GroundingEvaluator.evaluate
    replay = None

    def evaluate(evaluator, end_points, prefix):
        nonlocal replay
        original_evaluate(evaluator, end_points, prefix)
        assert prefix == "last_"
        if replay is None:
            replay = GeometryReplay(evaluator)
        assert replay.original is evaluator
        replay.observe(end_points)

    GroundingEvaluator.evaluate = evaluate
    sys.argv = [str(source), *training_arguments]
    namespace = runpy.run_path(str(source), run_name="__main__")
    GroundingEvaluator.evaluate = original_evaluate
    args = namespace["opt"]
    assert args.eval and not args.eval_train
    expected_rows = {"scanrefer": 9508, "nr3d": 7899, "sr3d": 17726}[args.test_dataset]
    result = replay.report(expected_rows)
    result["dataset"] = args.test_dataset
    result["run_log_dir"] = args.log_dir
    result["module_checkpoint"] = args.checkpoint_path
    result["init_spec"] = json.loads(Path(args.native_init_spec).read_text())
    formal = next(row for row in result["rows"]
                  if row["protocol"] == "official_filter" and row["geometry"] == "final")
    result["reference_metric"] = check_recorded_metrics(
        options.reference_metrics, args.checkpoint_path, expected_rows, formal)
    result["matches_recorded_metrics"] = True
    files = ("train_dist_mod.py", "src/grounding_evaluator.py",
             "models/pv_ground.py", "native_model_initialization.py",
             "extremal_span_mixer.py", "native_root_bbs.py")
    result["source_sha256"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
