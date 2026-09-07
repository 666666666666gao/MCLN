"""Execute isolated existing score expressions on synthetic CPU probabilities.

This is a score-contract check, not inference, a model test, or evidence of a
dataset-level metric gain. Full evaluator filtering and learned scorers are not
executed. AST extraction avoids unrelated dataset and CUDA imports.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def assigned_name(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


def run_nodes(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<existing-score-expression>', 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--expected-hashes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.source_root)
    hashes = json.loads(Path(args.expected_hashes).read_text())
    trees = {}
    for name, expected in hashes.items():
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        trees[name] = ast.parse(raw.decode('utf-8-sig'))
    namespace = {'torch': torch}
    source_tree = trees['models/source_choice_adapter.py']
    source_functions = [n for n in source_tree.body if isinstance(n, ast.FunctionDef)
                        and n.name in ['_first_row', '_align_token_scores', 'compute_default_source_scores']]
    assert len(source_functions) == 3
    run_nodes(source_functions, namespace)
    rec_tree = trees['models/rec_candidate_adapter.py']
    helpers = [n for n in rec_tree.body if assigned_name(n, '_COMPONENT_KEYS')
               or isinstance(n, ast.FunctionDef) and n.name == '_first_map_row']
    assert len(helpers) == 2
    run_nodes(helpers, namespace)
    rec_function = next(n for n in rec_tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'build_full_rec_query_state')
    rec_expressions = [n for n in rec_function.body if any(assigned_name(n, name)
                       for name in ['sem_maps', 'sem_prob', 'component_scores', 'default_scores'])]
    assert len(rec_expressions) == 4
    evaluator = trees['src/grounding_evaluator.py']
    methods = [n for cls in evaluator.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, ast.FunctionDef)]
    parse_gt = next(n for n in methods if n.name == '_parse_gt')
    run_nodes([parse_gt], namespace)
    evaluation = next(n for n in methods if n.name == 'evaluate_bbox_by_pos_align')
    loop = next(n for n in evaluation.body if isinstance(n, ast.For))
    last = next(i for i, n in enumerate(loop.body) if assigned_name(n, 'scores'))
    rows = []
    for token_count in [1, 2, 4]:
        tokens = token_count + 4
        probabilities = torch.full((1, 2, tokens), .025)
        probabilities[0, 0, 0] = .475
        probabilities[0, 1, 0] = .505
        probabilities[0, 0, 1:token_count+1] = .4 / token_count
        probabilities[0, 1, 1:token_count+1] = .1 / token_count
        modifier, other = token_count+1, token_count+2
        probabilities[0, 0, modifier] = .05
        probabilities[0, 1, modifier] = .31
        probabilities[0, 0, other] = .05
        probabilities[0, 1, other] = .06
        assert torch.allclose(probabilities.sum(-1), torch.ones(1, 2))
        logits = probabilities.log()
        inputs = {key: torch.zeros(1, 1, tokens) for _, key in namespace['_COMPONENT_KEYS']}
        inputs['positive_map'][0, 0, 1:token_count+1] = 1. / token_count
        inputs['modify_positive_map'][0, 0, modifier] = 1.
        inputs['other_entity_map'][0, 0, other] = 1.
        inputs['auxi_entity_positive_map'] = torch.zeros(1, 1, tokens)
        points = dict(inputs, center_label=torch.zeros(1, 1, 3), size_gts=torch.ones(1, 1, 3),
                      box_label_mask=torch.ones(1, 1), last_sem_cls_scores=logits)
        old_inputs = {k: v.clone() for k, v in inputs.items()}
        parsed = namespace['_parse_gt'](SimpleNamespace(only_root=True), points)
        native = dict(namespace, bid=0, end_points=points, sem_scores=logits.softmax(-1))
        native.update(zip(['positive_map', 'modify_positive_map', 'pron_positive_map',
                           'other_entity_map', 'auxi_entity_positive_map', 'rel_positive_map',
                           'gt_bboxes'], parsed))
        run_nodes(loop.body[:last+1], native)
        candidate = dict(namespace, inputs=inputs, batch_size=1, sem_logits=logits)
        run_nodes(rec_expressions, candidate)
        source = namespace['compute_default_source_scores'](points, inputs)
        native_scores = native['scores']
        adapter_scores = candidate['default_scores']
        assert torch.allclose(source, native_scores, atol=1e-7, rtol=0.)
        assert all(torch.equal(inputs[k], old_inputs[k]) for k in inputs)
        assert native_scores.argmax().item() == 0
        assert adapter_scores.argmax().item() == (0 if token_count == 1 else 1)
        rows.append({'main_token_count': token_count, 'probabilities': probabilities.tolist(),
                     'native_evaluator_scores': native_scores.tolist(),
                     'source_choice_scores': source.tolist(),
                     'candidate_adapter_default_scores': adapter_scores.tolist(),
                     'native_top': int(native_scores.argmax()),
                     'adapter_top': int(adapter_scores.argmax())})
    result = {'status': 'pass', 'source_hashes': hashes, 'synthetic_cases': rows,
              'scope': 'Existing score fragments; no full evaluator/model/learned reranker',
              'weights_loaded': False, 'gpu_forwards': 0, 'dataset_rows': 0,
              'optimizer_steps': 0, 'rule_changed': False,
              'metric_gain_proven': False, 'historical_source_choice_fix_already_present': True}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
