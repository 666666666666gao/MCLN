"""Read-only CPU comparison of pinned PV-Ground and native ReferIt3D inputs."""
import argparse
import ast
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method(tree, name):
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Joint3DDataset')
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def compile_functions(tree):
    selected = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'resolve_referit3d_csv')] if any(isinstance(node, ast.FunctionDef) and node.name == 'resolve_referit3d_csv' for node in tree.body) else []
    selected += [method(tree, name) for name in ['_is_view_dep', '_augment_nr3d']]
    module = copy.deepcopy(tree)
    module.body = copy.deepcopy(selected)
    for node in module.body:
        node.decorator_list = []
    namespace = {'os': os, 're': re}
    exec(compile(module, '<pinned_dataset_functions>', 'exec'), namespace)
    namespace['Joint3DDataset'] = SimpleNamespace(_is_view_dep=namespace['_is_view_dep'])
    return namespace


def annotation_expression(tree, dataset):
    node = method(tree, 'load_%s_annos' % dataset)
    expression = next(child.value for child in ast.walk(node)
                      if isinstance(child, ast.Assign) and isinstance(child.value, ast.ListComp)
                      and any(isinstance(target, ast.Name) and target.id == 'annos' for target in child.targets))
    return compile(ast.Expression(expression), '<pinned_annotation_comprehension>', 'eval')


def normalization(tree):
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'Scene_graph_parse')
    loop = next(node for node in function.body if isinstance(node, ast.For))
    statements = []
    collecting = False
    for node in loop.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'caption' for target in node.targets):
            collecting = True
        if collecting:
            statements.append(node)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
                break
    assert statements and isinstance(statements[-1].value, ast.Name) and statements[-1].value.id == 'caption'
    module = copy.deepcopy(tree)
    module.body = copy.deepcopy(statements)
    return compile(module, '<pinned_caption_normalization_before_parser>', 'exec'), ast.dump(module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--input-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    manifest = json.loads(args.input_manifest.read_bytes())
    native = Path(manifest['model_source'])
    assert sha(native / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    native_manifest = json.loads((native / 'appearance_source_manifest.json').read_bytes())['files']
    upstream = args.runtime / 'PV-Ground'
    bundle = json.loads((args.runtime / 'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']
    roots = {'native': native, 'upstream': upstream}
    trees = {}
    functions = {}
    files = {}
    for label, root in roots.items():
        paths = ['src/joint_det_dataset.py', 'data/cls_results.json'] + [
            'data/meta_data/%s_%s_scans.txt' % (dataset, split)
            for dataset in ['nr3d', 'sr3d'] for split in ['train', 'test']]
        for name in paths:
            digest = sha(root / name)
            expected = native_manifest[name] if label == 'native' else bundle['files'][name]['sha256']
            assert digest == expected, (label, name)
            files[label + '/' + name] = {'path': str(root / name), 'sha256': digest}
        trees[label] = ast.parse((root / 'src/joint_det_dataset.py').read_text())
        functions[label] = compile_functions(trees[label])
    native_normalize, native_norm_ast = normalization(trees['native'])
    _, upstream_norm_ast = normalization(trees['upstream'])
    assert native_norm_ast == upstream_norm_ast
    assert files['native/data/cls_results.json']['sha256'] == files['upstream/data/cls_results.json']['sha256']
    cls_results = json.loads((native / 'data/cls_results.json').read_bytes())
    datasets = {}
    for dataset in ['nr3d', 'sr3d']:
        csv_path = Path(functions['native']['resolve_referit3d_csv'](manifest['data_root'], dataset + '.csv'))
        with csv_path.open() as handle:
            reader = csv.reader(handle)
            headers = {name: index for index, name in enumerate(next(reader))}
            rows = list(reader)
        splits = {}
        split_scenes = {}
        for split in ['train', 'test']:
            meta = 'data/meta_data/%s_%s_scans.txt' % (dataset, split)
            scan_lists = {label: ast.literal_eval((root / meta).read_text()) for label, root in roots.items()}
            assert scan_lists['native'] == scan_lists['upstream'], (dataset, split)
            scan_ids = set(scan_lists['native'])
            split_scenes[split] = scan_ids
            scope = {'headers': headers, 'scan_ids': scan_ids, 'split': split, 'dset': dataset,
                     'eval': ast.literal_eval}
            selected = {}
            for label in roots:
                selected[label] = eval(annotation_expression(trees[label], dataset), dict(scope, csv_reader=iter(rows)))
            assert selected['native'] == selected['upstream'], (dataset, split)
            annos = selected['native']
            scene_ids = sorted({anno['scan_id'] for anno in annos})
            assert all(scene in cls_results for scene in scene_ids)
            identity = hashlib.sha256()
            raw_disagreements = 0
            normalized_disagreements = 0
            raw_restricted = {'native': 0, 'upstream': 0}
            normalized_restricted = {'native': 0, 'upstream': 0}
            for index, anno in enumerate(annos):
                identity.update((json.dumps([dataset, split, index, anno['scan_id'], anno['target_id'], anno['utterance']], ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                if dataset == 'nr3d':
                    raw = {label: not functions[label]['_augment_nr3d'](anno['utterance']) for label in roots}
                    normalized = {'anno': dict(anno)}
                    exec(native_normalize, normalized)
                    caption = normalized['anno']['utterance']
                    norm = {label: not functions[label]['_augment_nr3d'](caption) for label in roots}
                    raw_disagreements += raw['native'] != raw['upstream']
                    normalized_disagreements += norm['native'] != norm['upstream']
                    for label in roots:
                        raw_restricted[label] += raw[label]
                        normalized_restricted[label] += norm[label]
            result = {'rows': len(annos), 'declared_split_scenes': len(scan_ids), 'expression_scenes': len(scene_ids),
                      'raw_ordered_identity_sha256': identity.hexdigest(), 'same_selected_annotations': True,
                      'predicted_class_scene_coverage': len(scene_ids), 'scenes': scene_ids}
            if dataset == 'nr3d':
                result['view_augmentation_function_comparison'] = {
                    'raw_csv_disagreements': raw_disagreements, 'raw_csv_restricted_rows': raw_restricted,
                    'normalized_before_parser_disagreements': normalized_disagreements,
                    'normalized_before_parser_restricted_rows': normalized_restricted,
                    'scope': 'pure source functions before scene-graph parse; parser-added prefix and stochastic augmentation not executed'}
            splits[split] = result
        assert not split_scenes['train'].intersection(split_scenes['test'])
        assert splits['test']['rows'] == {'nr3d': 7899, 'sr3d': 17726}[dataset]
        datasets[dataset] = {'csv_path': str(csv_path), 'resolved_csv_path': str(csv_path.resolve()),
                             'csv_sha256': sha(csv_path), 'csv_rows': len(rows), 'splits': splits,
                             'declared_train_test_scene_overlap': 0}
    receipt = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'elapsed_seconds': time.time() - started, 'pv_commit': bundle['commit'],
               'native_source_manifest_sha256': manifest['source_manifest_sha256'], 'files': files,
               'datasets': datasets, 'same_predicted_class_file': True, 'same_preparse_normalization_ast': True,
               'identity_schema': '[dataset, train_or_test, selected_row_index, scan_id, target_id, raw_utterance] JSONL UTF-8',
               'torch_imported': 'torch' in sys.modules, 'dataset_getitem_calls': 0, 'gpu_forwards': 0,
               'optimizer_steps': 0, 'weight_downloads': 0, 'formal_evaluation_rows': 0}
    assert not receipt['torch_imported']
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'time_cst': receipt['time_cst'], 'elapsed_seconds': receipt['elapsed_seconds'],
                      'datasets': {name: {split: {key: value for key, value in info.items() if key != 'scenes'}
                                           for split, info in data['splits'].items()} for name, data in datasets.items()},
                      'output_sha256': sha(args.output)}), flush=True)


if __name__ == '__main__':
    main()
