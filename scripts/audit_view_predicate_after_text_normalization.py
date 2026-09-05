"""Compare G0 raw-text and actual pre-parser view predicates without training.

Executes frozen caption normalization and both actual augmentation methods.
Only prefix-sensitive rows need the real parser to resolve its generic prefix.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compile_statements(statements, filename, namespace):
    module = ast.parse("")
    module.body = statements
    code = compile(ast.fix_missing_locations(module), str(filename), "exec")
    exec(code, namespace)


def predicate_class(source, filename):
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "Joint3DDataset")
    node = ast.parse("class Joint3DDataset: pass").body[0]
    node.body = [method for method in cls.body
                 if isinstance(method, ast.FunctionDef)
                 and method.name in ("_is_view_dep", "_augment_nr3d")]
    namespace = {"re": re}
    compile_statements([node], filename, namespace)
    return namespace["Joint3DDataset"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    from scripts.run_nr3d_view_pair_role import read_train_rows
    from scripts.nr3d_view_pair_contract import split_rows, validate_census
    from src.joint_det_dataset import sng_parser

    paths = {role: opt.pair_root / "inputs_v3" / (role + "_source") /
             "src/joint_det_dataset.py" for role in ("old", "fixed")}
    sources = {role: path.read_bytes() for role, path in paths.items()}
    classes = {role: predicate_class(sources[role], paths[role]) for role in paths}
    tree = ast.parse(sources["fixed"])
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "Scene_graph_parse")
    loop = next(node for node in function.body if isinstance(node, ast.For))
    start = next(i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "caption")
    terminal = ast.dump(ast.parse("anno['utterance'] = caption").body[0])
    stop = next(i for i, node in enumerate(loop.body) if ast.dump(node) == terminal)
    statements = loop.body[start:stop + 1]
    module = ast.parse("")
    module.body = statements
    normalizer = compile(ast.fix_missing_locations(module), str(paths["fixed"]), "exec")

    rows = read_train_rows(Path("/root/autodl-tmp/DATA_ROOT"))
    partitions = split_rows(rows)
    validate_census(rows, partitions)
    normalized = []
    prefix_sensitive = 0
    prefix_added = 0
    for row in rows:
        namespace = {"anno": {"dataset": "nr3d", "utterance": row["utterance"]}}
        exec(normalizer, namespace)
        caption = namespace["anno"]["utterance"]
        prefixed = "This is an object . " + caption
        if any(cls._augment_nr3d(caption) != cls._augment_nr3d(prefixed)
               for cls in classes.values()):
            prefix_sensitive += 1
            nodes, _ = sng_parser.parse(caption, target_selection="first_object")
            if not nodes or nodes[0]["node_id"] != 0:
                caption = prefixed
                prefix_added += 1
        normalized.append(caption)
    groups = {}
    for group, ids in partitions.items():
        result = {"rows": len(ids)}
        old_view = [classes["old"]._is_view_dep(rows[i]["utterance"]) for i in ids]
        fixed_view = [classes["fixed"]._is_view_dep(rows[i]["utterance"]) for i in ids]
        result["raw_view_label_old_miss_fixed_hit"] = sum(
            not a and b for a, b in zip(old_view, fixed_view))
        for stage, texts in (("raw_augmentation", [row["utterance"] for row in rows]),
                             ("post_parser_augmentation", normalized)):
            old = [classes["old"]._augment_nr3d(texts[i]) for i in ids]
            fixed = [classes["fixed"]._augment_nr3d(texts[i]) for i in ids]
            result[stage] = {"old_rotate_allowed": sum(old),
                             "fixed_rotate_allowed": sum(fixed),
                             "old_allow_fixed_block": sum(a and not b for a, b in zip(old, fixed)),
                             "old_block_fixed_allow": sum(not a and b for a, b in zip(old, fixed))}
        groups[group] = result
    observed = json.loads((opt.pair_root / "results/old/training.json").read_text())
    assert groups["fit"]["post_parser_augmentation"]["old_rotate_allowed"] == observed["rotate_allowed_rows"]
    result = {"schema": "mcln-view-predicate-post-normalization-v1",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": {role: hashlib.sha256(data).hexdigest() for role, data in sources.items()},
              "normalization_ast_sha256": hashlib.sha256(ast.dump(module).encode()).hexdigest(),
              "train_rows": len(rows), "formal_validation_rows": 0, "optimizer_steps": 0,
              "prefix_sensitive_rows_parsed": prefix_sensitive,
              "prefix_added_sensitive_rows": prefix_added,
              "old_fit_observed_rotate_allowed": observed["rotate_allowed_rows"],
              "old_fit_observation_matches": True, "groups": groups}
    with opt.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
