import ast
import json
import sys
from pathlib import Path
p = Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_metadata_20261010/read_referit_metadata_authorized.py')
src = p.read_text(encoding='utf-8')
tree = ast.parse(src)
compile(tree, str(p), 'exec')
remote = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets))
remote_tree = ast.parse(remote, feature_version=(3, 7))
native_path = Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py')
native_tree = ast.parse(native_path.read_text(encoding='utf-8'), feature_version=(3, 7))
resolver = [n for n in native_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_referit3d_csv']
assert len(resolver) == 1
suffix_checks = []
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'write_text' and isinstance(n.args[0], ast.BinOp):
        suffix = ast.literal_eval(n.args[0].right)
        candidate = json.dumps({'example': True}) + suffix
        try:
            json.loads(candidate)
        except json.JSONDecodeError as error:
            parse_result = {'valid_json': False, 'error': str(error)}
        else:
            parse_result = {'valid_json': True}
        suffix_checks.append(dict(line=n.lineno, literal_repr=repr(suffix), literal_char_codes=list(map(ord, suffix)), **parse_result))
print(json.dumps(dict(local_python=sys.version, host_syntax_compile='PASS', remote_python37_grammar_parse='PASS', native_source_python37_grammar_parse='PASS', resolver_count=len(resolver), remote_imports=[ast.unparse(n) for n in remote_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))], remote_ast_compile_statement=[ast.unparse(n) for n in remote_tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'exec'], suffix_checks=suffix_checks, reader_executed=False, remote_python37_runtime_executed=False), indent=2))

