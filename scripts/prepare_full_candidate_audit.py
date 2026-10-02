"""Reuse the tested restore/loader/evaluator; add a read-only candidate dump."""
import ast
from pathlib import Path

root = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
source = (root / 'run_pvground_tail_support_retry.py').read_text(encoding='utf-8')
prefix = source[:source.index('    def step(batch,optimizer,update):')]
prefix = prefix.replace("    parser.add_argument('--mode', choices=['cpu', 'preflight', 'train', 'formal'], required=True)",
    "    parser.add_argument('--mode', choices=['formal'], required=True)\n"
    "    parser.add_argument('--audit-output', type=Path, required=True)\n"
    "    parser.add_argument('--limit', type=int, choices=[8,9508], required=True)")
prefix = prefix.replace("    output = Path(spec['root'])", "    output = Path(spec['root'])\n"
    "    sys.path.insert(0,str(output))\n"
    "    audit_output = args.audit_output\n"
    "    audit_output.mkdir()\n"
    "    assert args.mode == 'formal'")
prefix = prefix.replace("write_json(output/'imports.json'", "write_json(audit_output/'imports.json'")
prefix = prefix.replace("write_json(output/'load.json'", "write_json(audit_output/'load.json'")
prefix = prefix.replace('fresh_optimizer=True,', 'fresh_optimizer=False, optimizer_constructed=False,')
tail = (root / 'full_candidate_audit_body.py').read_text(encoding='utf-8')
destination = root / 'run_full_candidate_audit.py'
destination.write_text(prefix+tail, encoding='utf-8')
tree = ast.parse(destination.read_text(encoding='utf-8'))
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    and n.func.attr in ('backward','step','get_optimizer') for n in ast.walk(tree))
print(destination)
