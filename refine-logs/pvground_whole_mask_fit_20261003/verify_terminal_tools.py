"""Check new read-only analysis on an already closed historical row fixture."""
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path


root = Path(__file__).parent
for name in ('collect_terminal.py', 'analyze_terminal.py'):
    ast.parse((root/name).read_text(encoding='utf-8'), feature_version=(3, 7))
tree = ast.parse((root/'collect_terminal.py').read_text(encoding='utf-8'))
probe = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'probe' for target in node.targets))
ast.parse(probe, feature_version=(3, 7))
source = importlib.util.spec_from_file_location('range_terminal_analysis', root/'analyze_terminal.py')
analysis = importlib.util.module_from_spec(source)
source.loader.exec_module(analysis)
fixture = root.parent/'pvground_fused_support_20261002/complete_tail_fused_retry/arm/formal'
receipt = json.loads((fixture/'receipt.json').read_bytes())
raw = (fixture/'rows.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
rows = [json.loads(line) for line in raw.splitlines()]
assert len(rows) == 9508
diagnostics = {mode: analysis.refinement(rows, mode) for mode in ('bbs', 'bbf')}
assert diagnostics['bbs']['50']['coarse_selected_hits'] == 4412
assert diagnostics['bbs']['50']['final_selected_hits'] == 4406
assert diagnostics['bbs']['50']['repairs'] == 15
assert diagnostics['bbs']['50']['damages'] == 21
for mode in ('bbs', 'bbf'):
    for label in ('25', '50'):
        assert diagnostics[mode][label]['final_selected_hits'] == receipt['metrics'][mode]['rec_hits'+label]
record = dict(status='SOURCE_READY_NOT_CURRENT_TERMINAL_RESULT',
    time_cst=datetime.datetime.now().astimezone().isoformat(), python37_ast_pass=True,
    historical_fixture=str(fixture), historical_rows_sha256=receipt['rows_sha256'],
    actual_historical_row_check=diagnostics,
    tool_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('collect_terminal.py', 'analyze_terminal.py')},
    model_forwards=0, optimizer_updates=0, remote_connections=0, weights_downloaded=0,
    weights_deleted=0, current_pair_terminal_collection_started=False,
    current_pair_terminal_analysis_started=False)
(root/'terminal_tools_source_check.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'], historical_rows=9508,
    historical_bbs_strict_coarse_final=[4412, 4406], historical_repairs_damages=[15, 21],
    cpu_final_threshold_changes={mode: diagnostics[mode]['cpu_final_threshold_changes'] for mode in diagnostics},
    current_pair_terminal_results_available=False)))
