"""Prepare a CPU-only terminal recount for the warm, nonzero-output pair."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
old=root.parent/'pvground_mask_support_correction_20261008_v2/postrun/analyze_support_correction_formal.py'
text=old.read_text().replace('\r\n','\n')


def replace_once(before,after):
    global text
    assert text.count(before)==1,before
    text=text.replace(before,after)


replace_once("            if stage == 'initial_formal':\n                for arm in ARMS:\n                    np.testing.assert_array_equal(batch[arm], batch['parent'])",
    "            if stage == 'initial_formal':\n"
    "                np.testing.assert_array_equal(batch['content'], batch['box_conditioned'])")
replace_once("        table.append(dict(arm=arm, stage=stage, optimizer_updates=0 if stage == 'initial_formal' else 3723,",
    "        table.append(dict(arm=arm, stage=stage, optimizer_updates=0 if stage == 'initial_formal' else 3723,\n"
    "            support_prior_updates=3723, support_total_updates=3723 if stage == 'initial_formal' else 7446,\n"
    "            geometry_encoding='signed_log' if arm == 'box_conditioned' else 'zero',")
replace_once("    protected_path = ROOT.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal/rows.jsonl'\n    protected = read_rows(protected_path)",
    "    protected_path = ROOT.parent / 'pvground_mask_support_correction_20261008_v2/complete_fit/formal/rows.jsonl'\n"
    "    protected = [dict(row, bbs=row['arms']['content']) for row in read_rows(protected_path)]")
assert text.count('4848')==3
text=text.replace('4848','4856')
replace_once("Zero-output initial evaluations are diagnostics, not new checkpoint gains.",
    "Warm initial reevaluations are diagnostics, not new checkpoint gains.")
replace_once("        fit_rows_per_arm=29778,optimizer_updates_per_arm=3723,seed=2027,table=table,",
    "        fit_rows_per_arm=29778,optimizer_updates_per_arm=3723,seed=2027,table=table,\n"
    "        support_prior_updates=3723,support_total_updates_at_terminal=7446,\n"
    "        protected_formal_rows_sha256=digest(protected_path),\n"
    "        protected_model_checkpoint_sha256=read_json(COMPLETE/'pair_spec.json')['warm_support_terminal_sha256'],")
ast.parse(text)
destination=root/'postrun/analyze_compressed_geometry_formal.py'
destination.parent.mkdir(exist_ok=True)
with destination.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
(root/'CLOSED_ANALYSIS_PREPARED.json').write_text(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',
    analysis_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    inherited_analysis_source=str(old),inherited_analysis_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    old_source_unchanged=True,neural_or_optimizer_execution=False,new_result_claim=False,
    required_actual_closed_archive='complete_fit/INTAKE.json',
    protected_hits=[5598,4856],target_hits=[5620,4764],all256_retained=True,
    changes=['Warm initial arms equal each other, not uncorrected parent',
             'Historical protected rows are retained trained content',
             'Prior and new update counts and signed-log encoding disclosed']),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='WARM_CLOSED_ANALYSIS_PREPARED_NOT_EXECUTED',NN_queries=0)))
