"""Prepare guarded consumers; this does not launch fit or poll the active M0."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
old=root.parent/'pvground_mask_support_correction_20261008_v2'
text=(old/'launch_fit_authorized.py').read_text().replace('\r\n','\n')


def replace_once(before,after):
    global text
    assert text.count(before)==1,before
    text=text.replace(before,after)


replace_once("review = json.loads((local / 'FIT_LAUNCH_SOURCE_REVIEW.json').read_bytes())",
    "review = json.loads((local / 'FIT_LAUNCH_SOURCE_REVIEW.json').read_bytes())\n"
    "actual = json.loads((local / 'PREFLIGHT_ACTUAL_REVIEW.json').read_bytes())\n"
    "assert actual['execution_scope'] == 'ACTUAL_PREFLIGHT'\n"
    "assert actual['verdict'] in ('PASS', 'WARN') and not actual['blocking_findings']\n")
replace_once("    assert proof['witnesses'][0]['arms'][arm]['zero_update_mask_and_box_exact']",
    "    assert proof['witnesses'][0]['arms'][arm]['zero_update_warm_mask_exact']\n"
    "    assert proof['witnesses'][0]['arms'][arm]['warm_arm_mask_and_box_exact']")
replace_once("for key in ('base_terminal','selected_terminal'):",
    "for key in ('base_terminal','selected_terminal','warm_support_terminal'):")
replace_once("assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()",
    "assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()\n"
    "capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()\n"
    "assert len(capacity)==1\n"
    "index,name,used,total=[value.strip() for value in capacity[0].split(',')]\n"
    "assert int(index)==0 and 'A100' in name and 40000<=int(total)<=45000 and int(used)<500")
replace_once("screen = 'pvg_support_correction_fit_20261008'",
    "screen = 'pvg_compressed_support_fit_20261008'")
replace_once("estimated_seconds=19267,first_check_seconds=18967,later_poll_seconds=240,",
    "support_prior_updates=3723,support_total_updates_at_terminal=7446,\n"
    "    estimated_seconds=25415,first_check_seconds=25115,later_poll_seconds=240,")
replace_once("estimate_basis='Previous one-parent complete pair all3 phases19266.40s. No intermediate go/no-go decision is required in this fixed-budget pair; first check5min before estimated whole-job finish. Only then poll240s if needed, revise estimate from actual phase progress, and never infer completion from time.'",
    "estimate_basis='Last actual closed shared-parent pair all3 phases25415.26s. Fixed-budget warm continuation with the same number of forwards/updates; first check5min before estimated complete pair finish, then240s if needed. No middle go/no-go metric query is required; never infer completion or restart from elapsed time.'")
files={'launch_fit_authorized.py':text.encode()}
for name in ('observe_fit_authorized.py','collect_closed_fit_authorized.py'):
    files[name]=(old/name).read_bytes()
for name,data in files.items():
    ast.parse(data)
    with (root/name).open('xb') as stream:stream.write(data)
(root/'FIT_CONSUMERS_PREPARED.json').write_text(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',
    files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
    active_preflight_not_polled=True,formal_fit_launched=False,
    exact_m0_receipt_and_actual_audit_required=True,
    source_review_required='FIT_LAUNCH_SOURCE_REVIEW.json',expected_seconds=25415,
    first_observation_seconds=25115,later_observation_seconds=240),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='FIT_CONSUMERS_PREPARED_NOT_EXECUTED',files=len(files),NN_queries=0)))
