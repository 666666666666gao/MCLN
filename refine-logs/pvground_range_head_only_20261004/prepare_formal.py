"""Prepare the same bounded source pair under the reviewed stable-G protocol."""
import ast
import hashlib
import json
from pathlib import Path

local=Path(__file__).parent
prior=local.parent/'pvground_whole_mask_fit_20261003'
assert not (local/'formal_preparation.json').exists()
original=prior/'controller.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='8155cc9002f0f910a7b8659c7f0170ce212f825d716217c1ad78407d20022d49'
text=original.read_text(encoding='utf-8')
before="    assert proof['support_arm'] == arm and proof['use_whole_range'] == (arm == 'whole_range')\n"
assert text.count(before)==1
text=text.replace(before,"    assert arm_spec['head_only'] and arm_spec['use_whole_range'] == (arm == 'whole_range')\n"
    "    assert proof['support_arm'] == 'whole_range' and proof['use_whole_range']\n"
    "    assert proof['head_only'] and proof['original_g_state_unchanged']\n"
    "    assert proof['upstream_running_state_eval']\n")
assert text.count("str(root/'run_whole_mask_fit.py')")==1
text=text.replace("str(root/'run_whole_mask_fit.py')","str(root/'run_range_head_only.py')")
before="    assert fit['fit_seen_exactly_once'] and fit['fit_rows'] == 29778\n"
assert text.count(before)==1
text=text.replace(before,before+"    assert fit['head_only'] and fit['original_g_state_unchanged']\n")
ast.parse(text,feature_version=(3,7))
(local/'controller.py').write_text(text,encoding='utf-8',newline='\n')
spec=json.loads((local/'preflight_spec.json').read_bytes())
for arm in ('local_range','whole_range'):
    item=dict(spec)
    item.update(root='/root/autodl-tmp/pvground_range_head_only_fit_20261004/'+arm,
        support_arm=arm,use_whole_range=arm=='whole_range',preflight_only_at_install=False,
        preflight_root='/root/autodl-tmp/pvground_range_head_only_preflight_20261004/whole_range')
    (local/(arm+'_spec.json')).write_text(json.dumps(item,indent=2)+'\n',encoding='utf-8')
record=dict(status='SOURCE_ONLY_NOT_LAUNCHED',original_controller_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    current_controller_sha256=hashlib.sha256((local/'controller.py').read_bytes()).hexdigest(),
    sole_probe_covers='same cached zero-head local/whole pair plus two whole-range head updates',
    requires_actual_new_protocol_preflight=True,fit_rows_per_arm=29778,updates_per_arm=3723,
    head_only=True,weights_created=0,model_updates=0)
(local/'formal_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
