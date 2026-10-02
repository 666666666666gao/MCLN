"""Prepare the exact two-arm semantic-only comparison from the archived G run."""
import hashlib
import json
from pathlib import Path
import py_compile

local = Path(__file__).parent
parent = Path(r'C:\Users\gb\.codex\tmp\pvground_g_p2_20261002')
remote = '/root/autodl-tmp/pvground_candidate_consistency_20261003'
base = json.loads((parent / 'g_control_spec.json').read_bytes())
module_sha = hashlib.sha256((local / 'pvground_candidate_consistency.py').read_bytes()).hexdigest()
for arm, active in (('preflight', True), ('g_control', False), ('g_consistent', True)):
    spec = dict(base)
    for name in ('files', 'disk_budget', 'disk_free_before', 'preflight_only_at_install'):
        spec.pop(name)
    spec.update(root=remote + '/' + arm, p2=False, semantic_consistency=active,
                consistency_module_sha256=module_sha,
                comparison='same original G, model/source/RNG/3723 updates; final contrastive semantic correspondence only')
    (local / (arm + '_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
text = (parent / 'complete/source/pair.py').read_text(encoding='utf-8')
text = text.replace("root/'g_p2/spec.json'", "root/'g_consistent/spec.json'")
text = text.replace("('g_p2','train')", "('g_consistent','train')")
text = text.replace("('g_p2','formal')", "('g_consistent','formal')")
needle = "    assert preflight['batch_size']==8 and preflight['g_strict_restore']\n"
assert text.count(needle) == 1
text = text.replace(needle, needle + "    assert preflight['semantic_consistency'] and preflight['new_model_states'] == 0\n")
needle = "    record={'status':'running'"
assert text.count(needle) == 1
text = text.replace(needle,
    "    assert not (root/'pair_status.json').exists()\n"
    "    assert shutil.disk_usage(root).free >= 3*preflight['serialization_bytes'] + 256*1024**2, 'insufficient two-arm save space'\n"
    + needle)
(local / 'pair.py').write_text(text, encoding='utf-8')
py_compile.compile(str(local / 'pair.py'), doraise=True)
print(json.dumps(dict(prepared=True, arms=['g_control', 'g_consistent'], preflight_steps=2,
                      model_source=base['model_source'], new_model_states=0)))
