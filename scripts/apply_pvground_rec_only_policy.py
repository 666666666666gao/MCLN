"""Update only the not-yet-executed E formal promotion policy and its waiter."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import shlex
import paramiko
from pvground_rec_only_policy import rec_only_sources, replace_once

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_fixed_memory_v1'
archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
names=['evaluate.py','audit.py','plan.md']
old={name:(archive/name).read_bytes() for name in names}
updated=dict(zip(names,[text.encode() for text in rec_only_sources(*(old[name].decode() for name in names))]))
for name in names[:2]:compile(updated[name],name,'exec')
# Test both independently implemented REC decisions, including low Mask and one-hit failures.
def promotion_function(raw):
    tree=ast.parse(raw)
    node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='promotion_check')
    tree.body=[node]
    scope={};exec(compile(tree,'promotion_only','exec'),scope)
    return scope['promotion_check']
new_promote=promotion_function(updated['evaluate.py'])
old_promote=promotion_function(old['evaluate.py'])
audit_tree=ast.parse(updated['audit.py'])
audit_checks=next(node.value for node in ast.walk(audit_tree) if isinstance(node,ast.Assign)
    and any(isinstance(target,ast.Name) and target.id=='checks' for target in node.targets)
    and isinstance(node.value,ast.Dict) and any(isinstance(key,ast.Constant) and key.value=='rec25_historical_v99' for key in node.value.keys))
audit_expression=compile(ast.Expression(body=audit_checks),'independent_checks','eval')
cases=[]
for rec25,rec50,expected in [(5572,4797,True),(5571,4797,False),(5572,4796,False),(5571,4796,False),(5600,4900,True)]:
    for mask in [0,9508]:
        metrics=dict(rec_hits25=rec25,rec_hits50=rec50,mask_hits25=mask,mask_hits50=mask,mask_miou=100. if mask else 0.)
        actual=new_promote(metrics);independent=eval(audit_expression,{},dict(candidate=metrics))
        assert actual['advance_to_nr3d_sr3d_rec']==expected and all(independent.values())==expected
        assert actual['checks']==independent and len(independent)==2 and actual['scanrefer_mask_gate'] is False
        cases.append(dict(metrics=metrics,expected=expected,actual=actual,independent_checks=independent))
assert not old_promote(cases[0]['metrics'])['advance_to_nr3d_sr3d_rec']
# Every evaluator function except the promotion decision is byte-for-byte AST-equivalent,
# apart from main's protocol metadata; model/inference/Mask calculation code is unchanged.
before=ast.parse(old['evaluate.py']);after=ast.parse(updated['evaluate.py'])
for node in before.body:
    if isinstance(node,ast.FunctionDef) and node.name not in ['promotion_check','main']:
        paired=next(item for item in after.body if isinstance(item,ast.FunctionDef) and item.name==node.name)
        assert ast.dump(node)==ast.dump(paired),node.name

# Reuse the already executed exact-waiter replacement operation, with preserved provenance.
tree=ast.parse((repo/'scripts/reschedule_pvground_fixed_memory_formal.py').read_text())
remote=ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
    and any(isinstance(target,ast.Name) and target.id=='remote_code' for target in node.targets)))
remote=replace_once(remote,'assert parent==8872','assert parent==10116')
remote=remote.replace('2026-09-17T20:41:11.386051+08:00','2026-09-17T19:15:00+08:00')
remote=remote.replace('before_reschedule_20260917','before_rec_only_policy_20260917')
remote=remote.replace('reschedule_receipt.json','rec_only_policy_receipt.json')
remote=replace_once(remote,"for name in ['launch.json','spec.json','run.log']:","for name in ['launch.json','spec.json','run.log','evaluate.py','audit.py','plan.md']:")
anchor="spec['first_check_cst']='2026-09-17T19:15:00+08:00'"
digests={name:hashlib.sha256(raw).hexdigest() for name,raw in updated.items()}
install="new_digests="+repr(digests)+"\nfor name,digest in new_digests.items():\n    staged=root/'rec_only_staged'/name\n    assert sha(staged)==digest\n    (root/name).write_bytes(staged.read_bytes())\n    spec['files'][name]=digest\n"
remote=replace_once(remote,anchor,install+anchor)
remote=replace_once(remote,'formal_code_unchanged=True,','formal_code_unchanged=False,promotion_policy_only=True,scanrefer_mask_gate=False,')
remote=replace_once(remote,"reason='Measured fit and endpoint ETA 19:20-19:30; remove obsolete conservative wait to20:41'", "reason='Explicit user instruction2026-09-17: ScanRefer promotion uses only Acc0.25 and Acc0.50; retain Mask diagnostics and training loss'")
compile(remote,'apply_rec_only_remote.py','exec')
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for name,raw in old.items():
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw,name
sftp.mkdir(root+'/rec_only_staged')
for name,raw in updated.items():
    with sftp.open(root+'/rec_only_staged/'+name,'wb') as stream:stream.write(raw)
with sftp.open(root+'/apply_rec_only_remote.py','wb') as stream:stream.write(remote.encode())
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/apply_rec_only_remote.py'),timeout=30)
output=stdout.read().decode();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
(archive/'before_rec_only_policy_20260917').mkdir()
for name in ['launch.json','spec.json','run.log']+names:
    sftp.get(root+'/before_rec_only_policy_20260917/'+name,str(archive/'before_rec_only_policy_20260917'/name))
for name in ['launch.json','spec.json','rec_only_policy_receipt.json','apply_rec_only_remote.py']+names:
    sftp.get(root+'/'+name,str(archive/name))
test=dict(status='pass',scope='synthetic promotion boundary fixtures only; no benchmark evaluation',cases=cases,
          old_gate_rejects_low_mask=True,new_gate_accepts_low_mask_when_rec_passes=True,formal_rows=0)
test_raw=(json.dumps(test,indent=2)+'\n').encode()
(archive/'rec_only_policy_tests.json').write_bytes(test_raw)
with sftp.open(root+'/rec_only_policy_tests.json','wb') as stream:stream.write(test_raw)
sftp.close();client.close();print(output)
