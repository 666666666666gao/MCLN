import ast,datetime,hashlib,json,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).parent
tree=ast.parse((root/'post_training.py').read_text())
statement=[node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='controller' for t in node.targets)]
assert len(statement)==1
formal=Path(json.loads((root/'preparation.json').read_text())['formal_directory'])
controller=eval(compile(ast.Expression(statement[0].value),'<controller-only>','eval'),{'formal':formal,'shlex':shlex,'sys':sys})
assert controller.count("printf '%s\\n'")==1
checked=subprocess.run(['bash','-n'],input=controller.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert checked.returncode==0,checked.stderr
line=[line for line in controller.splitlines() if line.startswith('printf ')][0]
probe=subprocess.check_output(['bash','-c','status=0\n'+line.replace(' > controller.exit','')])
assert probe==b'0\n',repr(probe)
result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'status':'bash_syntax_and_exact_exit_record_pass','controller_sha256':hashlib.sha256(controller.encode()).hexdigest(),
 'post_training_source_sha256':hashlib.sha256((root/'post_training.py').read_bytes()).hexdigest(),
 'gpu_forwards':0,'optimizer_steps':0,'checkpoint_writes':0,'post_training_entry_executed':False}
with (root/'controller_check.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print(json.dumps(result))
