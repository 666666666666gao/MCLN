import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path('/root/autodl-tmp/mcln_openshape_object_source_20260907_v1')
spec=json.loads((r/'runtime_spec.json').read_text())
assert not (r/'deps').exists()
for name,entry in spec['files'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==entry['sha256'],name
before=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'])
(r/'base_packages_before.json').write_bytes(before)
for name in ['dgl_cu111-0.9.1.post1-cp37-cp37m-manylinux1_x86_64.whl','torch.redstone-0.0.6-py3-none-any.whl']:
    subprocess.check_call([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(r/'deps'),str(r/'assets'/name)])
after=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'])
(r/'base_packages_after.json').write_bytes(after)
assert json.loads(before)==json.loads(after)
subprocess.check_call(['bash',str(r/'witness.sh')])
(r/'runtime_build_receipt.json').write_text(json.dumps(dict(status='kernel_pass',base_packages_unchanged=True,spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest())))
