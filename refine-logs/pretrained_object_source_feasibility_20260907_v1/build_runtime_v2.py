import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path('/root/autodl-tmp/mcln_openshape_object_source_20260907_v1');spec=json.loads((r/'runtime_spec_v2.json').read_text())
assert (r/'build.exit').read_text().strip()=='1'
assert not (r/'deps/networkx').exists()
for name,entry in spec['files'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==entry['sha256'],name
subprocess.check_call([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(r/'deps'),str(r/'assets/networkx-2.6.3-py3-none-any.whl')])
current=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'])
assert json.loads(current)==json.loads((r/'base_packages_before.json').read_text())
(r/'base_packages_after_v2.json').write_bytes(current)
subprocess.check_call(['bash',str(r/'witness.sh')])
(r/'runtime_build_receipt_v2.json').write_text(json.dumps(dict(status='kernel_pass',base_packages_unchanged=True,spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),original_build_exit=1)))
