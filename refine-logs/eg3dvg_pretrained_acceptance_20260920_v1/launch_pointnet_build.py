import hashlib,json,os,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;e=Path('/root/mcln_eg3dvg_torch112_20260920_v1');b=e/'pointnet2_build';b.mkdir();(b/'pointnet2').mkdir();(b/'pointnet2/__init__.py').write_text('')
source=r/'source/pointnet2/_ext_src'
sources=sorted(str(p) for p in (source/'src').iterdir() if p.suffix in ['.cpp','.cu'])
manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
spec=json.loads((r/'env_torch112.json').read_text());spec['pointnet2_build']={'source_commit':'174e34894aea6513442da6b5dfa9b3e2bf8a1efa','source_files':manifest,'cuda_home':'/usr/local/cuda-11.6','architecture':'8.0','max_jobs':2,'reason':'old Torch1.10 binary fails real import with undefined TensorImpl ABI symbol'}
digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for p in [r/'env_torch112.json',e/'env_spec.json']:p.write_text(json.dumps(spec,indent=2))
(e/'env_spec.sha256').write_text(digest+'\n')
setup="from setuptools import setup\nfrom torch.utils.cpp_extension import BuildExtension, CUDAExtension\nsetup(name='pointnet2',version='0.0.0',packages=['pointnet2'],ext_modules=[CUDAExtension('pointnet2._ext',sources="+repr(sources)+",include_dirs=["+repr(str(source/'include'))+"],extra_compile_args={'cxx':['-O3','-std=c++17','-fPIC'],'nvcc':['-O3','-std=c++17','--use_fast_math']})],cmdclass={'build_ext':BuildExtension})\n"
(b/'setup.py').write_text(setup)
runner=r/'build_pointnet112.py';runner.write_text("import os,subprocess\nfrom pathlib import Path\nr=Path("+repr(str(r))+")\nenv=dict(os.environ,CUDA_HOME='/usr/local/cuda-11.6',PATH='/usr/local/cuda-11.6/bin:'+os.environ['PATH'],TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',TMPDIR="+repr(str(e/'tmp'))+")\ncode=subprocess.call(["+repr(str(e/'venv/bin/python'))+",'-m','pip','install','--no-deps','--no-build-isolation','--no-cache-dir','--ignore-installed',"+repr(str(b))+"],env=env)\n(r/'pointnet112_build.exit').write_text(str(code)+'\\n')\nraise SystemExit(code)\n")
f=open(str(r/'pointnet112_build.log'),'wb');p=subprocess.Popen([str(e/'venv/bin/python'),str(runner)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':p.pid,'env_spec_sha256':digest,'sources':len(sources),'modified_cuda_sources':0,'runtime_only_build_recipe':str(b/'setup.py')};(r/'pointnet112_build_launch.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
