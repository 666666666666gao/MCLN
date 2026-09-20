import os,subprocess
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
env=dict(os.environ,CUDA_HOME='/usr/local/cuda-11.6',PATH='/usr/local/cuda-11.6/bin:'+os.environ['PATH'],TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',TMPDIR='/root/mcln_eg3dvg_torch112_20260920_v1/tmp')
code=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-m','pip','install','--no-deps','--no-build-isolation','--no-cache-dir','--ignore-installed','/root/mcln_eg3dvg_torch112_20260920_v1/pointnet2_build'],env=env)
(r/'pointnet112_build.exit').write_text(str(code)+'\n')
raise SystemExit(code)
