import json,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v2')
prefix=['env', 'CUDA_HOME=/usr/local/cuda', 'CUDA_VISIBLE_DEVICES=', 'HF_HUB_OFFLINE=1', 'MAX_JOBS=4', 'MKL_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONPATH=/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground/pointnet2:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/OpenPCDet', 'TOKENIZERS_PARALLELISM=false', 'TORCH_CUDA_ARCH_LIST=8.0', 'TRANSFORMERS_OFFLINE=1', '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-u']
for stage in ['inventory','strict_load']:
    argv=prefix+[str(root/(stage+'.py'))]
    with (root/(stage+'.log')).open('w') as out,(root/(stage+'.stderr')).open('w') as err:
        result=subprocess.run(argv,stdout=out,stderr=err)
    (root/(stage+'.exit')).write_text(str(result.returncode)+'\n')
    (root/(stage+'_execution.json')).write_text(json.dumps({'argv':argv,'exit':result.returncode},indent=2)+'\n')
    print(stage+' '+str(result.returncode),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
