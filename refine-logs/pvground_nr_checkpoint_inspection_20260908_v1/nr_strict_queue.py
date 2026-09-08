import datetime,json,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1')
argv=['env', 'CUDA_HOME=/usr/local/cuda', 'CUDA_VISIBLE_DEVICES=', 'HF_HUB_OFFLINE=1', 'MAX_JOBS=4', 'MKL_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONPATH=/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground/pointnet2:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/OpenPCDet', 'TOKENIZERS_PARALLELISM=false', 'TORCH_CUDA_ARCH_LIST=8.0', 'TRANSFORMERS_OFFLINE=1', '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-u', '/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/strict_load.py']
print('NR_STRICT_WAITING_FOR_INVENTORY',flush=True)
while not (root/'inventory.exit').exists():time.sleep(300)
dependency=int((root/'inventory.exit').read_text().strip())
if dependency!=0:
    (root/'strict_dependency_failure.json').write_text(json.dumps({'inventory_exit':dependency})+'\n')
    raise SystemExit(1)
with (root/'strict_load.log').open('w') as out,(root/'strict_load.stderr').open('w') as err:
    result=subprocess.run(argv,stdout=out,stderr=err)
(root/'strict_load.exit').write_text(str(result.returncode)+'\n')
(root/'strict_execution.json').write_text(json.dumps({'argv':argv,'exit':result.returncode,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()},indent=2)+'\n')
print('NR_STRICT_COMPLETE '+str(result.returncode),flush=True)
raise SystemExit(result.returncode)
