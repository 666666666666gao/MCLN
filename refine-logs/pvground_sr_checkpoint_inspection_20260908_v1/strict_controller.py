import os,subprocess,json
from pathlib import Path
root=Path(__file__).parent
(root/"strict_controller.pid").write_text(str(os.getpid())+"\n")
with (root/"strict_load.log").open("xb") as out,(root/"strict_load.stderr").open("xb") as err:
    result=subprocess.run(['env', 'CUDA_HOME=/usr/local/cuda', 'CUDA_VISIBLE_DEVICES=', 'HF_HUB_OFFLINE=1', 'MAX_JOBS=4', 'MKL_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONPATH=/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground/pointnet2:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/OpenPCDet', 'TOKENIZERS_PARALLELISM=false', 'TORCH_CUDA_ARCH_LIST=8.0', 'TRANSFORMERS_OFFLINE=1', '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-u', '/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1/strict_load.py'],stdout=out,stderr=err)
(root/"strict_load.exit").write_text(str(result.returncode)+"\n")
raise SystemExit(result.returncode)
