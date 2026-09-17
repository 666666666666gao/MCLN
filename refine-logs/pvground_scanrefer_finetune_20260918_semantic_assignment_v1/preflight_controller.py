import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/"preflight.pid").write_text(str(os.getpid())+"\n")
spec=json.loads((root/"spec.json").read_bytes())
for name,digest in spec["files"].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
with (root/"preflight.log").open("xb") as log:
    result=subprocess.run(['flock', '-n', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-u', '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1/train.py', '--spec', '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1/spec.json', '--preflight-only'],env=dict(os.environ,**{'CUDA_HOME': '/usr/local/cuda', 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'MAX_JOBS': '4', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground/pointnet2:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/OpenPCDet', 'TOKENIZERS_PARALLELISM': 'false', 'TORCH_CUDA_ARCH_LIST': '8.0', 'TRANSFORMERS_OFFLINE': '1'}),stdout=log,stderr=subprocess.STDOUT)
(root/"preflight.exit").write_text(str(result.returncode)+"\n")
raise SystemExit(result.returncode)
