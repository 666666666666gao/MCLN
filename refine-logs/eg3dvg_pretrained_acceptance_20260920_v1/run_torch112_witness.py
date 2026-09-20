import json,os,shlex
from pathlib import Path
import paramiko
b=Path(__file__).resolve().parent;r='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.put(str(b/'torch112_witness.py'),r+'/torch112_witness.py')
cmd='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python '+shlex.quote(r+'/torch112_witness.py')
_,o,e=c.exec_command(cmd,timeout=90);out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
rec={'command':cmd,'stdout':out,'stderr':err,'exit':code};(b/'torch112_witness_result.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec,indent=2));s.close();c.close();raise SystemExit(code)
