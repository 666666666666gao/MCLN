import json,os,shlex,sys
from pathlib import Path
import paramiko

base=Path(__file__).resolve().parent
script=Path(sys.argv[1]);root='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.put(str(script),root+'/'+script.name)
cmd='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python '+shlex.quote(root+'/'+script.name)
_,o,e=c.exec_command(cmd,timeout=240)
out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
record={'command':cmd,'stdout':out,'stderr':err,'exit':code}
(base/(script.stem+'_result.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
s.close();c.close();print(json.dumps(record,indent=2));raise SystemExit(code)
