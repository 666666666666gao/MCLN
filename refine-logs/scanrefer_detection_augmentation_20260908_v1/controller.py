import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
runtime=Path("/root/autodl-tmp/mcln_pvground_runtime_20260908_v1")
env=os.environ.copy()
env.update(json.loads((runtime/"env_spec.json").read_bytes())["env"])
env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
(root/"controller.pid").write_text(str(os.getpid())+"\n")
with (root/"run.log").open("xb") as log:
    r=subprocess.run([str(runtime/"venv/bin/python"),"-u",str(root/"audit.py"),"--spec",str(root/"spec.json")],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/"controller.exit").write_text(str(r.returncode)+"\n")
raise SystemExit(r.returncode)
