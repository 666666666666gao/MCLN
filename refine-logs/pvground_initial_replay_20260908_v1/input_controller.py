import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
runtime=Path("/root/autodl-tmp/mcln_pvground_runtime_20260908_v1")
env=os.environ.copy();env.update(json.loads((runtime/"env_spec.json").read_bytes())["env"])
env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
(root/"input_controller.pid").write_text(str(os.getpid())+"\n")
with (root/"input.log").open("xb") as log:
    result=subprocess.run([str(runtime/"venv/bin/python"),"-u",str(root/"export_inputs.py"),"--training-root","/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_detalign_v1","--output",str(root/"inputs")],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/"input_controller.exit").write_text(str(result.returncode)+"\n")
raise SystemExit(result.returncode)
