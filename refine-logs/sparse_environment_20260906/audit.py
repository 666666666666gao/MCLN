import importlib.util,json,platform,shutil,subprocess,sys
from pathlib import Path
import pkg_resources
installed={d.key:d.version for d in pkg_resources.working_set}
names=['torch','torchvision','numpy','spconv','spconv-cu111','spconv-cu113','spconv-cu114','spconv-cu117','spconv-cu120','spconv-cu124','cumm','minkowskiengine','torchsparse','torch-scatter','transformers']
modules={name:importlib.util.find_spec(name) is not None for name in ['spconv','pcdet','MinkowskiEngine','torchsparse','torch_scatter']}
nvcc=shutil.which('nvcc')
driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
version=subprocess.run([nvcc,'--version'],capture_output=True,text=True,check=True).stdout.strip() if nvcc else None
print(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
 'installed_selected_packages':{name:installed.get(name) for name in names},'module_specs_available':modules,
 'cuda_compiler_path':nvcc,'cuda_compiler_version':version,'gpu_driver_and_model':driver,
 'conda_environment_names':sorted(p.name for p in Path('/root/miniconda3/envs').iterdir() if p.is_dir()),
 'package_installs':0,'native_model_forwards':0,'gpu_forwards':0,'optimizer_steps':0}))
