from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parent
assert Path.cwd() == root / 'OpenPCDet'
extensions = []
for module, name in [
    ('pcdet.ops.pointnet2.pointnet2_stack', 'pointnet2_stack_cuda'),
    ('pcdet.ops.roiaware_pool3d', 'roiaware_pool3d_cuda'),
]:
    source = Path(*module.split('.')) / 'src'
    files = sorted(str(path) for path in source.iterdir() if path.suffix in ('.cpp', '.cu'))
    assert len(files) == (13 if 'pointnet2_stack' in module else 2)
    extensions.append(CUDAExtension(name=module + '.' + name, sources=files))
setup(name='pcdet-pvg-ops', version='0.6.0', ext_modules=extensions, cmdclass={'build_ext': BuildExtension})
