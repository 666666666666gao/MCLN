import ast
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_stage_trace_preparation_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_trace_preparation_20260907_v1'
source = '/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source'
names = ['models/rec_geometry_reranker.py', 'models/rec_pareto_contextual_hierarchy.py',
         'models/rec_candidate_adapter.py', 'models/rec_hierarchical_reranker.py',
         'models/rec_selective_residual.py', 'models/rec_reranker.py', 'models/mask_fusion.py',
         'scripts/scanrefer_joint_readout.py', 'train_dist_mod.py']
paths = {name: source + '/' + name for name in names}
paths['scripts/scanrefer_joint_readout.py'] = (
    '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
    '/scripts/scanrefer_joint_readout.py')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
bound = {}
for name in names:
    with sftp.open(paths[name], 'rb') as stream:
        raw = stream.read()
    canonical = (repo / name).read_bytes()
    if name == 'train_dist_mod.py':
        functions = ['_build_rec_reranker_outputs_float32',
                     '_build_rec_geometry_runtime_outputs_float32',
                     '_build_rec_hierarchical_runtime_batch']
        actual_tree = {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
        canonical_tree = {node.name: node for node in ast.parse(canonical).body if isinstance(node, ast.FunctionDef)}
        for function in functions:
            assert ast.dump(actual_tree[function]) == ast.dump(canonical_tree[function]), function
        differences = '\n'.join(difflib.unified_diff(
            raw.decode().splitlines(), canonical.decode().splitlines(),
            fromfile='frozen_runtime', tofile='canonical_worktree')) + '\n'
        (local / 'runtime_diff.txt').write_text(differences, encoding='utf-8', newline='\n')
        with sftp.open(remote + '/runtime_diff.txt', 'wx') as stream:
            stream.write(differences.encode())
    else:
        assert raw == canonical, name
    bound[name] = {'path': paths[name], 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                   'canonical_bytes_equal': raw == canonical}
receipt = {'schema': 'mcln-stage-trace-source-binding-v1',
           'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'existing_source': source, 'files': bound,
           'readout_runtime_functions_ast_equal': functions,
           'runtime_diff': 'Canonical factory adds CandidateLocalVisual for later native preflight; frozen readout functions are unchanged.',
           'active_sources_modified': False,
           'training_polled': False, 'gpu_forwards': 0,
           'cpu_test_receipt_sha256': hashlib.sha256((local / 'receipt.json').read_bytes()).hexdigest()}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
(local / 'dependency_binding.json').write_bytes(raw)
with sftp.open(remote + '/dependency_binding.json', 'wx') as stream:
    stream.write(raw)
archive = Path(__file__).read_bytes()
(local / 'bind_dependencies.py').write_bytes(archive)
with sftp.open(remote + '/bind_dependencies.py', 'wx') as stream:
    stream.write(archive)
sftp.close()
client.close()
print(json.dumps({'files': len(bound), 'receipt_sha256': hashlib.sha256(raw).hexdigest(),
                  'cpu_test_receipt_sha256': receipt['cpu_test_receipt_sha256']}))
