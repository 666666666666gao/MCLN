"""Create a minimal isolated derivative of the completed responsibility runner."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.parent / 'pvground_query_supported_geometry_20261005'
assert not (root / 'run_geometry_fit.py').exists()


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


helper = (previous / 'query_supported_geometry.py').read_text(encoding='utf-8')
helper = change(helper, 'def query_supported_geometry_loss(predictions,batch,indices,set_criterion):',
                'def query_supported_geometry_loss(predictions,batch,indices,set_criterion,auxiliary_roots):')
helper = change(helper,
    "root=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])",
    'root=auxiliary_roots[bid]')
helper = change(helper,
    "{name:batch[name][bid:bid+1] for name in ('box_label_mask','center_label','size_gts')},single_indices)",
    "dict(box_label_mask=torch.ones((1,1),device=root.device),\n"
    "                 center_label=root[:3].view(1,1,3),size_gts=root[3:].view(1,1,3)),single_indices)")
(root / 'query_supported_geometry.py').write_text(helper, encoding='utf-8')
runner = (previous / 'run_geometry_fit.py').read_text(encoding='utf-8')
runner = change(runner, 'Protected4506 head continuation with Query-supported geometry targets.',
                'Protected4509 continuation; native versus member auxiliary geometry targets.')
runner = change(runner, "assert spec['geometry_hits50']==4506 and spec['head_only']",
                "assert spec['geometry_hits50']==4509 and spec['head_only']")
runner = change(runner, "assert spec['use_geometry_evidence'] is True and spec['extra_geometry_weight'] in (0.0,1.0)",
                "assert spec['use_geometry_evidence'] is True and spec['extra_geometry_weight']==1.0\n"
                "    assert spec['auxiliary_target_mode'] in ('native_gt','member_gt')")
runner = change(runner, '        def __getitem__(self, index):\n            result = super().__getitem__(index)\n            result[\'local_training_id\']',
    "        def _get_target_boxes(self, anno, scan):\n"
    "            target = anno['target_id'][0] if isinstance(anno['target_id'], list) else anno['target_id']\n"
    "            corners = scan.get_object_bbox(target).reshape(6).astype(np.float64)\n"
    "            self.pre_jitter_root = np.concatenate(((corners[:3]+corners[3:])*.5,\n"
    "                corners[3:]-corners[:3])).astype(np.float32)\n"
    "            return super()._get_target_boxes(anno, scan)\n\n"
    "        def __getitem__(self, index):\n            result = super().__getitem__(index)\n"
    "            result['pre_jitter_root_box'] = self.pre_jitter_root.copy()\n            result['local_training_id']")
runner = change(runner, "assert terminal['head_only'] and terminal['extra_geometry_weight']==spec['extra_geometry_weight']",
    "assert terminal['head_only'] and terminal['extra_geometry_weight']==spec['extra_geometry_weight']\n"
    "        assert terminal['auxiliary_target_mode']==spec['auxiliary_target_mode']")
runner = change(runner,
    '        extra,extra_counts,qualified=query_supported_geometry_loss(predictions,batch,matches[1],set_criterion)',
    "        native_roots=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1)\n"
    "        auxiliary_roots=native_roots if spec['auxiliary_target_mode']=='native_gt' else batch['pre_jitter_root_box']\n"
    "        extra,extra_counts,qualified=query_supported_geometry_loss(predictions,batch,matches[1],set_criterion,auxiliary_roots)")
runner = change(runner, "            extra_geometry_weight=spec['extra_geometry_weight'],extra_counts=extra_counts,",
    "            extra_geometry_weight=spec['extra_geometry_weight'],extra_counts=extra_counts,\n"
    "            auxiliary_target_mode=spec['auxiliary_target_mode'],\n"
    "            auxiliary_native_max_face_shift=float((torch.cat([native_roots[:,:3]-native_roots[:,3:]/2,native_roots[:,:3]+native_roots[:,3:]/2],-1)\n"
    "                -torch.cat([batch['pre_jitter_root_box'][:,:3]-batch['pre_jitter_root_box'][:,3:]/2,batch['pre_jitter_root_box'][:,:3]+batch['pre_jitter_root_box'][:,3:]/2],-1)).abs().max()),")
runner = change(runner, 'geometry_parent_fit_updates=3723,total_geometry_fit_updates=3723+step_number,',
    "geometry_parent_fit_updates=7446,total_geometry_fit_updates=7446+step_number,\n"
    "            auxiliary_target_mode=spec['auxiliary_target_mode'],")
runner = change(runner, "        assert batch_index==spec['preflight_batch_index']\n        witnesses=[step(batch_cpu,True) for _ in range(2)]",
    "        assert batch_index==spec['preflight_batch_index']\n"
    "        reference_rows=[json.loads(line) for line in Path(spec['preflight_reference_rows']).read_text().splitlines()]\n"
    "        for bid,row_id in enumerate(batch_cpu['local_training_id'].tolist()):\n"
    "            reference=reference_rows[batch_index*8+bid]\n"
    "            assert row_id==reference['row_id']\n"
    "            assert hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest()==reference['point_sha256']\n"
    "            native_root=torch.cat([batch_cpu['center_label'][bid,0,:3],batch_cpu['size_gts'][bid,0]])\n"
    "            assert native_root.tolist()==reference['noisy_root_box']\n"
    "            assert batch_cpu['pre_jitter_root_box'][bid].tolist()==reference['pre_jitter_root_box']\n"
    "        witnesses=[step(batch_cpu,True) for _ in range(2)]")
runner = change(runner, 'receipt=dict(status=\'pass\',optimizer_steps=2,weight_files_created=0,accuracy_result=False,',
                "receipt=dict(status='pass',optimizer_steps=2,weight_files_created=0,accuracy_result=False,native_data_and_member_target_exact=True,")
(root / 'run_geometry_fit.py').write_text(runner, encoding='utf-8')
controller = (previous / 'controller.py').read_text(encoding='utf-8')
controller = controller.replace("('control','query_supported')", "('control','member_target')")
controller = change(controller, 'protected_best_hits=[5616,4506]', 'protected_best_hits=[5614,4509]')
(root / 'controller.py').write_text(controller, encoding='utf-8')
parent = json.loads((previous / 'query_supported_spec.json').read_bytes())
for arm, mode in (('control', 'native_gt'), ('member_target', 'member_gt')):
    spec = dict(parent)
    spec.update(root='/root/autodl-tmp/pvground_auxiliary_target_20261005/' + arm,
        geometry_terminal='/root/autodl-tmp/pvground_query_supported_geometry_20261005/query_supported/terminal.pth',
        geometry_terminal_sha256='0b37986d4448b125405d573272282ef37396a781ba970789dfc9b080445cec51',
        geometry_hits50=4509, auxiliary_target_mode=mode, extra_geometry_weight=1.0,
        geometry_parent_fit_updates=7446, total_geometry_fit_updates=11169,
        preflight_reference_rows='/root/autodl-tmp/pvground_target_jitter_20261005/rows.jsonl')
    (root / (arm + '_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
for file in root.glob('*.py'):
    ast.parse(file.read_text(encoding='utf-8'))
changes = dict(status='PREPARED_NOT_LAUNCHED', new_model_parameters=0,
    parent_best_hits=[5614,4509], arms=['control/native_gt','member_target/member_gt'],
    deleted_heads_referenced=False, native_protocol_changed=False,
    files={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in root.glob('*.py')})
(root / 'PREPARATION.json').write_text(json.dumps(changes, indent=2) + '\n', encoding='utf-8')
print(json.dumps(changes))
