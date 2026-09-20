"""Freeze a single Nr task-read run, reusing the verified native control budget."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_one(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    root = Path(__file__).resolve().parent
    control = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3')
    assert json.loads((root / 'cpu_check.json').read_text())['status'] == 'pass'
    assert (control / 'controller.exit').read_text().strip() == '0'
    base = json.loads((control / 'spec.json').read_text())
    train = (control / 'train.py').read_text()
    assert sha(control / 'train.py') == base['trainer_sha256']
    train = replace_one(train, '    frozen = {n:',
                         '    from models.eg3dvg_task_read import install_task_read\n    install_task_read(model)\n    frozen = {n:')
    train = replace_one(train, '        grad_norm = torch.nn.utils.clip_grad_norm_',
                         '        task_grad = model.decoder[-1].task_queries.grad\n'
                         '        assert torch.isfinite(task_grad).all()\n'
                         '        task_grad_norms = [float(g.norm()) for g in task_grad]\n'
                         '        assert all(g > 0 for g in task_grad_norms)\n'
                         '        grad_norm = torch.nn.utils.clip_grad_norm_')
    train = replace_one(train, "                  'native_loss': native_value,", "                  'task_gradient_norms': task_grad_norms,\n                  'native_loss': native_value,")
    train = replace_one(train, "    params = dict(model.named_parameters())", "    assert all(torch.count_nonzero(v) > 0 for v in model.decoder[-1].task_queries)\n    params = dict(model.named_parameters())")
    evaluator = (control / 'evaluate_adapted.py').read_text()
    evaluator = replace_one(evaluator, "    checkpoint = torch.load(spec['checkpoint'], map_location='cpu')",
        "    from models.eg3dvg_task_read import install_task_read\n    install_task_read(model)\n    checkpoint = torch.load(spec['checkpoint'], map_location='cpu')")
    for name, text in [('train.py', train), ('evaluate_adapted.py', evaluator)]:
        compile(text, name, 'exec')
        (root / name).write_text(text)
    shutil.copyfile(control / 'audit_adapted.py', root / 'audit_adapted.py')
    spec = dict(base)
    spec['source'] = str(root / 'source')
    spec['source_files'] = {name: sha(root / 'source' / name) for name in base['source_files']}
    spec['source_files']['models/eg3dvg_task_read.py'] = sha(root / 'source/models/eg3dvg_task_read.py')
    spec['trainer_sha256'] = sha(root / 'train.py')
    spec['controller_sha256'] = sha(root / 'controller.py')
    spec['initial_checker_sha256'] = sha(root / 'preflight_eg3dvg_task_read_initial.py')
    spec['post_evaluator_sha256'] = sha(root / 'evaluate_adapted.py')
    spec['post_auditor_sha256'] = sha(root / 'audit_adapted.py')
    spec['control_root'] = str(control)
    spec['wait_for'] = '/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1'
    spec['state_root'] = '/root/mcln_eg3dvg_nr3d_task_read_states_20260920_v1'
    spec['scope'] = 'One last-layer semantic/geometry visual-read mechanism; same author parent and native-control budget'
    spec['new_parameters'] = 165888
    spec['task_read'] = True
    spec.pop('repair', None)
    spec.pop('replaces_failed_preflight', None)
    assert not (root / 'spec.json').exists()
    Path(spec['state_root']).mkdir()
    (root / 'spec.json').write_text(json.dumps(spec, indent=2))
    print(json.dumps({'status': 'prepared', 'spec_sha256': sha(root / 'spec.json'),
                      'training_started': False, 'planned_steps': 5614}), flush=True)


if __name__ == '__main__':
    main()
