"""Wait for Sr baseline, then run one fixed Nr task-read trial and formal REC."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    spec = json.loads((root / 'spec.json').read_text())
    assert sha(__file__) == spec['controller_sha256']
    wait = Path(spec['wait_for'])
    while not (wait / 'controller.exit').exists():
        time.sleep(300)
    assert (wait / 'controller.exit').read_text().strip() == '0'
    assert json.loads((wait / 'formal/audit.json').read_text())['integrity_pass']
    lock = open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
    assert shutil.disk_usage(spec['state_root']).free > 2_500_000_000

    def run(name, command, cwd):
        with (root / (name + '.log')).open('xb') as log:
            code = subprocess.call(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        (root / (name + '.exit')).write_text(str(code) + '\n')
        if code:
            (root / 'controller.exit').write_text(str(code) + '\n')
            raise SystemExit(code)

    assert sha(root / 'preflight_eg3dvg_task_read_initial.py') == spec['initial_checker_sha256']
    run('initial_equality', [sys.executable, '-u', str(root / 'preflight_eg3dvg_task_read_initial.py'),
                            '--spec', str(root / 'spec.json')], spec['source'])
    assert json.loads((root / 'initial_equality.json').read_text())['status'] == 'pass'
    for stage in ['preflight', 'fit']:
        run(stage, [sys.executable, '-u', str(root / 'train.py'), '--spec', str(root / 'spec.json'),
                    '--stage', stage], spec['source'])
    fit = json.loads((root / 'fit/receipt.json').read_text())
    assert fit['optimizer_steps'] == 5614 and fit['rows'] == 44909
    post = root / 'evaluation'
    post.mkdir()
    old = Path(spec['transfer_root'])
    eval_spec = json.loads((old / 'spec.json').read_text())
    eval_spec.update(checkpoint=fit['checkpoint_path'], checkpoint_sha256=fit['checkpoint_sha256'],
                     checkpoint_optimizer_steps=5614, experiment_type='single_epoch_task_read',
                     training_spec_sha256=fit['spec_sha256'], source=spec['source'],
                     source_files=spec['source_files'],
                     evaluator_sha256=sha(root / 'evaluate_adapted.py'),
                     auditor_sha256=sha(root / 'audit_adapted.py'))
    (post / 'spec.json').write_text(json.dumps(eval_spec, indent=2))
    shutil.copyfile(old / 'annotation_manifest.json', post / 'annotation_manifest.json')
    for stage in ['preflight', 'formal']:
        run('evaluation_' + stage, [sys.executable, '-u', str(root / 'evaluate_adapted.py'),
                                   '--spec', str(post / 'spec.json'), '--stage', stage], spec['source'])
    run('evaluation_audit', [sys.executable, '-u', str(root / 'audit_adapted.py'),
                             '--root', str(post)], spec['source'])
    (root / 'controller.exit').write_text('0\n')


if __name__ == '__main__':
    main()
