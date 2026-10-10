"""Reuse the existing read-only startup and single scheduled observer."""
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.parent
old_remote = '/root/autodl-tmp/pvground_native_joint_training_20261010/normal'
new_remote = '/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off'
for name in ('witness_normal_start_authorized.py', 'observe_normal_planned_authorized.py'):
    path = root / name
    assert not path.exists()
    content = (previous / name).read_text(encoding='utf-8')
    assert old_remote in content
    content = content.replace(old_remote, new_remote)
    content = content.replace('project = root.parent\n', 'project = root.parent.parent\n')
    content = content.replace('project=root.parent\n', 'project=root.parent.parent\n')
    content = content.replace(
        'Two-batch engineering steady update about3.5s times4583, plus initialization and formal evaluation; not a measured full epoch',
        'Prior ordinary three-epoch run took14.63349hours including initial validation; first node fivehours after launch, then use measured progress')
    content = content.replace("assert response.returncode == 0, response.stderr.decode()",
                              "assert response.returncode == 0, 'Inspect private preserved startup stderr; do not restart training'")
    content = content.replace("assert response.returncode==0,response.stderr.decode()",
                              "assert response.returncode==0,'Inspect private preserved observer stderr; do not restart training'")
    compile(content, str(path), 'exec')
    path.write_text(content, encoding='utf-8')
print('C_OFF_READ_ONLY_NORMAL_OBSERVERS_PREPARED')
