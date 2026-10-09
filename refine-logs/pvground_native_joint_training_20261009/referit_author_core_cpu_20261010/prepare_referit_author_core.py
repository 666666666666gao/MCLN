"""Preserve R2; correct the witnessed author position-id loading order only."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
old = root / 'referit_native_preparation_20261010'
new = root / 'referit_author_core_20261010'
assert not new.exists()
new.mkdir()
original = {}
for leaf in ('source', 'nr3d', 'sr3d'):
    for path in (old / leaf).rglob('*'):
        if path.is_file():
            name = path.relative_to(old).as_posix()
            raw = path.read_bytes()
            original[name] = hashlib.sha256(raw).hexdigest()
            target = new / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
path = new / 'source/native_model_initialization.py'
source = path.read_text(encoding='utf-8')
before = """    model.text_encoder.embeddings.register_buffer('position_ids', position_ids, persistent=False)
    assert len(model.state_dict()) == 1234
    if initialize_weights:
        official = checked_payload(spec['official_checkpoint'], spec['official_checkpoint_sha256'])
        initial = {name[7:]: value for name, value in official['model'].items()}
        assert all(name.startswith('module.') for name in official['model'])
        model.load_state_dict(initial, strict=True)
"""
after = """    assert len(model.state_dict()) == 1235
    if initialize_weights:
        official = checked_payload(spec['official_checkpoint'], spec['official_checkpoint_sha256'])
        initial = {name[7:]: value for name, value in official['model'].items()}
        assert all(name.startswith('module.') for name in official['model'])
        assert len(initial) == 1235
        assert torch.equal(initial['text_encoder.embeddings.position_ids'], position_ids)
        model.load_state_dict(initial, strict=True)
    # Both actual Nr/Sr author states include this deterministic buffer.
    # Load it strictly first; retained full-method states use the existing
    # nonpersistent convention only after the author core has been loaded.
    assert torch.equal(model.text_encoder.embeddings.position_ids, position_ids)
    model.text_encoder.embeddings.register_buffer('position_ids', position_ids, persistent=False)
    assert len(model.state_dict()) == 1234
"""
assert source.count(before) == 1
source = source.replace(before, after)
ast.parse(source)
path.write_text(source, encoding='utf-8')
for name, digest in original.items():
    assert hashlib.sha256((old / name).read_bytes()).hexdigest() == digest
changed = [name for name,digest in original.items()
    if hashlib.sha256((new / name).read_bytes()).hexdigest() != digest]
assert changed == ['source/native_model_initialization.py']
for path in (new / 'source').rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
value = dict(status='AUTHOR_LOAD_ORDER_R3_SOURCE_ONLY', changed=changed,
    R2_sources_and_specs_sha256=original,
    current_sources_sha256={name:hashlib.sha256((new/name).read_bytes()).hexdigest() for name in original},
    evidence=str(root/'AUTHOR_CORE_INVENTORY_LOCAL_COMPARISON.json'),
    active_scanrefer_source_changed=False, author_model_constructed=False,
    remote_training_queries=0, GPU_calls=0, CPU_check_pending=True, full_goal_complete=False)
(new/'SOURCE_PREPARATION.json').write_text(json.dumps(value, indent=2)+'\n')
print(json.dumps({k:value[k] for k in ('status','changed','active_scanrefer_source_changed','CPU_check_pending')}))
