"""Take a distinct read-only checkpoint snapshot without overwriting published evidence."""
from pathlib import Path

source=Path(__file__).with_name('observe_pair_once.py')
code=source.read_text(encoding='utf-8')
assert code.count("'pair_live_observation.json'")==1
code=code.replace("'pair_live_observation.json'","'pair_checkpoint_observation.json'")
assert code.count('records={}\n')==1
code=code.replace('records={}\n',"records={}\ncheckpoints=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in directory.glob('*.pth')]\n")
assert code.count('stage_log_tail=raw.decode')==1
code=code.replace('stage_log_tail=raw.decode','checkpoints=checkpoints,stage_log_tail=raw.decode')
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
