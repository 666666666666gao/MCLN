from pathlib import Path

root=Path(__file__).resolve().parent
source=(root.parent/'pvground_final_quality_20261005/collect_formal_authorized.py').read_text(encoding='utf-8')
assert source.count("local/'quality_fit_spec.json'")==1
source=source.replace("local/'quality_fit_spec.json'","local/'control_spec.json'")
source=source.replace("('.json','.jsonl','.log','.exit')","('.json','.jsonl','.log','.exit','.py','.md')")
path=root/'collect_formal_authorized.py'
assert not path.exists()
path.write_text(source,encoding='utf-8')
print(str(path))
