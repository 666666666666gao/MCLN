"""Change only SFTP prefetch; keep the original reviewed collector unchanged."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parents[1]
old=root/'postrun/collect_formal_authorized.py'
new=root/'postrun/collect_formal_prefetch_authorized.py'
assert not new.exists()
source=old.read_text(encoding='utf-8')
before="    with sftp.open(remote,'rb') as stream:\n        contents = stream.read()"
after="    with sftp.open(remote,'rb') as stream:\n        stream.prefetch(file_size=sftp.stat(remote).st_size,max_concurrent_requests=16)\n        contents = stream.read()"
assert source.count(before)==1
source=source.replace(before,after)
ast.parse(source)
new.write_text(source,encoding='utf-8')
print(new)
