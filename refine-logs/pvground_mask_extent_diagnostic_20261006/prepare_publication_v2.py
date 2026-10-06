"""Minimal receiver I/O correction after the observed empty stdout failure."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'publish_formal_launch.py').read_text(encoding='utf-8')
changes=[
    ("publication_review=json.loads((local/'PUBLISH_LAUNCH_REVIEW.json').read_bytes())",
     "publication_review=json.loads((local/'PUBLISH_LAUNCH_V2_REVIEW.json').read_bytes())"),
    ("stdin.write(json.dumps(bundle).encode());stdin.channel.shutdown_write()",
     "stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()"),
    ("remote=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()",
     "remote_raw=stdout.read();remote_exit=stdout.channel.recv_exit_status();remote_error=stderr.read().decode()\nassert remote_exit==0,remote_error\nremote=json.loads(remote_raw)"),
    ("'PUBLISH_LAUNCH_REVIEW.json','PUBLISH_LAUNCH_REVIEW.md',",
     "'PUBLISH_LAUNCH_REVIEW.json','PUBLISH_LAUNCH_REVIEW.md',\n          'PUBLISH_LAUNCH_V2_REVIEW.json','PUBLISH_LAUNCH_V2_REVIEW.md','PUBLICATION_FAILURE_INSPECTION.json',")]
for before,after in changes:
    assert source.count(before)==1,before
    source=source.replace(before,after)
ast.parse(source)
target=root/'publish_formal_launch_v2.py';assert not target.exists()
target.write_text(source,encoding='utf-8')
print('MINIMAL_STDIN_FLUSH_AND_EXIT_BEFORE_JSON_V2_PREPARED')
