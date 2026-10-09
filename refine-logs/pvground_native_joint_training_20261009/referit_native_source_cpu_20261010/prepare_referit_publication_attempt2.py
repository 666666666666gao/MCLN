"""Create one distinct static copy after the exact old state was witnessed."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
receipt = json.loads((root / 'REFERIT_NATIVE_CPU_STATIC_RECEIPT.json').read_bytes())
assert receipt['prior_static_state_confirmed'] is True and receipt['evidence_leaf_exists'] is False
target = root / 'publish_referit_native_cpu_attempt2_authorized.py'
assert not target.exists()
source = (root / 'publish_referit_native_cpu_authorized.py').read_text(encoding='utf-8')
source = source.replace('REFERIT_NATIVE_CPU_REMOTE', 'REFERIT_NATIVE_CPU_ATTEMPT2_REMOTE')
anchor = "root = Path(__file__).resolve().parent\n"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "static = json.loads((root / 'REFERIT_NATIVE_CPU_STATIC_RECEIPT.json').read_bytes())\nassert static['prior_static_state_confirmed'] is True and static['evidence_leaf_exists'] is False\n")
anchor = "assert prior['section'] == '20.376.131' and prior['remote_handoff_sync_complete'] is True\n"
source = source.replace(anchor, anchor + "assert static['doc_sha256'] == prior['doc_sha256']\n")
anchor = "    'prepare_referit_cpu_publication.py', 'publish_referit_native_cpu_authorized.py']"
assert anchor in source
source = source.replace(anchor, """    'prepare_referit_cpu_publication.py', 'publish_referit_native_cpu_authorized.py',
    'prepare_referit_publication_attempt2.py', 'publish_referit_native_cpu_attempt2_authorized.py',
    'read_referit_publication_static_authorized.py',
    'REFERIT_NATIVE_CPU_REMOTE_STDOUT.json', 'REFERIT_NATIVE_CPU_REMOTE_STDERR.txt',
    'REFERIT_NATIVE_CPU_REMOTE_EXIT.json', 'REFERIT_NATIVE_CPU_STATIC_RECEIPT.json',
    'REFERIT_NATIVE_CPU_STATIC_STDOUT.json', 'REFERIT_NATIVE_CPU_STATIC_STDERR.txt',
    'REFERIT_NATIVE_CPU_STATIC_EXIT.json']""")
ast.parse(source)
target.write_text(source, encoding='utf-8')
note = root / 'HANDOFF_REFERIT_NATIVE_SOURCE_CPU_20261010.md'
with note.open('a', encoding='utf-8') as stream:
    stream.write('\n本节首次静态同步出现SSH认证失败，保留退出255和空stdout。经既有授权连接作一次只读确认：远端仍为§131/e2b5f2cb，新增证据目录不存在，未读取训练状态。后续使用单独的第二次静态发布，原失败回执保留；没有重跑CPU实验或训练。\n')
print('Prepared distinct static attempt2; no remote call.')
