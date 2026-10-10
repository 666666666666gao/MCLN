"""Share one immutable old prediction cache across five identical closed outputs."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
assert not (root / 'OLD_INITIAL_ARRAY_DEDUPLICATION.json').exists()
inspection = json.loads((root / 'OLD_INITIAL_ARRAY_DUPLICATES.json').read_bytes())
groups = [group for group in inspection['duplicate_groups'] if len(group) == 5]
assert len(groups) == 1
rows = groups[0]
expected = {
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260909_observation_v1/initial/boxes.npy',
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_fixed_memory_v1/initial/boxes.npy',
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_rec_competition_v1/initial/boxes.npy',
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1/initial/boxes.npy',
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1/initial/boxes.npy'}
assert {row['path'] for row in rows} == expected
assert all(row['sha256'] == 'c5b7bc1da6c71cf3c7631e3f1c95060ccc9490e654cc0a300b60a9f0e63a0a36'
           and row['bytes'] == 42313856 for row in rows)
assert len({row['inode'] for row in rows}) == 5
code = r'''import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
b=json.load(sys.stdin);rows=b['files'];paths=[Path(row['path']) for row in rows]
assert len(paths)==5 and len(set(paths))==5
assert all(p.resolve()==p and not p.is_symlink() and p.parts[:3]==('/','root','autodl-tmp')
 and p.parts[-2:]==('initial','boxes.npy') for p in paths)
prefixes=[str(p.parent.parent).encode() for p in paths]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or not (proc/'cmdline').exists():continue
 argv=(proc/'cmdline').read_bytes()
 assert not any(prefix in argv for prefix in prefixes)
for p,row in zip(paths,rows):
 s=p.stat(); assert s.st_size==row['bytes'] and s.st_ino==row['inode'] and s.st_dev==row['device']
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==row['sha256'] and (p.parent.parent/'terminal'/'boxes.npy').is_file()
before=shutil.disk_usage('/root/autodl-tmp').free
canonical=paths[0];operations=[]
for p in paths[1:]:
 temporary=p.with_name('boxes.npy.dedup_20261010');assert not temporary.exists()
 assert p.stat().st_dev==canonical.stat().st_dev
 os.link(str(canonical),str(temporary))
 assert temporary.stat().st_ino==canonical.stat().st_ino
 os.replace(str(temporary),str(p))
 operations.append(dict(path=str(p),canonical=str(canonical),bytes_preserved=p.stat().st_size))
assert all(p.stat().st_ino==canonical.stat().st_ino and p.stat().st_size==42313856 for p in paths)
result=dict(status='EXACT_FIVE_CLOSED_IDENTICAL_INITIAL_ARRAY_PATHS_SHARE_ONE_STORAGE_COPY',
 time_cst=datetime.datetime.now().astimezone().isoformat(),sha256=rows[0]['sha256'],files=rows,
 operations=operations,canonical=str(canonical),same_device=canonical.stat().st_dev,
 canonical_inode=canonical.stat().st_ino,links=canonical.stat().st_nlink,
 duplicate_physical_copies_removed=4,duplicate_file_bytes=169255424,
 data_free_before=before,data_free_after=shutil.disk_usage('/root/autodl-tmp').free,
 all_five_paths_and_bytes_preserved=True,weights_data_runtime_V99_untouched=True,
 new_neural_calls=0)
receipt=Path('/root/autodl-tmp/pvground_normal_degradation_panel_20261010/initial_cache_deduplication.json')
assert not receipt.exists();receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
witness = json.loads((root.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
variables = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                 SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
        '-o', 'StrictHostKeyChecking=yes', '-o',
        'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
        '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
        'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
response = subprocess.run(argv, env=variables, input=json.dumps(dict(files=rows)).encode(), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'INITIAL_DEDUPLICATION_STDOUT.json').write_bytes(response.stdout)
(root / 'INITIAL_DEDUPLICATION_STDERR.txt').write_bytes(response.stderr)
(root / 'INITIAL_DEDUPLICATION_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n')
assert response.returncode == 0, 'Inspect exact paths/inodes and original receipt before any repair; no blind rerun'
record = json.loads(response.stdout)
(root / 'OLD_INITIAL_ARRAY_DEDUPLICATION.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key:record[key] for key in ('status','time_cst','duplicate_file_bytes','data_free_after',
                                            'all_five_paths_and_bytes_preserved')}),flush=True)
