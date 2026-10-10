"""One read-only CSV/split/input-file check; no loader, model, or GPU calls."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
assert not (root / 'METADATA_RESULT.json').exists()
review = json.loads((root / 'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and review['blocking_issue_count'] == 0
for name, digest in review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')

code = r'''import ast,csv,datetime,hashlib,json,os
from pathlib import Path
source=Path('/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground')
data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
file=source/'src/joint_det_dataset.py'
raw=file.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d'
tree=ast.parse(raw.decode())
resolver=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='resolve_referit3d_csv']
assert len(resolver)==1
namespace={'os':os}
exec(compile(ast.Module(body=resolver),str(file),'exec'),namespace)
records=[]
for dataset in ('nr3d','sr3d'):
 path=Path(namespace['resolve_referit3d_csv'](str(data),dataset+'.csv'))
 csv_raw=path.read_bytes()
 with path.open() as handle: rows=list(csv.DictReader(handle))
 for split,meta_split in (('train','train'),('val','test')):
  split_path=source/'data/meta_data'/(dataset+'_'+meta_split+'_scans.txt')
  split_raw=split_path.read_bytes();scan_ids=set(ast.literal_eval(split_raw.decode()))
  in_scenes=[row for row in rows if row['scan_id'] in scan_ids]
  selected=[row for row in in_scenes if (dataset=='nr3d' and (split=='train' or str(row['correct_guess']).lower()=='true')) or (dataset=='sr3d' and str(row['mentions_target_class']).lower()=='true')]
  assert all(int(row['target_id'])>=0 for row in selected)
  used=sorted({row['scan_id'] for row in selected})
  missing_superpoints=[str(data/'superpoints'/split/(scene+'_superpoint.pth')) for scene in used if not (data/'superpoints'/split/(scene+'_superpoint.pth')).is_file()]
  missing_proposals=[str(data/'group_free_pred_bboxes'/('group_free_pred_bboxes_'+split)/(scene+'.npy')) for scene in used if not (data/'group_free_pred_bboxes'/('group_free_pred_bboxes_'+split)/(scene+'.npy')).is_file()]
  records.append(dict(dataset=dataset,split=split,metadata_split=meta_split,csv_path=str(path),
   csv_bytes=len(csv_raw),csv_sha256=hashlib.sha256(csv_raw).hexdigest(),csv_total_rows=len(rows),
   split_path=str(split_path),split_sha256=hashlib.sha256(split_raw).hexdigest(),split_scenes=len(scan_ids),
   input_rows_in_scenes=len(in_scenes),metadata_eligible_expressions=len(selected),used_scenes=len(used),
   filtering='correct_guess=true only for Nr3D val; mentions_target_class=true for both Sr3D splits',
   missing_superpoint_paths=missing_superpoints,missing_groupfree_proposal_paths=missing_proposals,
   native_scans_pickle_path=str(data/(split+'_v3scans.pkl')),
   native_scans_pickle_bytes=(data/(split+'_v3scans.pkl')).stat().st_size,
   actual_native_loader_length=None))
 train_ids=set(ast.literal_eval((source/'data/meta_data'/(dataset+'_train_scans.txt')).read_text()))
 test_ids=set(ast.literal_eval((source/'data/meta_data'/(dataset+'_test_scans.txt')).read_text()))
 assert not train_ids.intersection(test_ids)
print(json.dumps(dict(status='REFERIT_REAL_METADATA_CHECKED_LOADER_NOT_CONSTRUCTED',
 observed_cst=datetime.datetime.now().astimezone().isoformat(),source=str(source),data=str(data),records=records,
 dataset_source_sha256=hashlib.sha256(raw).hexdigest(),cuda_imported=False,model_calls=0,
 current_normal_training_queries=0,remote_files_written=0,optimizer_updates=0,
 object_protocol_checked='predicted GroupFree input files only; author butd_cls protocol is different',
 architecture_finalized=False,Nr3D_or_Sr3D_training_launched=False,REC_accuracy=None)))
'''
witness = json.loads((root.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
started = datetime.datetime.now().astimezone().isoformat()
response = subprocess.run(argv, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'RAW_STDOUT.json').write_bytes(response.stdout)
(root / 'RAW_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root / 'TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode,
    started_cst=started, finished_cst=datetime.datetime.now().astimezone().isoformat())) + '\n')
assert response.returncode == 0, 'Metadata transport failed; no unchanged automatic retry'
value = json.loads(response.stdout)
assert value['current_normal_training_queries'] == 0 and value['remote_files_written'] == 0
(root / 'METADATA_RESULT.json').write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps(dict(status=value['status'], records=[{key:row[key] for key in
    ('dataset','split','metadata_eligible_expressions','used_scenes','missing_superpoint_paths',
     'missing_groupfree_proposal_paths')} for row in value['records']],
    current_training_queries=0, REC_accuracy=None)))
