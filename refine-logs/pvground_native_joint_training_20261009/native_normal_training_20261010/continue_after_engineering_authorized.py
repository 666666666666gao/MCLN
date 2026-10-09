"""Join the one original observer, then await its fresh actual audit locally."""
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

root=Path(__file__).resolve().parent
assert not (root/'NORMAL_QUEUE.json').exists() and not (root/'NORMAL_LAUNCH.json').exists()
review=json.loads((root/'SOURCE_NORMAL_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and not review['blocking_findings']
assert review['audited_input_hashes'][str(Path(__file__).resolve())]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
owner=json.loads((root/'ENGINEERING_OBSERVER_OWNER.json').read_bytes())
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.GetProcessTimes.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME))
kernel.GetProcessTimes.restype=wintypes.BOOL
kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
kernel.WaitForSingleObject.restype=wintypes.DWORD
kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
kernel.CloseHandle.restype=wintypes.BOOL
handle=kernel.OpenProcess(0x100000|0x1000,False,owner['local_pid'])
assert handle
created,ended,system,user=(wintypes.FILETIME() for _ in range(4))
assert kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(ended),ctypes.byref(system),ctypes.byref(user))
birth=datetime.datetime.fromtimestamp(((created.dwHighDateTime<<32)|created.dwLowDateTime)/10000000-11644473600,datetime.timezone.utc)
assert abs(birth.timestamp()-datetime.datetime.fromisoformat(owner['creation_time']).timestamp())<.001
record=dict(status='ONE_NORMAL_ARM_QUEUED_AFTER_ORIGINAL_ENGINEERING_AND_ACTUAL_AUDIT',
    local_pid=os.getpid(),time_cst=datetime.datetime.now().astimezone().isoformat(),
    original_observer_pid=owner['local_pid'],original_observer_creation=birth.isoformat(),
    new_remote_pollers=0,normal_training_started=False,preflight_states_used=False)
(root/'NORMAL_QUEUE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
assert kernel.WaitForSingleObject(handle,0xFFFFFFFF)==0 and kernel.CloseHandle(handle)
intake=json.loads((root/'ENGINEERING_INTAKE.json').read_bytes())
assert intake['remote_status']['status']=='complete'
seal=root/'actual_review/EXPERIMENT_AUDIT.seal.json'
while not seal.exists():
    time.sleep(240)
runpy.run_path(str(root/'launch_normal_authorized.py'),run_name='__main__')
record.update(status='NORMAL_CONTROLLER_LAUNCHED_AFTER_ACTUAL_AUDIT',
    launch=str(root/'NORMAL_LAUNCH.json'),ordinary_training_child_witness_pending=True,
    full_goal_complete=False)
(root/'NORMAL_QUEUE_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
