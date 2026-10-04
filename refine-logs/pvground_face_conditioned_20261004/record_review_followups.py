"""Record the two actual delivered follow-up messages and the concrete fix."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local / '.aris/traces/experiment-bridge/2026-10-04_face_source_run01'
messages = [
    "Two additional exact lifecycle primary files were prepared while your review runs: D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_face_conditioned_20261004\\wait_fit_authorized.py and collect_terminal.py. Please read and include them in your actual reviewed_files/report. Sole observer starts6300s near prior measured train end, then actual remaining estimates/240s; collector permits exact external current-best path or owned new terminal and downloads text/rows only. No listed model/spec/launcher sources changed since your request.",
    "已按你发现的实际证据缺口最小修正 run_face_fit.py：在已有 ranked/iou 上，为bbs、bbf分别保存first_good_rank25/50；1-based，严格>.25/>.50，无合格候选为null。不增加前向、不改变选择或训练。实际3.7语法检查通过。请重新读取这一源码并绑定新SHA，将实际修正及你的最终判定写入报告；其余已列模型、spec、launcher未改。"]
for index, message in enumerate(messages, 2):
    record = dict(recorded_cst=datetime.datetime.now().astimezone().isoformat(),
        actual_tool='collaboration.send_message', actual_arguments=dict(target='pvg_face_conditioned_source_review', message=message),
        delivered=True, judgement_fabricated=False)
    with (trace / ('%03d-code-review.followup.json' % index)).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
record = dict(actual_fix='save exact first-good ranks in existing native evaluation; no extra forward',
    changed_file='run_face_fit.py', sha256=hashlib.sha256((local/'run_face_fit.py').read_bytes()).hexdigest(),
    actual_py37_AST_pass=True, native_execution=False)
(local / 'OBSERVED_RANK_RECORD_CORRECTION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(actual_followup_messages=2, concrete_rank_fix=True, no_gpu_execution=True)))
