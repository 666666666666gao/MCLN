$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskQueueRoot = 'C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_local_visual_mesh_acceptance_queue_20260906_v1'
& uv run --no-project --with paramiko python "$taskQueueRoot/continue_after_formal.py" *> "$taskQueueRoot/queue.txt"
$taskQueueExit = $LASTEXITCODE
[System.IO.File]::WriteAllText("$taskQueueRoot/controller.exit", "$taskQueueExit`n")
exit $taskQueueExit
