$ErrorActionPreference = 'Stop'
$taskArchiveRoot = 'C:\Users\gb\.codex\archives'
$taskRetainedDirectory = Join-Path $taskArchiveRoot 'pvg_superseded_support_20261002_v4'
$taskRetainedPath = Join-Path $taskRetainedDirectory 'wholesale_preflight_delta.pth'
$taskArchiveReceipt = Get-Content -LiteralPath (Join-Path $taskRetainedDirectory 'archive_receipt.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskReference = @($taskArchiveReceipt.files | Where-Object { $_.local -eq 'wholesale_preflight_delta.pth' })
if ($taskReference.Count -ne 1) { throw 'Expected one complete archive reference' }
$taskRetainedHash = (Get-FileHash -LiteralPath $taskRetainedPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskRetainedHash -ne $taskReference[0].sha256 -or (Get-Item -LiteralPath $taskRetainedPath).Length -ne 358019598) { throw 'Complete archive verification failed' }
$taskTargets = @(
    @{ Directory = 'pvg_superseded_support_20261002'; Bytes = 276824064 },
    @{ Directory = 'pvg_superseded_support_20261002_v2'; Bytes = 260407296 },
    @{ Directory = 'pvg_superseded_support_20261002_v3'; Bytes = 25591808 }
)
$taskVerified = @()
foreach ($taskTarget in $taskTargets) {
    $taskPath = Join-Path (Join-Path $taskArchiveRoot $taskTarget.Directory) 'wholesale_preflight_delta.pth'
    $taskResolved = (Resolve-Path -LiteralPath $taskPath).Path
    if (-not $taskResolved.StartsWith($taskArchiveRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion target outside archive workspace' }
    if ((Get-Item -LiteralPath $taskResolved).Length -ne $taskTarget.Bytes) { throw 'Partial archive size changed' }
    $taskVerified += @{ path = $taskResolved; bytes = $taskTarget.Bytes; reason = 'Interrupted partial copy; full verified v4 retained' }
}
foreach ($taskFile in $taskVerified) { Remove-Item -LiteralPath $taskFile.path }
$taskRecord = @{
    time_cst = (Get-Date).ToString('o')
    deletion_executed = $true
    files = $taskVerified
    deleted_count = 3
    deleted_bytes = ($taskVerified | Measure-Object -Property bytes -Sum).Sum
    complete_archive_retained = $taskRetainedPath
    complete_archive_sha256 = $taskRetainedHash
    user_authorization = 'Project disk cleanup; keep best weights and remove obsolete partial copies'
}
$taskReceiptPath = 'C:\Users\gb\.codex\tmp\pvground_candidate_consistency_20261003\local_incomplete_archive_cleanup_receipt.json'
$taskRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReceiptPath -Encoding UTF8
Write-Output ($taskRecord | Select-Object time_cst, deleted_count, deleted_bytes, deletion_executed | ConvertTo-Json)
