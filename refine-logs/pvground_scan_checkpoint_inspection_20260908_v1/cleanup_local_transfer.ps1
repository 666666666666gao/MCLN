$ErrorActionPreference = 'Stop'
$archiveRoot = 'C:\Users\gb\.codex_mcln_g0_20260905\refine-logs\pvground_scan_checkpoint_inspection_20260908_v1'
$transfer = Get-Content -Raw -LiteralPath (Join-Path $archiveRoot 'transfer_receipt.json') | ConvertFrom-Json
$verification = Get-Content -Raw -LiteralPath (Join-Path $archiveRoot 'local_verification.json') | ConvertFrom-Json
if ($transfer.status -ne 'complete' -or $verification.status -ne 'pass' -or $verification.cpu_inventory_exit -ne 0) { throw 'Verified remote copy required' }
$expectedSha = '6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3'
if ($transfer.sha256 -ne $expectedSha -or $transfer.bytes -ne 830036622) { throw 'Remote identity mismatch' }
$transferRoot = (Resolve-Path -LiteralPath 'C:\Users\gb\.codex\tmp\mcln_pv_scan_transfer_20260908').ProviderPath
$target = Get-Item -LiteralPath (Join-Path $transferRoot 'PV-Ground_ScanRefer.pth')
$resolved = (Resolve-Path -LiteralPath $target.FullName).ProviderPath
if ($target.Directory.FullName -ne $transferRoot -or $resolved -ne (Join-Path $transferRoot 'PV-Ground_ScanRefer.pth') -or ($target.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected cleanup target' }
$localSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolved).Hash.ToLowerInvariant()
if ($localSha -ne $expectedSha -or $target.Length -ne $transfer.bytes) { throw 'Local identity mismatch' }
$before = (Get-PSDrive -Name C).Free
Remove-Item -LiteralPath $resolved
if (Test-Path -LiteralPath $resolved) { throw 'Temporary checkpoint was not removed' }
$receipt = [ordered]@{
    status = 'complete'; time_cst = [DateTimeOffset]::Now.ToString('o'); deleted_file = $resolved
    deleted_bytes = $target.Length; sha256 = $localSha; remote_file_retained = $transfer.remote_file
    remote_copy_verified = $true; files_removed = 1; local_free_before = $before
    local_free_after = (Get-PSDrive -Name C).Free; protected_files_modified = 0
}
$json = $receipt | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $archiveRoot 'local_transfer_cleanup.json'), $json + "`n", [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $archiveRoot 'cleanup_local_transfer.ps1')
$json
