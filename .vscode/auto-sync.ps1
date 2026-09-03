$ErrorActionPreference = 'Stop'
$git = 'C:\Program Files\Git\cmd\git.exe'
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

$ignoredPaths = @(
    '\.git\',
    '\.venv\',
    '\__pycache__\',
    '\.pytest_cache\',
    '\mlruns\',
    '\.streamlit\'
)

function Get-ProjectChanges {
    $status = & $git status --porcelain
    $changes = foreach ($line in $status) {
        $path = $line.Substring(3).Trim('"')
        if ($ignoredPaths | Where-Object { $path -like "*$_*" }) {
            continue
        }
        if ($path -like 'data/*.db') {
            continue
        }
        $line
    }
    return @($changes)
}

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $workspace
$watcher.IncludeSubdirectories = $true
$watcher.Filter = '*.*'
$watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite -bor [System.IO.NotifyFilters]::FileName -bor [System.IO.NotifyFilters]::Size
Register-ObjectEvent -InputObject $watcher -EventName Changed -SourceIdentifier AutoSyncChanged | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Created -SourceIdentifier AutoSyncCreated | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Deleted -SourceIdentifier AutoSyncDeleted | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Renamed -SourceIdentifier AutoSyncRenamed | Out-Null
$watcher.EnableRaisingEvents = $true

$lastChange = Get-Date
$pending = $false

while ($true) {
    $event = Wait-Event -Timeout 2
    if ($null -ne $event) {
        Remove-Event -EventIdentifier $event.EventIdentifier
        $pending = $true
        $lastChange = Get-Date
    }

    if ($pending -and ((Get-Date) - $lastChange).TotalSeconds -ge 3) {
        $pending = $false
        $changes = @(Get-ProjectChanges)
        if ($changes.Count -gt 0) {
            & $git add -A
            & $git reset -- api/__pycache__ data/*.db 2>$null
            $staged = & $git diff --cached --name-only
            if ($staged) {
                & $git commit -m 'Auto-sync project changes'
                & $git push origin HEAD:main
            }
        }
    }
}
