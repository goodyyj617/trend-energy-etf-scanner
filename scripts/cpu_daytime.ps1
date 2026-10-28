# Daytime CPU limit for long research runs (user request 2026-10-28).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\cpu_daytime.ps1 -Match combo_f_study
#
# Every 5 minutes finds the python processes whose command line contains -Match, plus all their
# descendants (multiprocessing workers), and sets:
#   daytime (DayStart <= hour < DayEnd, local time): priority BelowNormal, affinity = the last DayCores
#                                                    logical cores
#   night:                                           priority Normal, all cores
# The run itself is not stopped or restarted. Exits when no matching process is left.
# Use with scripts/crsp_remote.py: start the run, then start this script with -Match <study module>
# (e.g. -Match lab.stage_c_study) in a second window or as a hidden process.
param(
    [Parameter(Mandatory = $true)][string]$Match,
    [int]$DayStart = 10,
    [int]$DayEnd = 21,
    [int]$DayCores = 8,
    [int]$EverySeconds = 300
)

$n = [Environment]::ProcessorCount
$all = [IntPtr]::new([long]([math]::Pow(2, $n) - 1))
$dayMask = [IntPtr]::new([long](([math]::Pow(2, $DayCores) - 1) * [math]::Pow(2, $n - $DayCores)))

function Get-Tree {
    $procs = Get-CimInstance Win32_Process
    $ids = @($procs | Where-Object { $_.CommandLine -and $_.CommandLine -like "*$Match*" -and $_.ProcessId -ne $PID } |
        ForEach-Object { $_.ProcessId })
    $seen = @{}
    while ($ids.Count) {
        $next = @()
        foreach ($i in $ids) {
            if (-not $seen.ContainsKey($i)) {
                $seen[$i] = $true
                $next += @($procs | Where-Object { $_.ParentProcessId -eq $i } | ForEach-Object { $_.ProcessId })
            }
        }
        $ids = $next
    }
    $seen.Keys
}

while ($true) {
    $tree = @(Get-Tree)
    if (-not $tree.Count) { break }
    $h = (Get-Date).Hour
    $day = ($h -ge $DayStart) -and ($h -lt $DayEnd)
    foreach ($id in $tree) {
        try {
            $p = Get-Process -Id $id -ErrorAction Stop
            if ($day) { $p.PriorityClass = 'BelowNormal'; $p.ProcessorAffinity = $dayMask }
            else { $p.PriorityClass = 'Normal'; $p.ProcessorAffinity = $all }
        } catch {}
    }
    "{0} {1}: {2} processes" -f (Get-Date -Format 'HH:mm'), $(if ($day) { 'day' } else { 'night' }), $tree.Count
    Start-Sleep -Seconds $EverySeconds
}
