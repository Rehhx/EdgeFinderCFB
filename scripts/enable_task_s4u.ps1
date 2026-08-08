# Make the CFB scheduled tasks run WHETHER OR NOT the user is logged on.
#
#   RUN THIS FROM AN ELEVATED POWERSHELL (right-click -> Run as administrator):
#       powershell -ExecutionPolicy Bypass -File scripts\enable_task_s4u.ps1
#
# WHY IT NEEDS ADMIN: both tasks live in the root Task Scheduler folder (\),
# and changing a task's principal there requires elevation. A non-elevated
# session gets "Access is denied" (HRESULT 0x80070005).
#
# WHY S4U RATHER THAN A STORED PASSWORD:
#   S4U ("service for user") runs the task whether or not you are logged on
#   WITHOUT storing your password anywhere. Its only real limitation is that
#   the task cannot reach network resources that need YOUR credentials —
#   mapped drives, SMB shares. These tasks only make outbound HTTPS calls
#   (CFBD, The Odds API, Ourlads, GitHub), so S4U is sufficient and strictly
#   safer than keeping a password in the task store.
#
# If you ever DO need credentialed network access, the alternative is:
#   Set-ScheduledTask -TaskName $t -User $env:USERNAME -Password '<password>'
# which stores the password. Not needed here — do not use it casually.

$tasks = @("CFB August Refit", "CFB Splits Capture", "CFB Depth Charts")

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$pr = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "NOT ELEVATED. Re-run this from an administrator PowerShell." -ForegroundColor Red
    exit 1
}

foreach ($t in $tasks) {
    $existing = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if (-not $existing) { Write-Host "skip (not found): $t"; continue }

    $before = $existing.Principal.LogonType
    $next   = (Get-ScheduledTaskInfo -TaskName $t).NextRunTime

    $prn = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
                                      -LogonType S4U -RunLevel Limited
    Set-ScheduledTask -TaskName $t -Principal $prn | Out-Null

    $after     = (Get-ScheduledTask -TaskName $t).Principal.LogonType
    $nextAfter = (Get-ScheduledTaskInfo -TaskName $t).NextRunTime
    Write-Host ("{0,-22} LogonType {1} -> {2} | next run {3}" -f `
                $t, $before, $after, $nextAfter)
    if ($next -ne $nextAfter) {
        Write-Host "  [!] next run time CHANGED - re-check the trigger" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# Task Scheduler's Operational log is DISABLED by default, so a task that
# silently never fires leaves no trace at all. On 2026-08-07 the depth-chart
# task's Friday 10:00 trigger did not run and there was no way to tell why:
# the log had 0 records. Enabling it also needs elevation, so it rides along
# with the S4U change rather than costing a second admin prompt.
$log = "Microsoft-Windows-TaskScheduler/Operational"
& wevtutil.exe sl $log /e:true
if ($LASTEXITCODE -eq 0) {
    Write-Host "enabled event log: $log"
} else {
    Write-Host "[!] could not enable $log (exit $LASTEXITCODE)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Verify with:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTask | ? TaskName -like 'CFB*' |" -ForegroundColor Cyan
Write-Host "    Select TaskName, @{n='LogonType';e={`$_.Principal.LogonType}}, State" -ForegroundColor Cyan
Write-Host ""
Write-Host "S4U = runs whether or not you are logged on, no password stored."
Write-Host "The refit still needs the machine POWERED ON at the trigger time;"
Write-Host "StartWhenAvailable=True makes it catch up if the machine was off."
