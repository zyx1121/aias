#Requires -RunAsAdministrator
<#
  Removes everything install.ps1 created: the aias distro (Docker images,
  models and caches live inside its virtual disk), its scheduled tasks and
  C:\ProgramData\aias. WSL itself and other distros are left alone.
#>
$ErrorActionPreference = 'Continue'
$env:WSL_UTF8 = '1'

$Distro  = 'aias'
$DataDir = Join-Path $env:ProgramData 'aias'

# Exact names only: other tools on a machine may use an aias- prefix.
foreach ($task in 'aias-wsl', 'aias-resume') {
  if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $task -Confirm:$false
    Write-Host "Removed scheduled task $task"
  }
}

if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
  $distros = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { $_.Trim() })
  if ($distros -contains $Distro) {
    & wsl.exe --terminate $Distro | Out-Null
    & wsl.exe --unregister $Distro
    Write-Host "Removed WSL distro $Distro"
  }
}

if (Test-Path $DataDir) {
  Remove-Item $DataDir -Recurse -Force
  Write-Host "Removed $DataDir"
}
exit 0
