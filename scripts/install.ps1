#Requires -RunAsAdministrator
<#
  Sets up aias: a dedicated WSL distro with Docker, Ollama, vLLM and an MCP
  server on http://127.0.0.1:11400/mcp. Safe to run again; each step skips
  what is already in place.

  Exit codes: 0 done, 3010 reboot needed (setup resumes at next logon), 1 failed.
#>
param([string]$AppDir = (Split-Path $PSScriptRoot -Parent))

$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'  # plain UTF-8 from wsl.exe instead of UTF-16

$Distro     = 'aias'
$DataDir    = Join-Path $env:ProgramData 'aias'
$DistroDir  = Join-Path $DataDir 'wsl'
$CacheDir   = Join-Path $DataDir 'cache'
$KeepAlive  = 'aias-wsl'
$Resume     = 'aias-resume'
$RootfsBase = 'https://cloud-images.ubuntu.com/wsl/releases/24.04/current'
$RootfsName = 'ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz'

New-Item -ItemType Directory -Force $DataDir, $CacheDir | Out-Null
Start-Transcript -Path (Join-Path $DataDir 'install.log') -Append | Out-Null

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Invoke-Wsl {
  # --exec skips the distro's shell, so arguments arrive exactly as given.
  & wsl.exe -d $Distro -u root --exec @args
  if ($LASTEXITCODE -ne 0) { throw "wsl $($args -join ' ') exited $LASTEXITCODE" }
}

function Test-WslReady {
  $vmp = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
  if ($vmp.State -ne 'Enabled') { return $false }
  & wsl.exe --version *> $null
  return $LASTEXITCODE -eq 0
}

function Register-Resume {
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -AppDir `"$AppDir`""
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
  Register-ScheduledTask -TaskName $Resume -Action $action -Trigger $trigger -Principal $principal -Force | Out-Null
}

try {
  if (Get-ScheduledTask -TaskName $Resume -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $Resume -Confirm:$false
  }

  Step 'Checking for an NVIDIA GPU'
  $gpu = Get-CimInstance Win32_VideoController | Where-Object Name -match 'NVIDIA'
  if (-not $gpu) { throw 'No NVIDIA GPU found. aias needs one, with a current driver.' }
  Write-Host ($gpu.Name -join ', ')

  Step 'Checking WSL'
  if (-not (Test-WslReady)) {
    Write-Host 'Installing WSL. Windows must restart once; setup continues after you log in.'
    & wsl.exe --install --no-distribution
    if ($LASTEXITCODE -ne 0) { throw "wsl --install exited $LASTEXITCODE" }
    Register-Resume
    Stop-Transcript | Out-Null
    exit 3010
  }
  # No `wsl --update` here: an update restarts WSL and stops every other distro.

  $distros = @(& wsl.exe --list --quiet | ForEach-Object { $_.Trim() } | Where-Object { $_ })

  # All WSL2 distros share one network namespace, so a second Docker daemon
  # would fight ours over the docker0 bridge and iptables.
  Step 'Checking other distros for Docker'
  $running = @(& wsl.exe --list --running --quiet | ForEach-Object { $_.Trim() } | Where-Object { $_ -and $_ -ne $Distro })
  foreach ($d in $running) {
    & wsl.exe -d $d -u root --exec pgrep -x dockerd *> $null
    if ($LASTEXITCODE -eq 0) {
      throw "Docker is running in WSL distro '$d'. Stop it first: wsl -d $d -u root systemctl disable --now docker.service docker.socket"
    }
  }

  if ($distros -notcontains $Distro) {
    Step 'Downloading Ubuntu 24.04'
    $rootfs = Join-Path $CacheDir $RootfsName
    # curl.exe ships with Windows and is far faster than Invoke-WebRequest.
    $line = & curl.exe -fsSL "$RootfsBase/SHA256SUMS" | Where-Object { $_ -like "*$RootfsName" } | Select-Object -First 1
    if (-not $line) { throw "No checksum for $RootfsName in $RootfsBase/SHA256SUMS" }
    $want = ($line -split '\s+')[0].ToUpper()
    if (-not (Test-Path $rootfs) -or (Get-FileHash $rootfs -Algorithm SHA256).Hash -ne $want) {
      & curl.exe -fL --retry 3 -o $rootfs "$RootfsBase/$RootfsName"
      if ($LASTEXITCODE -ne 0) { throw "Download of $RootfsName failed (curl exit $LASTEXITCODE)" }
      if ((Get-FileHash $rootfs -Algorithm SHA256).Hash -ne $want) { throw "Checksum mismatch for $RootfsName" }
    }

    Step "Creating WSL distro '$Distro'"
    & wsl.exe --import $Distro $DistroDir $rootfs --version 2
    if ($LASTEXITCODE -ne 0) { throw "wsl --import exited $LASTEXITCODE" }
    Remove-Item $rootfs
  }

  Step 'Enabling systemd'
  # Copy files in rather than build them with sh -c: Windows PowerShell drops
  # the inner double quotes when it passes arguments to wsl.exe.
  $src = (& wsl.exe -d $Distro -u root --exec wslpath -a $AppDir).Trim()
  Invoke-Wsl cp "$src/wsl/wsl.conf" /etc/wsl.conf
  if (Get-ScheduledTask -TaskName $KeepAlive -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $KeepAlive  # a rerun: let the restart below pick up wsl.conf
  }
  & wsl.exe --terminate $Distro | Out-Null

  Step 'Keeping the distro running across logoff and reboot'
  $action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument "-d $Distro -u root --exec sleep infinity"
  $trigger = New-ScheduledTaskTrigger -AtStartup
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0
  Register-ScheduledTask -TaskName $KeepAlive -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $KeepAlive
  for ($i = 0; $i -lt 30; $i++) {
    & wsl.exe -d $Distro -u root --exec systemctl is-system-running --wait *> $null
    if ($LASTEXITCODE -ne 255) { break }  # 255 means the distro is not up yet
    Start-Sleep 2
  }

  Step 'Provisioning the distro'
  Invoke-Wsl bash "$src/wsl/bootstrap.sh" $src

  Step 'Checking the MCP server from Windows'
  $ok = $false
  for ($i = 0; $i -lt 15 -and -not $ok; $i++) {
    try { $ok = (Invoke-WebRequest 'http://127.0.0.1:11400/health' -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200 } catch { Start-Sleep 2 }
  }
  if (-not $ok) { throw 'The MCP server runs in WSL but does not answer on 127.0.0.1:11400 from Windows.' }

  Write-Host "`naias is ready." -ForegroundColor Green
  Write-Host 'Add it to Claude Code:  claude mcp add --transport http aias http://127.0.0.1:11400/mcp'
  Stop-Transcript | Out-Null
  exit 0
}
catch {
  Write-Host "`nSetup failed: $_" -ForegroundColor Red
  Write-Host "Log: $(Join-Path $DataDir 'install.log')"
  Stop-Transcript | Out-Null
  exit 1
}
