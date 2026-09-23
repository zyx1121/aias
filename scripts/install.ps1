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

# ProgramData lets any user create folders, so a user could plant ours (or a
# folder inside it) and swap the rootfs before import. Create it with a
# protected SYSTEM + Administrators ACL in one step, reset anything inside that
# grants other accounts, and refuse to go on unless every item checks out.
$System = New-Object Security.Principal.SecurityIdentifier 'S-1-5-18'
$Admins = New-Object Security.Principal.SecurityIdentifier 'S-1-5-32-544'
$Me     = [Security.Principal.WindowsIdentity]::GetCurrent().User

function New-LockedAcl {
  $acl = New-Object Security.AccessControl.DirectorySecurity
  $acl.SetAccessRuleProtection($true, $false)
  $acl.SetOwner($Admins)
  foreach ($sid in $System, $Admins) {
    $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
      $sid, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')))
  }
  return $acl
}

function Test-Trusted([string]$sid, [switch]$Owner) {
  if ($sid -eq $System.Value -or $sid -eq $Admins.Value) { return $true }
  if ($Owner) { return $sid -eq $Me.Value }
  # WSL grants its VM (NT VIRTUAL MACHINE\<id>) and a capability SID access to
  # ext4.vhdx. Neither is an account a user can log on as.
  return $sid -like 'S-1-5-83-*' -or $sid -like 'S-1-15-3-*'
}

function Test-Locked($path) {
  $acl = Get-Acl -LiteralPath $path
  if (-not (Test-Trusted $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -Owner)) { return $false }
  foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
    if (-not (Test-Trusted $rule.IdentityReference.Value)) { return $false }
  }
  return $true
}

function Get-Unlocked {
  @(Get-Item -LiteralPath $DataDir) + @(Get-ChildItem -LiteralPath $DataDir -Recurse -Force) |
    Where-Object { -not (Test-Locked $_.FullName) }
}

try {
  if (Test-Path $DataDir) {
    $owner = (Get-Acl $DataDir).GetOwner([Security.Principal.SecurityIdentifier]).Value
    if (-not (Test-Trusted $owner -Owner)) {
      throw "$DataDir exists and is owned by $owner, not an administrator. Remove it and run setup again."
    }
    Set-Acl -LiteralPath $DataDir -AclObject (New-LockedAcl)
    # Reset only what fails the check, so WSL's own grants on ext4.vhdx stay.
    foreach ($item in @(Get-Unlocked)) {
      & icacls.exe $item.FullName /setowner '*S-1-5-32-544' /C /Q | Out-Null
      & icacls.exe $item.FullName /reset /C /Q | Out-Null
    }
  } else {
    [IO.Directory]::CreateDirectory($DataDir, (New-LockedAcl)) | Out-Null
  }
  $bad = @(Get-Unlocked) | Select-Object -First 1
  if ($bad) { throw "$($bad.FullName) is open to accounts other than SYSTEM and Administrators." }
} catch {
  Write-Host "Setup stopped: $_" -ForegroundColor Red
  exit 1
}
# Start every run with an empty cache: a handle opened on an old file before
# the ACL was tightened would still be valid.
if (Test-Path $CacheDir) { Remove-Item -LiteralPath $CacheDir -Recurse -Force }
New-Item -ItemType Directory -Force $CacheDir | Out-Null
Start-Transcript -Path (Join-Path $DataDir 'install.log') -Append | Out-Null

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Get-NativeExit {
  # Exit code of a probe whose output we discard. With the script-wide 'Stop',
  # Windows PowerShell turns any stderr line of a native command into an error.
  $ErrorActionPreference = 'Continue'
  & $args[0] @($args | Select-Object -Skip 1) *> $null
  return $LASTEXITCODE
}

function Invoke-Wsl {
  # --exec skips the distro's shell, so arguments arrive exactly as given.
  & wsl.exe -d $Distro -u root --exec @args
  if ($LASTEXITCODE -ne 0) { throw "wsl $($args -join ' ') exited $LASTEXITCODE" }
}

function Test-WslReady {
  $vmp = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
  if ($vmp.State -ne 'Enabled') { return $false }
  return (Get-NativeExit wsl.exe --version) -eq 0
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
    if ((Get-NativeExit wsl.exe -d $d -u root --exec pgrep -x dockerd) -eq 0) {
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
    if ((Get-NativeExit wsl.exe -d $Distro -u root --exec systemctl is-system-running --wait) -ne 255) { break }  # 255 means the distro is not up yet
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
