; Inno Setup script. Build: iscc installer\aias.iss  ->  installer\Output\aias-setup-<version>.exe
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6F1C2E7A-4B8D-4E0B-9A51-3C2D7B9E8A14}
AppName=aias
AppVersion={#AppVersion}
AppPublisher=zyx1121
AppPublisherURL=https://github.com/zyx1121/aias
DefaultDirName={autopf}\aias
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; WSL 2 with `wsl --install --no-distribution` needs Windows 10 22H2 or later.
MinVersion=10.0.19045
OutputBaseFilename=aias-setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=aias
SetupLogging=yes

[Files]
Source: "..\scripts\*"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\wsl\*"; DestDir: "{app}\wsl"; Flags: ignoreversion
Source: "..\mcp\*"; DestDir: "{app}\mcp"; Flags: ignoreversion

[UninstallRun]
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\uninstall.ps1"""; \
  Flags: waituntilterminated; RunOnceId: "aias-uninstall"; StatusMsg: "Removing the aias WSL distro..."

[Code]
var
  RebootNeeded: Boolean;
  InstallExitCode: Integer;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
  Params: String;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Setting up WSL, Docker and the engines. This takes a while...';
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\scripts\install.ps1') +
              '" -AppDir "' + ExpandConstant('{app}') + '"';
    if not Exec('powershell.exe', Params, '', SW_SHOW, ewWaitUntilTerminated, Code) then
      Code := -1;
    if Code = 3010 then
      RebootNeeded := True
    else if Code <> 0 then
    begin
      InstallExitCode := Code;
      if not WizardSilent then
        MsgBox('aias setup did not finish (exit code ' + IntToStr(Code) + ').' + #13#10 +
             'Log: ' + ExpandConstant('{commonappdata}\aias\install.log') + #13#10 +
             'Run setup again after fixing the cause.', mbError, MB_OK);
    end;
  end;
end;

// Silent installs report a failed install.ps1 through setup's own exit code.
function GetCustomSetupExitCode(): Integer;
begin
  Result := InstallExitCode;
end;

function NeedRestart(): Boolean;
begin
  Result := RebootNeeded;
end;
