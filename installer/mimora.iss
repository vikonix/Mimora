; Inno Setup script for the Mimora Windows installer.
; Compiled by .github/workflows/windows-installer.yml, which puts uv.exe and
; the wheel into installer\build\ and passes /DAppVersion.
;
; The installer is small on purpose: it does not contain Python, torch or the
; models. install.cmd downloads them with uv, so the computer keeps the GPU
; build of torch that a frozen executable cannot carry.

#ifndef AppVersion
  #error AppVersion is not defined. Pass /DAppVersion=x.y.z to ISCC.
#endif

[Setup]
; Do not change AppId: Windows uses it to find the previous installation.
AppId={{8E3B6C4A-5D2F-4B7A-9C1E-6F0A2D4B8E17}
AppName=Mimora
AppVersion={#AppVersion}
AppPublisher=Valeriy Kovalev
AppPublisherURL=https://github.com/vikonix/Mimora
; Per-user installation: no administrator rights, and uv can write into {app}.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\Mimora
DefaultGroupName=Mimora
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
; The same file that the application windows use (mimora\window_icon.py).
SetupIconFile=..\mimora\icons\mimora.ico
; The icon in the "Installed apps" list of Windows.
UninstallDisplayIcon={app}\mimora.ico
OutputDir=Output
OutputBaseFilename=Mimora-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
; Inno Setup 6.7 and later start Setup with RedirectionGuard, and the programs
; that Setup starts get it too. uv creates a directory junction for its Python
; without administrator rights, and Windows then refuses the path with error
; 448 (untrusted mount point). The protection is for elevated processes; this
; Setup is not elevated. Older compilers do not know the directive.
#if Ver >= EncodeVer(6,7,0)
RedirectionGuard=no
#endif

[InstallDelete]
; An upgrade must not leave the old wheel: install.cmd takes the wheel by mask.
Type: files; Name: "{app}\*.whl"

[Files]
; torch and llama-server do not load without the Visual C++ runtime, and a
; clean Windows does not have it.
Source: "build\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: VCRuntimeMissing
Source: "build\uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "build\*.whl"; DestDir: "{app}"; Flags: ignoreversion
Source: "install.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "fetch_models.py"; DestDir: "{app}"; Flags: ignoreversion
; For the shortcuts and the uninstall entry. The copy inside the wheel is deep
; in the uv tool environment, and its path is not stable.
Source: "..\mimora\icons\mimora.ico"; DestDir: "{app}"; Flags: ignoreversion

[Tasks]
; Each check box is the user's agreement to that download. A component that
; is not selected here is offered again by the first-run window of Mimora.
; The names are the arguments of fetch_models.py.
Name: "core"; Description: "Speech models for English: recognition and voice (1.6 GB)"; GroupDescription: "Download now:"
Name: "chat"; Description: "Chat model for phrase generation (2.0 to 2.7 GB)"; GroupDescription: "Download now:"
Name: "translator"; Description: "Offline translator (2.5 GB)"; GroupDescription: "Download now:"; Flags: unchecked
Name: "spanish"; Description: "Spanish voice (0.4 GB)"; GroupDescription: "Download now:"; Flags: unchecked
Name: "acoustic"; Description: "Model of the alternative acoustic engine (1.3 GB)"; GroupDescription: "Download now:"; Flags: unchecked

[Icons]
; mimora-gui.exe starts without a console window. The second shortcut keeps
; the console, because that is the only place a startup error is visible.
; AppUserModelID is the same value as APP_USER_MODEL_ID in
; mimora\window_icon.py. With different values, a pinned shortcut and the
; running window become two separate taskbar buttons.
Name: "{group}\Mimora"; Filename: "{app}\bin\mimora-gui.exe"; WorkingDir: "{app}"; IconFilename: "{app}\mimora.ico"; AppUserModelID: "vikonix.Mimora"
Name: "{group}\Mimora (console, for diagnostics)"; Filename: "{app}\bin\mimora.exe"; WorkingDir: "{app}"; IconFilename: "{app}\mimora.ico"
Name: "{group}\Uninstall Mimora"; Filename: "{uninstallexe}"

; There is no [Run] section: a [Run] entry cannot read the exit code of its
; program. CurStepChanged in [Code] starts install.cmd and checks the result.

[UninstallDelete]
; Removes the uv tool environment, the private Python and the uv cache.
; The user data directory is outside {app}: CurUninstallStepChanged in
; [Code] asks about it.
Type: filesandordirs; Name: "{app}"

[Code]
// True when a runtime DLL that torch links against is absent. The same
// three files that step_check_vcredist in install.py tries to load.
function VCRuntimeMissing: Boolean;
begin
  Result := not (FileExists(ExpandConstant('{sys}\vcruntime140.dll'))
    and FileExists(ExpandConstant('{sys}\vcruntime140_1.dll'))
    and FileExists(ExpandConstant('{sys}\msvcp140.dll')));
end;

// The name of a task when the user selected it, with a separator after it.
function ComponentIfSelected(Name: String): String;
begin
  if WizardIsTaskSelected(Name) then
    Result := Name + ' '
  else
    Result := '';
end;

// The selected tasks as one argument list for install.cmd, for example
// "core chat translator". An empty list means "download nothing".
function SelectedComponents(Param: String): String;
begin
  Result := ComponentIfSelected('core') + ComponentIfSelected('chat')
    + ComponentIfSelected('translator') + ComponentIfSelected('spanish')
    + ComponentIfSelected('acoustic');
end;

var
  // True when install.cmd did not end with exit code 0.
  InstallFailed: Boolean;

// Starts a program with a visible window and waits for it. Returns its exit
// code, or -1 when the program did not start.
function RunAndWait(Filename, Params, Status: String): Integer;
var
  ExitCode: Integer;
begin
  WizardForm.StatusLabel.Caption := Status;
  if Exec(Filename, Params, ExpandConstant('{app}'), SW_SHOW,
      ewWaitUntilTerminated, ExitCode) then
    Result := ExitCode
  else
    Result := -1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;

  // First, because install.cmd imports torch for the hardware probe. The
  // redistributable asks for administrator rights itself (one UAC prompt).
  if VCRuntimeMissing then
    RunAndWait(ExpandConstant('{tmp}\vc_redist.x64.exe'),
      '/install /passive /norestart',
      'Installing the Microsoft Visual C++ runtime...');

  // The console window stays visible: the download is several gigabytes, and
  // the progress output is the only sign that the installation continues.
  // The outer pair of quotation marks is for cmd.exe, which removes it.
  InstallFailed := RunAndWait(ExpandConstant('{cmd}'),
    '/C ""' + ExpandConstant('{app}\install.cmd') + '" '
    + SelectedComponents('') + '"',
    'Downloading Python, the dependencies and the selected models. '
    + 'This can take a long time...') <> 0;

  // [Icons] has already created the shortcuts. Without the program they
  // point at nothing. After a failed upgrade the previous program can still
  // be there, and its shortcuts stay.
  if InstallFailed
      and not FileExists(ExpandConstant('{app}\bin\mimora-gui.exe')) then
  begin
    DeleteFile(ExpandConstant('{group}\Mimora.lnk'));
    DeleteFile(ExpandConstant('{group}\Mimora (console, for diagnostics).lnk'));
  end;
end;

// The last page must not report success after install.cmd failed.
procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpFinished) and InstallFailed then
  begin
    WizardForm.FinishedHeadingLabel.Caption := 'Mimora was not installed';
    WizardForm.FinishedLabel.Caption :=
      'The download of Python or of the dependencies failed. Make sure that '
      + 'the computer is connected to the internet and that Mimora is not '
      + 'running, then start this installer again.';
  end;
end;

// The models, settings and logs are in the user data directory, not in {app}.
// The uninstaller asks before it deletes them: the directory holds several
// gigabytes of downloads that a later installation would use again, and a
// Mimora installed from PyPI on the same computer uses the same directory.
// A silent uninstallation keeps the data (the default answer is No).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  DataDir := ExpandConstant('{userappdata}\Mimora');
  if not DirExists(DataDir) then
    Exit;
  if SuppressibleMsgBox(
      'Also delete the downloaded models, the settings and the logs?' + #13#10
      + #13#10 + DataDir, mbConfirmation, MB_YESNO, IDNO) <> IDYES then
    Exit;
  if not DelTree(DataDir, True, True, True) then
    MsgBox('Some files could not be deleted:' + #13#10 + DataDir,
      mbInformation, MB_OK);
end;
