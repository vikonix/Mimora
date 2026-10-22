; Inno Setup script for the Mimora Windows installer.
; Compiled by .github/workflows/windows-installer.yml, which puts uv.exe and
; vc_redist into installer\build\ and passes /DAppVersion.
;
; The installer is small on purpose: it does not contain Mimora, Python, torch
; or the models. install.cmd installs mimora==AppVersion from PyPI with uv, so
; the installer gets the same wheel as every other way of installation, and
; the computer gets the GPU build of torch that a frozen executable cannot
; carry.

#ifndef AppVersion
  #error AppVersion is not defined. Pass /DAppVersion=x.y.z to ISCC.
#endif

; A test build only: put a wheel into installer\build and pass
; /DLocalWheel=<its file name>. The installer then carries that wheel and
; installs it instead of the version on PyPI, so the installer and the wheel
; can be checked together before the upload. The file name ends in LOCAL, and
; such a file is never published: its wheel is not the one on PyPI. The
; workflow never passes LocalWheel.

[Setup]
; Do not change AppId: Windows uses it to find the previous installation.
AppId={{8E3B6C4A-5D2F-4B7A-9C1E-6F0A2D4B8E17}
AppName=Mimora
AppVersion={#AppVersion}
AppPublisher=Valeriy Kovalev
AppPublisherURL=https://github.com/vikonix/Mimora
; The version resource of the Setup file. An executable file without one is
; a sign that antivirus programs count against the file.
VersionInfoCompany=Valeriy Kovalev
VersionInfoProductName=Mimora
VersionInfoDescription=Mimora Setup
VersionInfoCopyright=Copyright (c) 2026 Valeriy Kovalev
VersionInfoTextVersion={#AppVersion}
VersionInfoProductTextVersion={#AppVersion}
; The numeric field accepts digits and dots only. A pre-release version
; (1.2.0rc1, 1.2.0.dev1, 1.2.0.post1) contains one of these letters and
; would stop the compiler, so the field then keeps its default 0.0.0.0.
#if Pos("a", AppVersion) + Pos("b", AppVersion) + Pos("c", AppVersion) + Pos("d", AppVersion) + Pos("p", AppVersion) == 0
VersionInfoVersion={#AppVersion}
#endif
; The same name as RUNNING_MUTEX_NAME in mimora\cli.py, which the running
; application holds. Setup and the uninstaller ask to close Mimora first:
; Windows does not let uv replace or delete the files of a running program.
AppMutex=vikonix.Mimora.Running
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
#ifdef LocalWheel
OutputBaseFilename=Mimora-{#AppVersion}-windows-x64-setup-LOCAL
#else
OutputBaseFilename=Mimora-{#AppVersion}-windows-x64-setup
#endif
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

[Files]
; torch and llama-server do not load without the Visual C++ runtime, and a
; clean Windows does not have it.
Source: "build\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: VCRuntimeMissing
Source: "build\uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "install.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "fetch_models.py"; DestDir: "{app}"; Flags: ignoreversion
#ifdef LocalWheel
Source: "build\{#LocalWheel}"; DestDir: "{app}"; Flags: ignoreversion
#endif
; For the shortcuts and the uninstall entry. The copy inside the package is
; deep in the uv tool environment, and its path is not stable.
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

// The HTTP status of the PyPI page of this version, or 0 when PyPI could not
// be asked (no network, a proxy that WinHTTP does not know, no WinHTTP).
function PyPIStatus: Integer;
var
  Request: Variant;
begin
  try
    Request := CreateOleObject('WinHttp.WinHttpRequest.5.1');
    Request.Open('GET', 'https://pypi.org/pypi/mimora/{#AppVersion}/json', False);
    // Resolve, connect, send and receive, in milliseconds.
    Request.SetTimeouts(10000, 10000, 10000, 30000);
    Request.Send('');
    Result := Request.Status;
  except
    Result := 0;
  end;
end;

// Runs before Setup changes anything. install.cmd installs
// mimora=={#AppVersion} from PyPI, so without that version on PyPI the
// installation cannot succeed: Setup stops here, and no files, uninstaller,
// shortcuts or Visual C++ runtime are installed. Only a 404 stops it. When
// PyPI cannot be asked, the installation continues, and uv reports the cause
// (it can also use a package index mirror that this check does not know).
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
#ifdef LocalWheel
  // A test build installs its own wheel, so PyPI is not asked.
  Exit;
#endif
  if PyPIStatus = 404 then
    Result := 'This installer is for Mimora {#AppVersion}, and this version '
      + 'is not available for download. Nothing was installed.' + #13#10
      + #13#10 + 'Download the current installer from '
      + 'https://github.com/vikonix/Mimora/releases';
end;

// What install.cmd installs: mimora=={#AppVersion} from PyPI, or in a test
// build the wheel that Setup copied into {app}. Always in quotation marks:
// cmd splits a batch argument at "=", and {app} can contain spaces.
function InstallSpec: String;
begin
#ifdef LocalWheel
  Result := '"' + ExpandConstant('{app}\{#LocalWheel}') + '"';
#else
  Result := '"mimora=={#AppVersion}"';
#endif
end;

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

// The user data directory, found the way mimora/paths.py finds it for an
// installed Mimora: MIMORA_HOME when it is set (one pair of surrounding
// quotation marks removed, as paths.py does: cmd keeps them in the value),
// otherwise %APPDATA%\Mimora. FromEnv tells which of the two it is.
// A relative MIMORA_HOME is returned as it is: paths.py resolves it against
// the working directory of the application, not of Setup.
function DataDir(var FromEnv: Boolean): String;
var
  Value: String;
begin
  Value := Trim(GetEnv('MIMORA_HOME'));
  if (Length(Value) >= 2) and (Value[1] = Value[Length(Value)])
      and ((Value[1] = '"') or (Value[1] = '''')) then
    Value := Trim(Copy(Value, 2, Length(Value) - 2));
  FromEnv := Value <> '';
  if FromEnv then
    Result := Value
  else
    Result := ExpandConstant('{userappdata}\Mimora');
end;

// The Ready page also names the directory that receives the models, before
// several gigabytes go there. When MIMORA_HOME is set, the user sees that
// Setup uses it. The default text of the page stays: the Memo* parameters
// are its parts, and an empty part is left out.
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo,
  MemoTasksInfo: String): String;
var
  FromEnv: Boolean;
  Dir: String;
begin
  Dir := DataDir(FromEnv);
  Result := '';
  if MemoDirInfo <> '' then
    Result := Result + MemoDirInfo + NewLine + NewLine;
  if MemoGroupInfo <> '' then
    Result := Result + MemoGroupInfo + NewLine + NewLine;
  Result := Result + 'Models, settings and logs:' + NewLine + Space + Dir;
  if FromEnv then
    Result := Result + NewLine + Space
      + '(from the MIMORA_HOME environment variable)';
  if MemoTasksInfo <> '' then
    Result := Result + NewLine + NewLine + MemoTasksInfo;
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
  // The first argument is what install.cmd installs (InstallSpec).
  InstallFailed := RunAndWait(ExpandConstant('{cmd}'),
    '/C ""' + ExpandConstant('{app}\install.cmd') + '" ' + InstallSpec + ' '
    + SelectedComponents('') + '"',
    'Downloading Mimora, Python, the dependencies and the selected models. '
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
    // Setup has already copied its own files and the uninstaller; after a
    // failed upgrade the previous version also stays. The uninstaller
    // removes both.
    WizardForm.FinishedLabel.Caption :=
      'The download of Mimora, Python or the dependencies failed. Make sure '
      + 'that the computer is connected to the internet, then start this '
      + 'installer again.' + #13#10 + #13#10
      + 'To remove the files that were copied, use "Uninstall Mimora" in the '
      + 'Start menu.';
  end;
end;

// True for "C:\...", "C:/..." and a UNC path. Anything else is relative.
function IsAbsolutePath(Path: String): Boolean;
begin
  Result := ((Length(Path) >= 3) and (Path[2] = ':')
      and ((Path[3] = '\') or (Path[3] = '/')))
    or (Copy(Path, 1, 2) = '\\') or (Copy(Path, 1, 2) = '//');
end;

// Deletes Dir when it exists. False only when a part of it stays.
function DelTreeIfExists(Dir: String): Boolean;
begin
  Result := (not DirExists(Dir)) or DelTree(Dir, True, True, True);
end;

// The models, settings and logs are in the user data directory, not in {app}.
// The uninstaller asks about the same directory that the Ready page of Setup
// named (DataDir), read again now: MIMORA_HOME when it is set, otherwise
// %APPDATA%\Mimora. It asks before it deletes: the directory holds several
// gigabytes of downloads that a later installation would use again, and a
// Mimora installed from PyPI on the same computer uses the same directory.
// A silent uninstallation keeps the data (the default answer is No).
//
// %APPDATA%\Mimora belongs to Mimora and is deleted as a whole. A MIMORA_HOME
// directory was chosen by the user and can hold other files (it can even be
// a drive root), so only the subdirectories that Mimora creates
// (mimora/paths.py) are deleted, and the directory itself only when it is
// empty after that. The variable stays: Setup did not set it.
// A relative MIMORA_HOME is not asked about: the shortcuts start Mimora in
// {app}, so its data is inside {app}, which [UninstallDelete] removes.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  FromEnv, Complete: Boolean;
  Dir, Root: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  Dir := DataDir(FromEnv);
  if FromEnv and not IsAbsolutePath(Dir) then
    Exit;
  if not DirExists(Dir) then
    Exit;

  if not FromEnv then
  begin
    if SuppressibleMsgBox(
        'Also delete the downloaded models, the settings and the logs?'
        + #13#10 + #13#10 + Dir, mbConfirmation, MB_YESNO, IDNO) <> IDYES then
      Exit;
    if not DelTree(Dir, True, True, True) then
      MsgBox('Some files could not be deleted:' + #13#10 + Dir,
        mbInformation, MB_OK);
    Exit;
  end;

  Root := AddBackslash(Dir);
  if SuppressibleMsgBox(
      'Also delete the downloaded models, the settings and the logs?'
      + #13#10 + #13#10 + Dir + #13#10
      + '(from the MIMORA_HOME environment variable)' + #13#10 + #13#10
      + 'Only these folders of Mimora are deleted, other files stay:'
      + #13#10 + 'config, models, model_cache, bin\llama, logs',
      mbConfirmation, MB_YESNO, IDNO) <> IDYES then
    Exit;
  // Each call runs: one failure does not stop the others.
  Complete := DelTreeIfExists(Root + 'config');
  Complete := DelTreeIfExists(Root + 'models') and Complete;
  Complete := DelTreeIfExists(Root + 'model_cache') and Complete;
  Complete := DelTreeIfExists(Root + 'bin\llama') and Complete;
  Complete := DelTreeIfExists(Root + 'logs') and Complete;
  // RemoveDir removes only an empty directory: bin can hold other programs,
  // and the data directory other files.
  RemoveDir(Root + 'bin');
  RemoveDir(Dir);
  if Complete then
    MsgBox('The data of Mimora was deleted. The MIMORA_HOME environment '
      + 'variable still exists; remove it if no other program uses it.',
      mbInformation, MB_OK)
  else
    MsgBox('Some files could not be deleted:' + #13#10 + Dir,
      mbInformation, MB_OK);
end;
